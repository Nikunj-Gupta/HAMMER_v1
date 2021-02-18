import os

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from tensorboardX import SummaryWriter

device = torch.device("cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, actor_layer, critic_layer):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        # actor
        layers = [] 
        layers.append(nn.Linear(state_dim, actor_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(actor_layer[1:])): 
            layers.append(nn.Linear(actor_layer[i], actor_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(actor_layer[-1], action_dim)) 
        layers.append(nn.Tanh())
        
        self.actor = nn.Sequential(*layers) 

        # critic
        # critic 
        layers = [] 
        layers.append(nn.Linear(state_dim, critic_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(critic_layer[1:])): 
            layers.append(nn.Linear(critic_layer[i], critic_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(critic_layer[-1], 1)) 
        
        self.critic = nn.Sequential(*layers) 


        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, actor_layer, critic_layer):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std, actor_layer, critic_layer).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std, actor_layer, critic_layer).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, action_logprobs = self.policy_old.act(state)
        return action.cpu().data.numpy().flatten(), action_logprobs.cpu().data.numpy().flatten()

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))

    def update(self, local_memory):
        rewards_list = []
        old_states_list = []
        old_actions_list = []
        old_logprobs_list = []
        n_agents = len(local_memory)
        for i in range(n_agents):
            # Monte Carlo estimate of rewards:
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(local_memory[i].rewards), reversed(local_memory[i].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # making a list for updates.
            rewards_list.append(rewards)
            old_states_list.append(torch.squeeze(torch.tensor(local_memory[i].states, dtype=torch.float32).to(device), 1).detach())
            old_actions_list.append(torch.squeeze(torch.tensor(local_memory[i].actions, dtype=torch.float32).to(device), 1).detach())
            old_logprobs_list.append(torch.squeeze(torch.tensor(local_memory[i].logprobs, dtype=torch.float32).to(device), 1).detach())
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # For each agent:
            for i in range(n_agents):
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_list[i], old_actions_list[i])

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs_list[i].detach())

                # Finding Surrogate Loss:
                advantages = rewards_list[i] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_list[i]) - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
