import argparse
import pickle

import torch
import torch.nn as nn
from torch.distributions import Categorical
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
    def __init__(self, state_dim, action_dim, actor_layer, critic_layer):
        super(ActorCritic, self).__init__()

        # actor
        layers = [] 
        layers.append(nn.Linear(state_dim, actor_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(actor_layer[1:])): 
            layers.append(nn.Linear(actor_layer[i], actor_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(actor_layer[-1], action_dim)) 
        layers.append(nn.Softmax(dim=-1))
        
        self.action_layer = nn.Sequential(*layers) 
        
        # critic 
        layers = [] 
        layers.append(nn.Linear(state_dim, critic_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(critic_layer[1:])): 
            layers.append(nn.Linear(critic_layer[i], critic_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(critic_layer[-1], 1)) 
        
        self.value_layer = nn.Sequential(*layers) 
        
    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)

        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample() 

        return action.item(), dist.log_prob(action) 

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() 

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, actor_layer, critic_layer, lr, betas, gamma, K_epochs, eps_clip, shared=True):
        self.lr = lr
        self.shared = shared 
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim=state_dim, action_dim=action_dim, actor_layer=actor_layer, critic_layer=critic_layer).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim=state_dim, action_dim=action_dim, actor_layer=actor_layer, critic_layer=critic_layer).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))

    def update(self, memory, writer=None, i_episode=None):
        rewards_list = []
        old_states_list = []
        old_actions_list = []
        old_logprobs_list = [] 
        if not self.shared: 
            memory = [memory]
        n_agents = len(memory) 

        # Monte Carlo estimate of state rewards:
        for i in range(n_agents):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory[i].rewards), reversed(memory[i].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # making lists to update
            rewards_list.append(rewards)
            old_states_list.append(torch.squeeze(torch.tensor(memory[i].states, dtype=torch.float32).to(device)).detach())
            old_actions_list.append(torch.squeeze(torch.tensor(memory[i].actions).to(device)).detach())
            old_logprobs_list.append(torch.squeeze(torch.tensor(memory[i].logprobs).to(device)).detach()) 

        # Optimize policy for K epochs: 

        for epoch in range(self.K_epochs): 
            for i in range(n_agents):

                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_list[i], old_actions_list[i])

                # Finding the ratio (pi_theta / pi_theta__old): 
                ratios = torch.exp(logprobs - old_logprobs_list[i].detach())

                # Finding Surrogate Loss:
                advantages = rewards_list[i] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                critic_loss = 0.5 * self.MseLoss(state_values, rewards_list[i])
                actor_loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
                loss = actor_loss + critic_loss 

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            # if writer is not None and epoch == self.K_epochs-1:
            #     writer.add_scalar('actor_loss/local_agent', actor_loss.mean(), i_episode)
            #     writer.add_scalar('critic_loss/local_agent', critic_loss.mean(), i_episode)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

