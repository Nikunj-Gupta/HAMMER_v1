import argparse
import pickle

import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
from tensorboardX import SummaryWriter
from utils import read_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

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
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))

    def update(self, memory):
        rewards_list = []
        old_states_list = []
        old_actions_list = []
        old_logprobs_list = []
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

        for _ in range(self.K_epochs): 
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

