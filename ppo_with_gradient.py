# PPO implementation of the global messenger for HAMMER.
import os

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
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
    def __init__(self, single_state_dim, action_dim, n_agents, actor_layer, critic_layer, is_discrete):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        layers = [] 
        layers.append(nn.Linear(single_state_dim + meslen, actor_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(actor_layer[1:])): 
            layers.append(nn.Linear(actor_layer[i], actor_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(actor_layer[-1], action_dim)) 
        layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*layers) 
        
        # global actor
        layers = [] 
        layers.append(nn.Linear(single_state_dim * n_agents, actor_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(actor_layer[1:])): 
            layers.append(nn.Linear(actor_layer[i], actor_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(actor_layer[-1], meslen)) 
        self.global_actor = nn.Sequential(*layers) 
        
        # critic
        layers = [] 
        layers.append(nn.Linear(state_dim + meslen, critic_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(critic_layer[1:])): 
            layers.append(nn.Linear(critic_layer[i], critic_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(critic_layer[-1], 1)) 
        self.critic = nn.Sequential(*layers) 

        # global critic
        layers = [] 
        layers.append(nn.Linear(state_dim * n_agents, critic_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(critic_layer[1:])): 
            layers.append(nn.Linear(critic_layer[i], critic_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(critic_layer[-1], n_agents)) 
        self.global_critic = nn.Sequential(*layers)
        
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.n_agents = n_agents

    def forward(self):
        raise NotImplementedError

    def act(self, obs, memory):
        global_agent_state = [obs[i] for i in obs]
        global_agent_state = (global_agent_state).reshape((-1,))
        global_actor_message = self.global_actor(global_agent_state)
        global_actor_message = global_actor_message.reshape(self.nagents, self.meslen) 
        action_array = []
        log_prob_array = []
        for i, agent in self.agents:
            local_state = np.concatenate([obs[agent], global_actor_message[i]])
            local_state = torch.from_numpy(local_state).float().to(device)
            action_probs = self.actor(local_state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_array.append(action.item())
            log_prob_array.append(dist.log_prob(action))
        return {agent : action_array[i] for i, agent in enumerate(self.agents)} , {agent : log_prob_array[i] for i, agent in enumerate(self.agents)}  


    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() 

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, agents, single_state_dim, single_action_dim, meslen, n_agents, lr, betas, gamma, K_epochs, eps_clip, actor_layer, critic_layer):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.agents = agents
        self.memory = [Memory() for _ in self.agents]

        self.policy = ActorCritic(single_state_dim, single_action_dim, n_agents, action_std, n_agents, actor_layer, critic_layer, is_discrete).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std, n_agents, actor_layer, critic_layer, is_discrete).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.meslen = meslen

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, action_logprobs = self.policy_old.act(state)
        return action.cpu().data.numpy().squeeze(), action_logprobs.cpu().data.numpy().squeeze()

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))

    def memory_record(self, obs, actions, log_probs, rewards, is_terminals):
        for i, agent in enumerate(self.agents):
            self.memory[i].states.append(obs[agent])
            self.memory[i].actions.append(actions[agent])
            self.memory[i].logprobs.append(log_probs[agent])
            self.memory[i].rewards.append(rewards[agent])
            self.memory[i].is_terminals.append(is_terminals[agent])

    def update(self, writer=None, i_episode=None):
        rewards_list = []
        old_states_list = []
        old_actions_list = []
        old_logprobs_list = [] 

        # Monte Carlo estimate of state rewards:
        for i in range(self.n_agents):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.memory[i].rewards), reversed(self.memory[i].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

            # making lists to update
            rewards_list.append(rewards)
            old_states_list.append(torch.squeeze(torch.tensor(self.memory[i].states, dtype=torch.float32).to(device)).detach())
            old_actions_list.append(torch.squeeze(torch.tensor(self.memory[i].actions).to(device)).detach())
            old_logprobs_list.append(torch.squeeze(torch.tensor(self.memory[i].logprobs).to(device)).detach()) 

        # Optimize policy for K epochs: 

        for epoch in range(self.K_epochs): 
            for i in range(self.n_agents):

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
