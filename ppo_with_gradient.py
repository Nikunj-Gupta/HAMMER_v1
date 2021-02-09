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
    def __init__(self, single_state_dim, single_action_dim, n_agents, actor_layer, critic_layer, meslen, agents):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1

        self.meslen = meslen
        self.n_agents = n_agents
        self.agents = agents
        layers = [] 
        layers.append(nn.Linear(single_state_dim + self.meslen, actor_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(actor_layer[1:])): 
            layers.append(nn.Linear(actor_layer[i], actor_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(actor_layer[-1], single_action_dim)) 
        layers.append(nn.Softmax(dim=-1))
        self.actor = nn.Sequential(*layers) 
        
        # global actor
        layers = [] 
        layers.append(nn.Linear(single_state_dim * self.n_agents, actor_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(actor_layer[1:])): 
            layers.append(nn.Linear(actor_layer[i], actor_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(actor_layer[-1], self.meslen * self.n_agents)) 
        self.global_actor = nn.Sequential(*layers) 
        
        # critic
        layers = [] 
        layers.append(nn.Linear(single_state_dim + self.meslen, critic_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(critic_layer[1:])): 
            layers.append(nn.Linear(critic_layer[i], critic_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(critic_layer[-1], 1)) 
        self.critic = nn.Sequential(*layers) 
        

    def forward(self):
        raise NotImplementedError

    def act(self, obs, memory, global_memory):
        global_agent_state = [obs[i] for i in obs]
        global_agent_state = torch.FloatTensor(global_agent_state).to(device).reshape(1, -1)
        
        # Adding to global memory
        global_memory.states.append(global_agent_state)
        
        # Calculating messages
        global_actor_message = self.global_actor(global_agent_state)
        global_actor_message = global_actor_message.reshape(self.n_agents, self.meslen) 

        action_array = []
        for i, agent in enumerate(self.agents):
            state = torch.FloatTensor(obs[agent])
            local_state = torch.cat((state, global_actor_message[i].detach()), 0).to(device)
            action_probs = self.actor(local_state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_array.append(action.item())
            
            # Adding to memory:
            memory[i].states.append(state)
            memory[i].actions.append(action)
            memory[i].logprobs.append(dist.log_prob(action))

        return {agent : action_array[i] for i, agent in enumerate(self.agents)} 

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() 

        # Messages from global_actor should be detached!!
        state_value = self.critic(state.detach())

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
        self.global_memory = Memory()

        self.policy = ActorCritic(single_state_dim, single_action_dim, n_agents, actor_layer, critic_layer, meslen, agents=self.agents).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(single_state_dim, single_action_dim, n_agents, actor_layer, critic_layer, meslen=meslen, agents=self.agents).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.single_action_dim = single_action_dim
        self.n_agents = n_agents
        self.meslen = meslen        

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))

    def memory_record(self, rewards, is_terminals):
        for i, agent in enumerate(self.agents):
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
            old_actions_list.append(torch.squeeze(torch.tensor(self.memory[i].actions).to(device)).detach())
            old_logprobs_list.append(torch.squeeze(torch.tensor(self.memory[i].logprobs).to(device)).detach()) 
            old_states_list.append(torch.squeeze(torch.stack(self.memory[i].states).to(device)).detach())
        
        # Optimize policy for K epochs: 

        for epoch in range(self.K_epochs): 
            
            old_global_state = torch.stack(self.global_memory.states) # 800x1x54
            old_global_state = torch.squeeze(old_global_state) # 800x54

            
            for i in range(self.n_agents):
                ################## CAVEAT: This is redundant, slows the process!#############
                message = self.policy.global_actor(old_global_state) # 800x12
                message = message.reshape(-1, self.n_agents, self.meslen) # 800x3x4

                # state: 800x18 Message: 800x4   new:800x22 ; so we use dimension 1
                old_state = torch.cat((old_states_list[i], message[:, i, :]), 1)
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_state, old_actions_list[i])

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
