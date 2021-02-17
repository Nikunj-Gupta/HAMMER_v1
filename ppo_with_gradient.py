# PPO implementation of the global messenger for HAMMER.
import os

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import gym
import numpy as np
from tensorboardX import SummaryWriter
from dru import DRU 
# torch.autograd.set_detect_anomaly(True)
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
    def __init__(self, single_state_dim, single_action_dim, n_agents, actor_layer, critic_layer, meslen, agents, dru_toggle=0, is_discrete=1):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1

        self.meslen = meslen
        self.n_agents = n_agents
        self.agents = agents 
        self.is_discrete = is_discrete 
        self.action_std = 0.5 
        layers = [] 
        layers.append(nn.Linear(single_state_dim + self.meslen, actor_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(actor_layer[1:])): 
            layers.append(nn.Linear(actor_layer[i], actor_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(actor_layer[-1], single_action_dim)) 
        if self.is_discrete:
            layers.append(nn.Softmax(dim=-1)) 
        else: 
            layers.append(nn.Tanh()) 
        self.actor = nn.Sequential(*layers) 
        
        # global actor
        layers = [] 
        layers.append(nn.Linear(single_state_dim * self.n_agents, actor_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(actor_layer[1:])): 
            layers.append(nn.Linear(actor_layer[i], actor_layer[i+1]))
            layers.append(nn.Tanh()) 
        self.global_encoder = nn.Sequential(*layers)

        # not using nn.ModuleList to ensure that the global_actor_decoder parameters are not taken into the parameters.
        # We want separate optimizer for decoders.
        self.global_actor_decoder = [nn.Linear(actor_layer[-1], self.meslen) for _ in range(self.n_agents)]
        self.dru_toggle = dru_toggle 
        if self.dru_toggle: 
            self.dru = DRU(hard=True) 
        
        # critic
        layers = [] 
        layers.append(nn.Linear(single_state_dim + self.meslen, critic_layer[0])) 
        layers.append(nn.Tanh()) 
        for i in range(len(critic_layer[1:])): 
            layers.append(nn.Linear(critic_layer[i], critic_layer[i+1]))
            layers.append(nn.Tanh()) 
        layers.append(nn.Linear(critic_layer[-1], 1)) 
        self.critic = nn.Sequential(*layers) 

        self.action_var = torch.full((single_action_dim,), self.action_std * self.action_std).to(device)

    def global_actor(self, state):
        latent_vector = self.global_encoder(state)
        message = []
        for decoder in self.global_actor_decoder: 
            # Obtaining message using decoder and then Passing message through DRU 
            if self.dru_toggle: 
                message.append(self.dru.forward(message=decoder(latent_vector), mode="R")) 
            else: 
                message.append(decoder(latent_vector)) 
        return message

    def forward(self):
        raise NotImplementedError

    def act(self, obs, memory, global_memory):
        global_agent_state = [obs[i] for i in obs]
        global_agent_state = torch.FloatTensor(global_agent_state).to(device).reshape(1, -1)
        
        # Adding to global memory
        global_memory.states.append(global_agent_state)
        
        # Calculating messages
        global_actor_message = self.global_actor(global_agent_state) 

        if self.is_discrete: 
            action_array = []
            for i, agent in enumerate(self.agents):
                state = torch.FloatTensor(obs[agent])
                local_state = torch.cat((state, global_actor_message[i].reshape(-1).detach()), 0).to(device)
                action_probs = self.actor(local_state)
                dist = Categorical(action_probs)
                action = dist.sample()
                action_array.append(action.item())
                
                # Adding to memory:
                memory[i].states.append(state)
                memory[i].actions.append(action)
                memory[i].logprobs.append(dist.log_prob(action))

            return {agent : action_array[i] for i, agent in enumerate(self.agents)} 
        else: 
            action_array = []
            for i, agent in enumerate(self.agents):
                state = torch.FloatTensor(obs[agent])
                local_state = torch.cat((state, global_actor_message[i].reshape(-1).detach()), 0).to(device)
                action_mean = self.actor(local_state) 
                cov_mat = torch.diag(self.action_var).to(device) 
                dist = MultivariateNormal(action_mean, cov_mat)
                action = dist.sample()
                action_logprob = dist.log_prob(action) 
                action_array.append(np.array(action.detach())) 

                # Adding to memory:
                memory[i].states.append(state)
                memory[i].actions.append(action)
                memory[i].logprobs.append(action_logprob) 

            return {agent : action_array[i] for i, agent in enumerate(self.agents)} 

    def evaluate(self, state, action): 
        if self.is_discrete: 
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy() 

            # Messages from global_actor should be detached!!
            state_value = self.critic(state.detach())

        else: 
            action_mean = self.actor(state.float())
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)

            dist = MultivariateNormal(action_mean, cov_mat)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            
            # Messages from global_actor should be detached!!
            state_value = self.critic(state.float().detach())
            
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, agents, single_state_dim, single_action_dim, meslen, n_agents, lr, betas, gamma, K_epochs, eps_clip, actor_layer, critic_layer, dru_toggle=0, is_discrete=1):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.agents = agents
        self.memory = [Memory() for _ in self.agents]
        self.global_memory = Memory()
        self.n_agents = n_agents

        self.policy = ActorCritic(single_state_dim, single_action_dim, n_agents, actor_layer, critic_layer, meslen, agents=self.agents, dru_toggle=dru_toggle, is_discrete=is_discrete).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.decoder_optimizer = [torch.optim.Adam(self.policy.global_actor_decoder[i].parameters(), lr=lr, betas=betas) for i in range(self.n_agents)]

        self.policy_old = ActorCritic(single_state_dim, single_action_dim, n_agents, actor_layer, critic_layer, meslen=meslen, agents=self.agents, dru_toggle=dru_toggle, is_discrete=is_discrete).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.single_action_dim = single_action_dim
        self.meslen = meslen 
        self.is_discrete = is_discrete  

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
            if self.is_discrete: 
                old_actions_list.append(torch.squeeze(torch.tensor(self.memory[i].actions).to(device)).detach()) 

            else: 
                old_actions_list.append(torch.squeeze(torch.stack(self.memory[i].actions).to(device)).detach()) 

            old_logprobs_list.append(torch.squeeze(torch.tensor(self.memory[i].logprobs).to(device)).detach()) 
            old_states_list.append(torch.squeeze(torch.stack(self.memory[i].states).to(device)).detach())
        
        # Optimize policy for K epochs: 

        for epoch in range(self.K_epochs): 
            
            old_global_state = torch.stack(self.global_memory.states) # 800x1x54
            old_global_state = torch.squeeze(old_global_state) # 800x54
            
            for i in range(self.n_agents):
                ################## CAVEAT: This is redundant, slows the process!#############
                message = self.policy.global_actor(old_global_state) # 3x800x4

                # state: 800x18 Message: 800x4   new:800x22 ; so we use dimension 1
                old_state = torch.cat((old_states_list[i], message[i]), 1)
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
                self.decoder_optimizer[i].zero_grad()
                loss.mean().backward()
                # for j in range(self.n_agents):
                #     print(j, self.policy.global_actor_decoder[j].weight[:10, 0])
                self.optimizer.step()
                self.decoder_optimizer[i].step()
                # print("STEP")
                # for j in range(self.n_agents):
                #     print(j, self.policy.global_actor_decoder[j].weight[:10, 0])
                # print()


            # if writer is not None and epoch == self.K_epochs-1:
            #     writer.add_scalar('actor_loss/local_agent', actor_loss.mean(), i_episode)
            #     writer.add_scalar('critic_loss/local_agent', critic_loss.mean(), i_episode)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.global_actor_decoder[0].load_state_dict(self.policy.global_actor_decoder[0].state_dict())