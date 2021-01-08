# Building on top of the TD3 code
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, action_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros((size, obs_dim))
        self.acts_buf = np.zeros((size, action_dim))
        self.rews_buf = np.zeros((size, 1))
        self.next_obs_buf = np.zeros((size, obs_dim))
        self.done_buf = np.zeros((size, 1))
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self,
              obs: np.ndarray,
              act: np.ndarray,
              rew: float,
              next_obs: np.ndarray,
              done: bool,
              ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return (
            torch.FloatTensor(self.obs_buf[idxs]),
            torch.FloatTensor(self.acts_buf[idxs]),
            torch.FloatTensor(self.rews_buf[idxs]),
            torch.FloatTensor(self.next_obs_buf[idxs]),
            torch.FloatTensor(self.done_buf[idxs])
        )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_max: float):
        """Initialization."""
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(128, action_dim)
        )
        self.std_layer = nn.Sequential(
            nn.Linear(128, action_dim)
        )
        self.action_max = torch.FloatTensor(action_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.layers(x)
        mean, std = self.mean_layer(x), torch.exp(self.std_layer(x))
        normal = Normal(mean, std)
        x_t = normal.rsample()  # (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_max
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound (Appendix of https://arxiv.org/abs/1801.01290)
        log_prob -= torch.log(self.action_max * (1 - y_t.pow(2)) + 1e-6)

        # In case of multi dimentional action space, the probs of all actions are added up. 
        log_prob = log_prob.sum(1, keepdim=True)

        # At test time, we want the policy to be deterministic
        action_test = torch.tanh(mean) * self.action_max
        return action, log_prob, action_test.detach()
        

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """Initialization."""
        super(Critic, self).__init__()

        self.Q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.Q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat([state, action], 1)
        return self.Q1_net(x), self.Q2_net(x)


class SACAgent:
    """DQN Agent interacting with environment."""

    def __init__(
            self,
            obs_dim,
            action_dim,
            action_max,
            memory_size: int = 40000,
            batch_size: int = 128,
            tau: float = 0.001,
            gamma: float = 0.99,
            alpha: float = 0.2):

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_max = action_max

        self.memory = ReplayBuffer(
            self.obs_dim, self.action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # networks: critic, critic_target, actor
        self.critic = Critic(self.obs_dim, self.action_dim)
        self.critic_target = Critic(self.obs_dim, self.action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor = Actor(self.obs_dim, self.action_dim, self.action_max)

        # optimizer
        self.optimizer_actor = optim.Adam(self.actor.parameters())
        self.optimizer_critic = optim.Adam(self.critic.parameters())

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state.reshape(1, -1))
        if not self.is_test:
            action, _, _ = self.actor(state)
            action = action.detach().numpy().flatten()
            self.transition = [state, action]
        else:
            _, _, action = self.actor(state)
            action = action.detach().numpy().flatten()
        return action

    def save_in_mem(self, reward, next_state, done) -> Tuple[np.ndarray, np.float64, bool]:
        """Saves reward, next_state, done in memory"""
        self.transition += [reward, next_state, done]
        self.memory.store(*self.transition)

    def train(self):
        """Train the agent."""
        # if training is ready
        if self.memory.size >= self.batch_size:
            B_states, B_actions, B_rewards, B_next_states, B_dones = self.memory.sample_batch()

            # Compute the target Q value
            with torch.no_grad():
                target_actions, target_log_probs, _ = self.actor(B_next_states)
                target_Q1, target_Q2 = self.critic_target(B_next_states, target_actions)
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * target_log_probs
                target_Q = B_rewards + ((1 - B_dones) * self.gamma * target_Q)

            # Get current Q estimate
            current_Q1, current_Q2 = self.critic(B_states, B_actions)

            # Compute critic loss
            critic_loss=F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            # Compute actor loss
            actions, log_probs, _ = self.actor(B_states)
            Q1, Q2 = self.critic(B_states, actions)
            Q = torch.min(Q1, Q2)
            actor_loss = (self.alpha * log_probs - Q).mean()

            # Optimize the actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            return actor_loss, critic_loss
        else:
            return 0, 0

from pettingzoo.sisl import multiwalker_v6
from itertools import count
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter(logdir='logs/')
    i_episode = 0
    parallel_env = multiwalker_v6.parallel_env()
    obs = parallel_env.reset()
    obs_dim = parallel_env.observation_spaces[parallel_env.agents[0]].shape[0]
    action_dim = parallel_env.action_spaces[parallel_env.agents[0]].shape[0]
    agent_action_space = parallel_env.action_spaces[parallel_env.agents[0]]
    score = 0
    agents = [agent for agent in parallel_env.agents]
    step_lenght = 0
    actor_loss = [0 for agent in agents]
    critic_loss = [0 for agent in agents]

    # Initializing agents
    local_agents = [SACAgent(obs_dim, action_dim, agent_action_space.high) for agent in parallel_env.agents]

    # Training loop:
    for timestep in count(1):

        # Selecting actions.
        action_list = []
        for i, agent in enumerate(agents):
            local_state =  obs[agent]
            local_action = local_agents[i].select_action(local_state)
            action_list.append(local_action)

        actions = {agent : np.clip(action_list[i], agent_action_space.low, agent_action_space.high) for i, agent in enumerate(agents)}
        next_obs, rewards, is_terminals, infos = parallel_env.step(actions)
        step_lenght += 1
        score += sum(rewards.values())/len(rewards.values())

        for i, agent in enumerate(agents):
            local_agents[i].save_in_mem(rewards[agent], next_obs[agent], is_terminals[agent])
            a_l, c_l = local_agents[i].train()
            actor_loss[i] += a_l
            critic_loss[i] += c_l
        

        obs = next_obs

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            writer.add_scalar('episode reward', score, i_episode)
            for i, agent in enumerate(agents):
                writer.add_scalar('actor_loss/{}'.format(agent), actor_loss[i], i_episode)
                writer.add_scalar('critic_loss/{}'.format(agent), critic_loss[i], i_episode)
                actor_loss[i] = 0
                critic_loss[i] = 0

            print('Episode {} \t  episode reward: {} \t Step length: {}'.format(i_episode, score, step_lenght))
            obs = parallel_env.reset()
            step_lenght = 0
            i_episode += 1
            score = 0
    


