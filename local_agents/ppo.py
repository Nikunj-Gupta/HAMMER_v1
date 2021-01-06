import os

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from tensorboardX import SummaryWriter

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
    def __init__(self, state_dim, action_dim, action_std, hidden_nodes=64):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, 1)
        )
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
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, hidden_nodes=64):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std, hidden_nodes).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std, hidden_nodes).to(device)
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
            old_states_list.append(torch.squeeze(torch.tensor(local_memory[i].states).to(device), 1).detach())
            old_actions_list.append(torch.squeeze(torch.tensor(local_memory[i].actions).to(device), 1).detach())
            old_logprobs_list.append(torch.squeeze(torch.tensor(local_memory[i].logprobs).to(device), 1).detach())
        
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


def main():
    ############## Hyperparameters ##############
    env_name = "BipedalWalker-v3"
    exp_name = "ppo_continuous_check"
    render = False
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    random_seed = None
    #############################################

    writer = SummaryWriter(logdir=os.path.join("logs/ppo_/", exp_name))
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        t = 0
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/ppo_/", exp_name))
            break

        # save every 500 episodes
        if i_episode % 500 == 0:
            if not os.path.exists(os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/ppo_/", exp_name)):
                os.makedirs(os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/ppo_/", exp_name))
            torch.save(ppo.policy.state_dict(), os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/ppo_/", exp_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            writer.add_scalar('Running Reward', running_reward, i_episode)
            writer.add_scalar('Average Length', avg_length, i_episode)

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()

