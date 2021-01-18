import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

from itertools import count

from tensorboardX import SummaryWriter

from pettingzoo.mpe import simple_spread_v2


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
        
    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
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
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    writer = SummaryWriter(logdir='TRUE_INDEPENDENT_PPO/logs/800_3e-4') 

    ############## Hyperparameters ##############
    n_agents = 3
    env = simple_spread_v2.parallel_env(N=n_agents, local_ratio=0.5, max_cycles=25)
    env.reset()
    obs_dim = env.observation_spaces[env.agents[0]].shape[0]
    action_dim = env.action_spaces[env.agents[0]].n
    agent_action_space = env.action_spaces[env.agents[0]]

    # creating environment
    render = False
    log_interval = 1           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 800      # update policy every n timesteps
    lr = 0.0003
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 8                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    ppo_memories = [Memory() for _ in range(n_agents)]
    ppo_agents = [PPO(obs_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip) for _ in range(n_agents)]
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    agents = [agent for agent in env.agents] 
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        while True:
            timestep += 1
            
            action_array = [] 
            # Running policy_old:
            for i, agent in enumerate(agents):
                action = ppo_agents[i].policy_old.act(state[agent], ppo_memories[i])
                action_array.append(action)

            actions = {agent : action_array[i] for i, agent in enumerate(agents)}  
            state, reward, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                # Saving reward and is_terminal:
                ppo_memories[i].rewards.append(reward[agent])
                ppo_memories[i].is_terminals.append(done[agent])
                
            # update if its time
            if timestep % update_timestep == 0:
                for i, agent in enumerate(agents):
                    ppo_agents[i].update(ppo_memories[i])
                    ppo_memories[i].clear_memory()
                timestep = 0
        
            running_reward += sum(reward.values())/len(reward.values())
            if render:
                env.render()
            if all([done[agent] for agent in agents]):
                break


                
        # logging
        if i_episode % log_interval == 0:
            writer.add_scalar('reward (avg agent)', running_reward, i_episode)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t reward: {}'.format(i_episode, running_reward))
            running_reward = 0
            
if __name__ == '__main__':
    main()
    
