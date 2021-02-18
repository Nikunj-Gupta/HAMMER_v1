from itertools import count
import argparse

from pettingzoo.sisl import multiwalker_v6
import numpy as np

from local_agents.ppo import PPO as Local_Agent
from local_agents.ppo import Memory 
from utils import read_config

from tensorboardX import SummaryWriter
import os
import torch


parser = argparse.ArgumentParser("Multi-Agent Walker") 

parser.add_argument("--config", type=str, default="configs/2021/cn/hyperparams.yaml", help="config file name")
parser.add_argument("--load", type=bool, default=False, help="load true / false") 

parser.add_argument("--nagents", type=int, default=3) 
parser.add_argument("--expname", type=str, default="il_mw")

parser.add_argument("--maxepisodes", type=int, default=50_000) 
parser.add_argument("--randomseed", type=int, default=10)
parser.add_argument("--render", type=bool, default=False)

parser.add_argument("--saveinterval", type=int, default=50)
parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

args = parser.parse_args()
config = read_config(args.config)

n_agents = args.nagents
parallel_env = multiwalker_v6.parallel_env(n_walkers=n_agents) 

parallel_env.reset()
obs_dim = parallel_env.observation_spaces[parallel_env.agents[0]].shape[0]
action_dim = parallel_env.action_spaces[parallel_env.agents[0]].shape[0]
agent_action_space = parallel_env.action_spaces[parallel_env.agents[0]]


random_seed = args.randomseed
if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    parallel_env.seed(random_seed)
    np.random.seed(random_seed)

writer = SummaryWriter(logdir=os.path.join(args.logdir, args.expname))
local_memory = [Memory() for _ in range(n_agents)]

betas = (0.9, 0.999)

local_state_dim = obs_dim
local_agent = Local_Agent(
        state_dim=local_state_dim,
        action_dim=action_dim,
        action_std=config["global"]["action_std"],
        lr=config["global"]["lr"],
        betas=betas,
        gamma=config["main"]["gamma"],
        K_epochs=config["global"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"],
        actor_layer=config["local"]["actor_layer"],
        critic_layer=config["local"]["critic_layer"],
    )
obs = parallel_env.reset()
i_episode = 0
episode_rewards = 0
agents = [agent for agent in parallel_env.agents]

for timestep in count(1):
    for i, agent in enumerate(agents):
        local_state = obs[agent]
        local_action, local_log_prob = local_agent.select_action(local_state)

        local_memory[i].states.append(local_state)
        local_memory[i].actions.append(local_action)
        local_memory[i].logprobs.append(local_log_prob)
    
    actions = {agent : np.clip(local_memory[i].actions[-1], agent_action_space.low, agent_action_space.high) for i, agent in enumerate(agents)}
    next_obs, rewards, is_terminals, infos = parallel_env.step(actions)

    for i, agent in enumerate(agents):
        local_memory[i].rewards.append(rewards[agent])
        local_memory[i].is_terminals.append(is_terminals[agent])
        episode_rewards += rewards[agent]

    if timestep % config["local"]["update_timestep"] == 0:
        local_agent.update(local_memory)
        [mem.clear_memory() for mem in local_memory]
    obs = next_obs

    if all([is_terminals[agent] for agent in agents]):
        i_episode += 1
        writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards/n_agents, i_episode)
        obs = parallel_env.reset()
        print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards/n_agents))
        episode_rewards = 0

    if i_episode % args.saveinterval == 0:
        if not os.path.exists(os.path.join(args.savedir, args.expname)):
            os.makedirs(os.path.join(args.savedir, args.expname))
        torch.save(local_agent.policy.state_dict(),
                   os.path.join(args.savedir, args.expname, "local_agent.pth"))
    if i_episode == args.maxepisodes:
        break

