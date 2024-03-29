from itertools import count
import argparse

from pettingzoo.sisl import multiwalker_v6
import numpy as np

from local_agents.ppo import PPO as Local_Agent
from global_messenger.ppo import PPO as Global_Messenger
from local_agents.ppo import Memory 
from utils import read_config

from tensorboardX import SummaryWriter
import os
import torch

parser = argparse.ArgumentParser("Reinforcement Learning experiments: Config Generation")
parser.add_argument("--config", type=str, default="configs/random_seed_runs/mw-baseline-agents_5-rs_15.yaml", help="config file name")
parser.add_argument("--load", type=bool, default=False, help="load true / false") 

parser.add_argument("--hammer", type=int, default=1, help="1 for hammer; 0 for IL") 
parser.add_argument("--nagents", type=int, default=3) 
parser.add_argument("--expname", type=str, default=None)

parser.add_argument("--maxepisodes", type=int, default=30000) 

parser.add_argument("--meslen", type=int, default=4, help="message length")
parser.add_argument("--randomseed", type=int, default=10)
parser.add_argument("--render", type=bool, default=False)

parser.add_argument("--saveinterval", type=int, default=50)
parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

args = parser.parse_args()
config = read_config(args.config)

n_agents = args.nagents
parallel_env = multiwalker_v6.parallel_env(n_walkers=n_agents, position_noise=1e-3, angle_noise=1e-3,
                                forward_reward=1.0, fall_reward=-100.0, terminate_reward=-100.0,
                                terminate_on_fall=True, max_cycles=500)

MAIN = args.hammer
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
global_memory = Memory()

betas = (0.9, 0.999)

local_state_dim = obs_dim+args.meslen if MAIN else obs_dim
local_agent = Local_Agent(
        state_dim=local_state_dim,
        action_dim=action_dim,
        action_std=config["global"]["action_std"],
        lr=config["global"]["lr"],
        betas=betas,
        gamma=config["main"]["gamma"],
        K_epochs=config["global"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"],
        hidden_nodes=config["global"]["hidden_nodes"]
    )
global_messenger = Global_Messenger(
        state_dim=(obs_dim * n_agents) + (action_dim * n_agents),  # all local observations concatenated + all agents' previous actions
        action_dim=args.meslen,
        n_agents=n_agents,
        action_std=config["global"]["action_std"],
        lr=config["global"]["lr"],
        betas=betas,
        gamma = config["main"]["gamma"],
        K_epochs=config["global"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"],
        hidden_nodes=config["global"]["hidden_nodes"]
    )

obs = parallel_env.reset()
if MAIN: 
    global_agent_state = [obs[i] for i in obs]
    global_agent_state = np.array(global_agent_state).reshape((-1,))
    global_agent_state = np.concatenate([global_agent_state, np.random.uniform(-1, 1, action_dim * n_agents)])
i_episode = 0
episode_rewards = 0
agents = [agent for agent in parallel_env.agents]

# Training loop, starting count from 1:
for timestep in count(1):
    # Before running through each agent, we get messages for them, from the global agent.
    if MAIN: 
        global_agent_output, global_agent_log_prob = global_messenger.select_action(global_agent_state)    

    for i, agent in enumerate(agents):
        # "MAIN" decides whether the local agents take in any input from the global agent.
        local_state = np.concatenate([obs[agent], global_agent_output[i]]) if MAIN else obs[agent]
        local_action, local_log_prob = local_agent.select_action(local_state)

        # Storing in states, actions, logprobs in local_memory:
        local_memory[i].states.append(local_state)
        local_memory[i].actions.append(local_action)
        local_memory[i].logprobs.append(local_log_prob)
    
    # Creating a format for the actions in parallel_env:
    actions = {agent : np.clip(local_memory[i].actions[-1], agent_action_space.low, agent_action_space.high) for i, agent in enumerate(agents)}
    next_obs, rewards, is_terminals, infos = parallel_env.step(actions)

    # Storing in rewards, is_terminals in local_memory, :
    for i, agent in enumerate(agents):
        local_memory[i].rewards.append(rewards[agent])
        local_memory[i].is_terminals.append(is_terminals[agent])
        # So essentially, episode_rewards will be a sum of all agent's rewards at each time step.
        # And finally, we save the epside's performance by saving episode_rewards/n_agents
        episode_rewards += rewards[agent]

    if MAIN: 
        global_memory.states.append(global_agent_state)
        global_memory.actions.append(global_agent_output)
        global_memory.logprobs.append(global_agent_log_prob)
        global_memory.rewards.append([rewards[agent] for agent in agents])
        global_memory.is_terminals.append([is_terminals[agent] for agent in agents])


    # update if its time
    if timestep % config["local"]["update_timestep"] == 0:
        local_agent.update(local_memory)
        [mem.clear_memory() for mem in local_memory]

    if MAIN and timestep % config["global"]["update_timestep"] == 0:
        global_messenger.update(global_memory)
        global_memory.clear_memory()

    obs = next_obs

    # If episode had ended
    if all([is_terminals[agent] for agent in agents]):
        i_episode += 1
        writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards/n_agents, i_episode)
        obs = parallel_env.reset()
        print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards/n_agents))
        episode_rewards = 0

    # save every 50 episodes
    if i_episode % args.saveinterval == 0:
        if not os.path.exists(os.path.join(args.savedir, args.expname)):
            os.makedirs(os.path.join(args.savedir, args.expname))
        torch.save(local_agent.policy.state_dict(),
                   os.path.join(args.savedir, args.expname, "local_agent.pth"))
        if MAIN: 
            torch.save(global_messenger.policy.state_dict(),
                   os.path.join(args.savedir, args.expname, "global_messenger.pth"))
    
    if i_episode == args.maxepisodes:
        break
    
    # Update global_agent_state here:
    # Note: we use obs here, to insure that if the episode ends, we give the global agent, the new reset state.
    if MAIN: 
        global_agent_state = np.array([obs[agent] for agent in agents]).reshape((-1,))
        prev_actions = np.array([actions[agent] for agent in agents]).reshape((-1,))
        global_agent_state = np.concatenate([global_agent_state, prev_actions])
        global_agent_state = global_agent_state

