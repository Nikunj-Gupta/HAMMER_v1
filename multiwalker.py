import argparse

from pettingzoo.sisl import multiwalker_v2
import numpy as np

from ppo_cont import PPO as Policy
from ppo import Memory
from utils import read_config

from tensorboardX import SummaryWriter
import os
import torch

parser = argparse.ArgumentParser("Reinforcement Learning experiments: Config Generation")
parser.add_argument("--config", type=str, default=None, help="config file name")
parser.add_argument("--load", type=bool, default=False, help="load true / false")
args = parser.parse_args()
config = read_config(args.config)

n_walkers = config["main"]["n_walkers"]
env = multiwalker_v2.env(n_walkers=n_walkers, position_noise=1e-3, angle_noise=1e-3,
                         forward_reward=1.0, fall_reward=-100.0, drop_reward=-100.0,
                         terminate_on_fall=True, max_frames=500)

MAIN = config["main"]["main"]
obs_dim = env.observation_spaces[env.agents[0]].shape[0]
action_dim = env.action_spaces[env.agents[0]].shape[0]

random_seed = config["main"]["random_seed"]
if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

writer = SummaryWriter(logdir=os.path.join(config["main"]["logdir"], config["main"]["exp_name"]))
local_memory = Memory()
global_memory = Memory()

betas = (0.9, 0.999)

local_state_dim = obs_dim+config["main"]["message_len"] if MAIN else obs_dim
local_agent = Policy(
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
global_agent = Policy(
        state_dim=(obs_dim * n_walkers) + (action_dim * n_walkers),  # all local observations concatenated + all agents' previous actions
        action_dim=config["main"]["message_len"],
        action_std=config["global"]["action_std"],
        lr=config["global"]["lr"],
        betas=betas,
        gamma = config["main"]["gamma"],
        K_epochs=config["global"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"],
        hidden_nodes=config["global"]["hidden_nodes"]
    )

prev_actions = np.random.uniform(-1, 1, action_dim * n_walkers)
global_agent_state = [env.reset() for i in range(n_walkers)]
global_agent_state = np.array(global_agent_state).reshape((-1,))
global_agent_state = np.concatenate([global_agent_state, prev_actions])
global_agent_next_state = []
prev_actions = []
local_timestep = 0
global_timestep = 0
i_episode = 0
timestep = 0
ep_rew = 0
ep_rew_array = []

obs = env.reset()

episodes = config["main"]["max_episodes"]
for agent in env.agent_iter():
    # print(agent)
    local_timestep += 1
    global_timestep += 1

    timestep+=1

    if i_episode==episodes:
        break

    reward, done, info = env.last()
    ep_rew += reward

    global_memory.rewards.append(reward)
    global_memory.is_terminals.append(done)
    local_memory.rewards.append(reward)
    local_memory.is_terminals.append(done)

    global_agent_action = global_agent.select_action(global_agent_state, global_memory)
    global_agent_action = np.array(global_agent_action)

    state = np.concatenate([obs, global_agent_action]) if MAIN else obs
    action = local_agent.select_action(state, local_memory)

    global_agent_next_state.append(obs)
    prev_actions.append(action)

    obs = env.step(action)

    if agent == env.agents[-1]:
        global_agent_next_state = np.array(global_agent_next_state).reshape((-1,))
        prev_actions = np.array(prev_actions).reshape((-1,))
        global_agent_next_state = np.concatenate([global_agent_next_state, prev_actions])
        global_agent_state = global_agent_next_state
        global_agent_next_state = []
        prev_actions = []

    # update if its time
    if local_timestep % config["local"]["update_timestep"] == 0:
        local_agent.update(local_memory)
        local_memory.clear_memory()
        local_timestep = 0

    if global_timestep % config["global"]["update_timestep"] == 0:
        global_agent.update(global_memory)
        global_memory.clear_memory()
        global_timestep = 0

    # save every 50 episodes
    if i_episode % config["main"]["save_interval"] == 0:
        if not os.path.exists(os.path.join(config["main"]["save_dir"], config["main"]["exp_name"])):
            os.makedirs(os.path.join(config["main"]["save_dir"], config["main"]["exp_name"]))
        torch.save(local_agent.policy.state_dict(),
                   os.path.join(config["main"]["save_dir"], config["main"]["exp_name"], "local_agent.pth"))
        torch.save(global_agent.policy.state_dict(),
                   os.path.join(config["main"]["save_dir"], config["main"]["exp_name"], "global_agent.pth"))


    dones = [env.dones[a] for a in env.agents]
    if not all(dones) and timestep==500:
        ep_rew_array.append(ep_rew)
        ep_rew = 0
        timestep = 0
    if all(dones):
        i_episode += 1
        ep_rew_array.append(ep_rew)
        writer.add_scalar('Episode Reward', ep_rew, i_episode)
        ep_rew=0
        timestep=0
        obs = env.reset()
        global_agent_next_state = []
        prev_actions = []
    print('\nEpisode {} \t  Episode reward: {}\n'.format(i_episode, ep_rew))

