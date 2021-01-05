from itertools import count
import argparse

from pettingzoo.sisl import multiwalker_v5
import numpy as np

from ppo_cont import PPO as Local_Agent
from global_messenger.ppo import PPO as Global_Messenger
from ppo import Memory
from utils import read_config

from tensorboardX import SummaryWriter
import os
import torch

parser = argparse.ArgumentParser("Reinforcement Learning experiments: Config Generation")
parser.add_argument("--config", type=str, default="configs/random_seed_runs/mw-baseline-agents_5-rs_15.yaml", help="config file name")
parser.add_argument("--load", type=bool, default=False, help="load true / false")
args = parser.parse_args()
config = read_config(args.config)

n_walkers = config["main"]["n_walkers"]
parallel_env = multiwalker_v5.parallel_env(n_walkers=n_walkers, position_noise=1e-3, angle_noise=1e-3,
                                forward_reward=1.0, fall_reward=-100.0, terminate_reward=-100.0,
                                terminate_on_fall=True, max_cycles=500)

MAIN = config["main"]["main"]
parallel_env.reset()
obs_dim = parallel_env.observation_spaces[parallel_env.agents[0]].shape[0]
action_dim = parallel_env.action_spaces[parallel_env.agents[0]].shape[0]

random_seed = config["main"]["random_seed"]
if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    parallel_env.seed(random_seed)
    np.random.seed(random_seed)

writer = SummaryWriter(logdir=os.path.join(config["main"]["logdir"], config["main"]["exp_name"]))
local_memory = [Memory() for _ in range(n_walkers)]
global_memory = Memory()

betas = (0.9, 0.999)

local_state_dim = obs_dim+config["main"]["message_len"] if MAIN else obs_dim
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
        state_dim=(obs_dim * n_walkers) + (action_dim * n_walkers),  # all local observations concatenated + all agents' previous actions
        action_dim=config["main"]["message_len"] * n_walkers,
        action_std=config["global"]["action_std"],
        lr=config["global"]["lr"],
        betas=betas,
        gamma = config["main"]["gamma"],
        K_epochs=config["global"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"],
        hidden_nodes=config["global"]["hidden_nodes"]
    )

obs = parallel_env.reset()
global_agent_state = [obs[i] for i in obs]
global_agent_state = np.array(global_agent_state).reshape((-1,))
global_agent_state = np.concatenate([global_agent_state, np.random.uniform(-1, 1, action_dim * n_walkers)])
i_episode = 0
episode_rewards = 0


# Training loop:
for timestep in count(0):
    # Before running through each agent, we get messages for them, from the global agent.
    global_agent_output = global_messenger.select_action(global_agent_state, global_memory)
    global_agent_output = np.array(global_agent_output).reshape(n_walkers, -1)
    

    for i, agent in enumerate(parallel_env.agents):
        # "MAIN" decides whether the local agents take in any input from the global agent.
        local_state = np.concatenate([obs[agent], global_agent_output[i]]) if MAIN else obs[agent]
        local_action, local_log_prob = local_agent.select_action(local_state)

        # Storing in states, actions, logprobs in memory:
        local_memory[i].states.append(local_state)
        local_memory[i].actions.append(local_action)
        local_memory[i].logprobs.append(local_log_prob)
    
    # Creating a format for the actions in parallel_env:
    actions = {agent : local_memory[i].actions[-1] for i, agent in enumerate(parallel_env.agents)}
    next_obs, rewards, is_terminals, infos = parallel_env.step(actions)

    # Storing in rewards, is_terminals in memory:
    for i, agent in enumerate(parallel_env.agents):
        local_memory[i].rewards.append(rewards[agent])
        local_memory[i].is_terminals.append(is_terminals[agent])
        # So essentially, episode_rewards will be a sum of all agent's rewards at each time step.
        # And finally, we save the epside's performance by saving episode_rewards/n_walkers
        episode_rewards += rewards[agent]

    global_memory.rewards.append(reward)
    global_memory.is_terminals.append(done)

    global_agent_next_state.append(obs)


    if agent_number == n_walkers:
        global_agent_next_state = np.array(global_agent_next_state).reshape((-1,))
        prev_actions = np.array(prev_actions).reshape((-1,))
        global_agent_next_state = np.concatenate([global_agent_next_state, prev_actions])
        global_agent_state = global_agent_next_state
        global_agent_next_state = []
        prev_actions = []

    # update if its time
    if timestep % config["local"]["update_timestep"] == 0:
        local_agent.update(local_memory)
        local_memory.clear_memory()

    if timestep % config["global"]["update_timestep"] == 0:
        global_messenger.update(global_memory)
        global_memory.clear_memory()

    obs = next_obs

    is_terminals = [parallel_env.is_terminals[a] for a in parallel_env.agents]
    if not parallel_env.agents:
        i_episode += 1
        writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards/n_walkers, i_episode)
        timestep=0
        obs = parallel_env.reset()
        global_agent_next_state = []
        prev_actions = []
        print('\nEpisode {} \t  Avg reward for each agent, after an episode: {}\n'.format(i_episode, episode_rewards/n_walkers))
        episode_rewards = 0

    if i_episode == config["main"]["max_episodes"]:
        break

    # save every 50 episodes
    if i_episode % config["main"]["save_interval"] == 0:
        if not os.path.exists(os.path.join(config["main"]["save_dir"], config["main"]["exp_name"])):
            os.makedirs(os.path.join(config["main"]["save_dir"], config["main"]["exp_name"]))
        torch.save(local_agent.policy.state_dict(),
                   os.path.join(config["main"]["save_dir"], config["main"]["exp_name"], "local_agent.pth"))
        torch.save(global_messenger.policy.state_dict(),
                   os.path.join(config["main"]["save_dir"], config["main"]["exp_name"], "global_messenger.pth"))


