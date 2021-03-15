import argparse
from itertools import count 

from tensorboardX import SummaryWriter

from ppo_with_gradient import PPO

from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_reference_v2 
from pettingzoo.utils.capture_stdout import capture_stdout 
from utils import read_config
import os, time 
import numpy as np
import torch
import json, matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from local_agents.ppo_discrete import PPO as LocalPolicy 


from pathlib import Path

def preprocess_one_obs(obs, which=1, limit=10): 
    agent = "agent_" + str(which) 
    obs[agent][limit:] = [0.]*(len(obs["agent_0"])-(limit)) 
    return obs 


def preprocess_obs(obs, limit): 
    for i in obs: 
        obs[i] = obs[i][:limit] 
    return obs 

def reset_world(env):
    # random properties for agents
    for i, agent in enumerate(env.agents):
        agent.color = np.array([0.35, 0.35, 0.85])
    # # random properties for landmarks
    # for i, landmark in enumerate(world.landmarks):
    #     landmark.color = np.array([0.25, 0.25, 0.25])
    # # set random initial states
    # for agent in world.agents:
    #     agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
    #     agent.state.p_vel = np.zeros(world.dim_p)
    #     agent.state.c = np.zeros(world.dim_c)
    # for i, landmark in enumerate(world.landmarks):
    #     landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
    #     landmark.state.p_vel = np.zeros(world.dim_p) 
    print()

def run(args):
    env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=args.maxcycles)

    env.reset()
    agents = [agent for agent in env.agents] 

    if args.partialobs: 
        print("Using Partial Observations") 
    
    if args.heterogeneity: 
        print("Using Heterogeneous Local Agents") 

    if args.heterogeneity: 
        assert args.limit == 10 
        obs_dim = len(preprocess_one_obs(env.reset(), limit=args.limit)["agent_0"]) 
    elif args.partialobs:
        obs_dim = len(preprocess_obs(env.reset(), limit=args.limit)["agent_0"]) 
    else:
        obs_dim = env.observation_spaces[env.agents[0]].shape[0]
        
    action_dim = env.action_spaces[env.agents[0]].n if args.envname == "cn" else 5 

    agent_action_space = env.action_spaces[env.agents[0]]


    config = read_config(args.config) 
    if not config:
        print("config required")
        return
    
    random_seed = args.randomseed 
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed) 
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)


    expname = args.envname if args.expname == None else args.expname
    betas = (0.9, 0.999)

    HAMMER = PPO(
        agents=agents,
        single_state_dim=obs_dim, # all local observations concatenated + all agents' previous actions
        single_action_dim=action_dim,
        meslen = args.meslen, 
        n_agents=len(agents), # required for discrete messages
        lr=config["global"]["lr"],
        betas=betas,
        gamma = config["main"]["gamma"],
        K_epochs=config["global"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"],        
        actor_layer=config["global"]["actor_layer"],
        critic_layer=config["global"]["critic_layer"], 
        dru_toggle=args.dru_toggle 
    ) 

    HAMMER.policy_old.load_state_dict(torch.load(str(os.path.join(args.load_hammer, "local_agent.pth")))) 

    IL = LocalPolicy(
        state_dim=obs_dim, 
        action_dim=action_dim,
        actor_layer=config["local"]["actor_layer"],
        critic_layer=config["local"]["critic_layer"],
        lr=config["local"]["lr"],
        betas=betas,
        gamma=config["main"]["gamma"],
        K_epochs=config["local"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"], 
        shared=False 
    ) 
    IL.policy_old.load_state_dict(torch.load(str(os.path.join(args.load_il, "local_agent.pth")))) 


    if args.dru_toggle: 
        print("Using DRU") 
    else: 
        print("Not Using DRU")

    # logging variables
    ep_reward = 0
    global_timestep = 0

    if args.heterogeneity: 
        obs = preprocess_one_obs(env.reset(), limit=args.limit) 
    elif args.partialobs: 
        obs = preprocess_obs(env.reset(), limit=args.limit)
    else:  
        obs = env.reset() 
    obs_copy = obs 

    episode_rewards = 0 

    all_messages = [] 
    all_hammer_actions = [] 
    all_il_actions = [] 
    # env.render() 
    # time.sleep(10) 


    positions = {"x":[], "y":[]} 

    # for timestep in count(1): 
    #     env.render() 
    #     # time.sleep(0.1) 
    #     positions["x"].append(obs["agent_2"][2]) 
    #     positions["y"].append(obs["agent_2"][3]) 
        

    #     actions, messages = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory) 
    #     all_hammer_actions.append(actions) 
    #     all_messages.append(messages) 
    #     next_obs, rewards, is_terminals, infos = env.step(actions) 

    #     HAMMER.memory_record(rewards, is_terminals)
    #     episode_rewards += list(rewards.values())[0] 

    #     if args.partialobs: 
    #         next_obs = preprocess_obs(next_obs, limit=args.limit) 
    #     elif args.heterogeneity: 
    #         next_obs = preprocess_one_obs(next_obs, limit=args.limit) 
    #     obs = next_obs 

    #     # If episode had ended
    #     if all([is_terminals[agent] for agent in agents]):
    #         print('Episode reward: {}'.format(episode_rewards)) 
    #         break
    

    for timestep in count(1): 
        env.render() 
        time.sleep(0.1) 
        positions["x"].append(obs["agent_2"][2]) 
        positions["y"].append(obs["agent_2"][3]) 
        

        action_array = [] 
        for i, agent in enumerate(agents): 
            local_state = obs[agent]
            action, local_log_prob = IL.policy_old.act(local_state) 
            action_array.append(action) 
        
        actions = {agent : action_array[i] for i, agent in enumerate(agents)} 
        all_il_actions.append(actions) 

        next_obs, rewards, is_terminals, infos = env.step(actions) 
        if args.partialobs: 
            next_obs = preprocess_obs(next_obs, limit=args.limit) 
        elif args.heterogeneity: 
            next_obs = preprocess_one_obs(next_obs, limit=args.limit) 

        episode_rewards += rewards["agent_0"]

        obs = next_obs

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            print('Episode reward: {}'.format(episode_rewards))
            break 
    plt.plot(positions["x"][:25],positions["y"][:25]) 
    plt.show() 

    # for i in range(50): 
    #     if all_messages: print(all_messages[i]) 
    #     if all_hammer_actions: print(all_hammer_actions[i]) 
    #     print()
    #     if all_il_actions: print(all_il_actions[i]) 

    #     print("================================================================") 
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/cn/hyperparams.yaml', help="config file name") 
    parser.add_argument("--load_hammer", type=str, default=None, help="load path") 
    parser.add_argument("--load_il", type=str, default=None, help="load path") 


    parser.add_argument("--expname", type=str, default=None) 
    parser.add_argument("--envname", type=str, default='cn') 
    parser.add_argument("--nagents", type=int, default=3) 

    parser.add_argument("--maxepisodes", type=int, default=10) 
    parser.add_argument("--partialobs", type=int, default=0) 

    parser.add_argument("--heterogeneity", type=int, default=0) 
    parser.add_argument("--limit", type=int, default=10) # 11 for sr, 10 for cn
    parser.add_argument("--maxcycles", type=int, default=25) 

    parser.add_argument("--dru_toggle", type=int, default=1) 

    parser.add_argument("--meslen", type=int, default=1, help="message length")
    parser.add_argument("--randomseed", type=int, default=1)

    parser.add_argument("--saveinterval", type=int, default=50_000) 
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    # for i in range(10): 
    #     args.randomseed = i 
    run(args=args)
