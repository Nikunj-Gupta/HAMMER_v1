import argparse
from itertools import count

# from tensorboardX import SummaryWriter

from local_agents.ppo_discrete import PPO as LocalPolicy 
# from global_messenger.ppo import PPO as GlobalPolicy 
from global_messenger.ppo_single_update import PPO as GlobalPolicy 
from local_agents.ppo_discrete import Memory 
# from baselines.independent_learners.sac import SACAgent 

from pettingzoo.mpe import simple_spread_v2
from utils import read_config
import os
import numpy as np
import torch
import json 

from hammer_cn import preprocess_obs

from sklearn.decomposition import PCA
import pandas as pd

import matplotlib.pyplot as plt

def append_message(message, episodic_messages):
    for i, agent_mes in enumerate(message):
        episodic_messages[i].append(agent_mes)

def analyse_message(discretemes, episodic_messages):
    plt.figure(figsize=[30, 6])
    for i, agent_messages in enumerate(episodic_messages):
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(agent_messages)

        plt.subplot(1, 3, i+1)
        plt.title(f"scores per episode")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.scatter(principalComponents[:,0], principalComponents[:,1])

    # plt.grid()
    # plt.show()
    plt.savefig('plots/1.dqn-simple.png')
    plt.close()
            

        
    # pass

def run(args = None):

    env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=args.maxcycles) 
    env.reset()
    obs_space = env.observation_spaces 

    if args.partialobs: 
        print("Using Partial Observations") 
    
    if args.heterogeneity: 
        print("Using Heterogeneous Local Agents") 

    if args.heterogeneity: 
        obs_dim = len(preprocess_one_obs(env.reset(), limit=args.limit)["agent_0"]) 
    elif args.partialobs:  
        obs_dim = len(preprocess_obs(env.reset(), limit=args.limit)["agent_0"])  
    else:
        obs_dim = env.observation_spaces[env.agents[0]].shape[0] 
        
    action_dim = env.action_spaces[env.agents[0]].n

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


    expname = args.expname if args.expname is not None else 'cn----L-lr-{}-updatestep-{}-epoch-{}----G-lr-{}-updatestep-{}-epoch-{}----nagents-{}-hammer-{}-meslen-{}'.format(config["local"]["lr"], config["local"]["update_timestep"], config["local"]["K_epochs"], config["global"]["lr"], config["global"]["update_timestep"], config["global"]["K_epochs"], args.nagents, args.hammer, args.meslen)
    
    # writer = SummaryWriter(logdir=os.path.join(args.logdir, expname)) 
    MAIN = args.hammer 


    betas = (0.9, 0.999)
    local_state_dim = obs_dim + args.meslen if MAIN else obs_dim  

    if args.sharedparams: 
        print("Using Shared Parameters")
        local_agent = LocalPolicy(
            state_dim=local_state_dim, 
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
    else: 
        print("Using Independent Parameters") 
        local_agent = [LocalPolicy(
            state_dim=local_state_dim, 
            action_dim=action_dim,
            actor_layer=config["local"]["actor_layer"],
            critic_layer=config["local"]["critic_layer"],
            lr=config["local"]["lr"],
            betas=betas,
            gamma=config["main"]["gamma"],
            K_epochs=config["local"]["K_epochs"],
            eps_clip=config["main"]["eps_clip"], 
            shared=False 
        ) for _ in range(args.nagents)] 

    def load(i, path):
        local_agent[i].policy_old.load_state_dict(torch.load(path))

    if args.prevactions: 
        print("Using Previous Actions") 
    global_state_dim = (obs_dim * args.nagents) + args.nagents if args.prevactions else (obs_dim * args.nagents)
    global_agent = GlobalPolicy(
        state_dim=global_state_dim, # all local observations concatenated + all agents' previous actions
        action_dim=args.meslen*args.nagents, 
        n_agents=args.nagents, # required for discrete messages
        action_std=config["global"]["action_std"],
        lr=config["global"]["lr"],
        betas=betas,
        gamma = config["main"]["gamma"],
        K_epochs=config["global"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"],        
        actor_layer=config["global"]["actor_layer"],
        critic_layer=config["global"]["critic_layer"],
        is_discrete = args.discretemes
    )

    load(0, "runs/2021/new_shared/save-dir/350000_cn----L-lr-0.0003-updatestep-800-epoch-8----G-lr-0.0003-updatestep-800-epoch-8----nagents-3-hammer-1-meslen-4/local_agent.pth")
    load(1, "runs/2021/new_shared/save-dir/350000_cn----L-lr-0.0003-updatestep-800-epoch-8----G-lr-0.0003-updatestep-800-epoch-8----nagents-3-hammer-1-meslen-4/local_agent.pth")
    load(2, "runs/2021/new_shared/save-dir/350000_cn----L-lr-0.0003-updatestep-800-epoch-8----G-lr-0.0003-updatestep-800-epoch-8----nagents-3-hammer-1-meslen-4/local_agent.pth")
    global_agent.policy_old.load_state_dict(torch.load("runs/2021/new_shared/save-dir/350000_cn----L-lr-0.0003-updatestep-800-epoch-8----G-lr-0.0003-updatestep-800-epoch-8----nagents-3-hammer-1-meslen-4/global_agent.pth"))


    # logging variables
    ep_reward = 0
    local_timestep = 0
    global_timestep = 0

    if args.heterogeneity: 
        obs = preprocess_one_obs(env.reset(), limit=args.limit)
    elif args.partialobs: 
        obs = preprocess_obs(env.reset(), limit=args.limit)
    else:  
        obs = env.reset() 

    global_agent_state = [obs[i] for i in obs]
    global_agent_state = np.array(global_agent_state).reshape((-1,)) 
    if args.prevactions: 
        global_agent_state = np.concatenate([global_agent_state, np.random.randint(0, action_dim, args.nagents)])
    i_episode = 0
    episode_rewards = 0
    avg_returns = []
    agents = [agent for agent in env.agents] 
    actor_loss = [0 for agent in agents]
    critic_loss = [0 for agent in agents]
    episodic_messages = [[] for agent in agents]

    for timestep in count(1):
        env.render()
        if MAIN: 
            if args.randommes: 
                global_agent_output = np.random.uniform(0, 1, args.nagents*args.meslen) 
            else: 
                global_agent_output, global_agent_log_prob = global_agent.select_action(global_agent_state)
            global_agent_output = global_agent_output.reshape(args.nagents, args.meslen) 
            append_message(global_agent_output, episodic_messages) 

        action_array = [] 
        for i, agent in enumerate(agents): 
            local_state = np.concatenate([obs[agent], global_agent_output[i]]) if MAIN else obs[agent]
            if args.sharedparams: 
                action, local_log_prob = local_agent.policy_old.act(local_state) 
            else: 
                action, local_log_prob = local_agent[i].policy_old.act(local_state) 

            action_array.append(action) 

        actions = {agent : action_array[i] for i, agent in enumerate(agents)}

        next_obs, rewards, is_terminals, infos = env.step(actions) 
        if args.partialobs: 
            next_obs = preprocess_obs(next_obs, limit=args.limit) 
        elif args.heterogeneity: 
            next_obs = preprocess_one_obs(next_obs, limit=args.limit) 

        for i, agent in enumerate(agents):
            episode_rewards += rewards[agent]

        if MAIN and (not args.randommes): 
            global_agent_output = global_agent_output.reshape(-1) 

        obs = next_obs
        

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            i_episode += 1
            if MAIN:
                analyse_message(args.discretemes, episodic_messages)
            # writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards/args.nagents, i_episode)
            if args.heterogeneity: 
                obs = preprocess_one_obs(env.reset(), limit=args.limit) 
            elif args.partialobs: 
                obs = preprocess_obs(env.reset(), limit=args.limit) 
            else: 
                obs = env.reset() 
            avg_returns.append(episode_rewards/args.nagents)
            print('Episode {} \t  return: {} \t Avg return: {}'.format(i_episode, episode_rewards/args.nagents, sum(avg_returns)/len(avg_returns)))
            episode_rewards = 0
            episodic_messages = [[] for agent in agents]
        
        global_agent_state = np.array([obs[agent] for agent in agents]).reshape((-1,))
        if args.prevactions: 
            prev_actions = np.array([actions[agent] for agent in agents]).reshape((-1,))
            global_agent_state = np.concatenate([global_agent_state, prev_actions])
            global_agent_state = global_agent_state

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/cn/hyperparams.yaml', help="config file name")
    parser.add_argument("--load", type=bool, default=False, help="load true / false") 

    parser.add_argument("--hammer", type=int, default=1, help="1 for hammer; 0 for IL")
    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--nagents", type=int, default=3)

    parser.add_argument("--maxepisodes", type=int, default=500_000) 
    parser.add_argument("--prevactions", type=int, default=0) 
    parser.add_argument("--partialobs", type=int, default=1) 
    parser.add_argument("--sharedparams", type=int, default=0) 
    parser.add_argument("--heterogeneity", type=int, default=0) 
    parser.add_argument("--limit", type=int, default=10) 
    parser.add_argument("--maxcycles", type=int, default=250) 
    parser.add_argument("--randommes", type=int, default=0) 


    parser.add_argument("--meslen", type=int, default=4, help="message length")
    parser.add_argument("--discretemes", type=int, default=1)
    parser.add_argument("--randomseed", type=int, default=10)
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--saveinterval", type=int, default=1000) 
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args)
