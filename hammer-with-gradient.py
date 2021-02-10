import argparse
from itertools import count

from tensorboardX import SummaryWriter

from ppo_with_gradient import PPO

from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_reference_v2
from utils import read_config
import os
import numpy as np
import torch
import json 

def preprocess_obs(obs, limit): 
    for i in obs: 
        obs[i] = obs[i][:limit] 
    return obs 

def run(args):
    if args.envname == 'cn':
        if args.partialobs == 1:
            assert args.limit == 10
        env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=args.maxcycles)
    elif args.envname == 'sr':
        if args.partialobs == 1:
            assert args.limit == 11
        env = simple_reference_v2.parallel_env(local_ratio=0.5, max_cycles=args.maxcycles)

    env.reset()
    agents = [agent for agent in env.agents] 

    if args.partialobs:
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


    expname = args.envname if args.expname == None else args.expname
    
    writer = SummaryWriter(logdir=os.path.join(args.logdir, expname)) 

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
    )

    # logging variables
    ep_reward = 0
    global_timestep = 0

    if args.partialobs: 
        obs = preprocess_obs(env.reset(), limit=args.limit)
    else:  
        obs = env.reset() 

    i_episode = 0
    episode_rewards = 0

    for timestep in count(1):

        action_array = [] 
        actions = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory)
        
        next_obs, rewards, is_terminals, infos = env.step(actions) 

        HAMMER.memory_record(rewards, is_terminals)
        episode_rewards += list(rewards.values())[0]         
        # update if its time
        if timestep % config["global"]["update_timestep"] == 0: 
            HAMMER.update()
            [mem.clear_memory() for mem in HAMMER.memory]
            HAMMER.global_memory.clear_memory()

        if args.partialobs: 
            next_obs = preprocess_obs(next_obs, limit=args.limit) 
        obs = next_obs

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            i_episode += 1
            writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards, i_episode)
            if args.partialobs: 
                obs = preprocess_obs(env.reset(), limit=args.limit)
            else: 
                obs = env.reset() 
            print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards))
            episode_rewards = 0

        # save every 50 episodes
        if i_episode % args.saveinterval == 0:
            if not os.path.exists(os.path.join(args.savedir, str(i_episode)+"_"+expname)):
                os.makedirs(os.path.join(args.savedir, str(i_episode)+"_"+expname))
            torch.save(HAMMER.policy.state_dict(),
                    os.path.join(args.savedir, str(i_episode)+"_"+expname, "local_agent.pth"))        
        if i_episode == args.maxepisodes:
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/cn/hyperparams.yaml', help="config file name")

    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--envname", type=str, default='cn')
    parser.add_argument("--nagents", type=int, default=3)

    parser.add_argument("--maxepisodes", type=int, default=500_000) 
    parser.add_argument("--partialobs", type=int, default=1) 
    parser.add_argument("--limit", type=int, default=10) # 11 for sr, 10 for cn
    parser.add_argument("--maxcycles", type=int, default=25) 

    parser.add_argument("--meslen", type=int, default=2, help="message length")
    parser.add_argument("--randomseed", type=int, default=10)

    parser.add_argument("--saveinterval", type=int, default=50_000) 
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args)
