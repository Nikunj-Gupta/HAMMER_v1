import argparse
from itertools import count
from tensorboardX import SummaryWriter

from local_agents.ppo_discrete import PPO as LocalPolicy 
from local_agents.ppo_discrete import Memory 
from pettingzoo.mpe import simple_spread_v2

from utils import read_config

import os
import numpy as np
import torch
import json 



def run(args):

    env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=args.maxcycles) 

    env.reset() 

    obs_space = env.observation_spaces 
    obs_dim = (env.observation_spaces[env.agents[0]].shape[0]) * args.nagents 
    action_dim = env.action_spaces[env.agents[0]].n if args.envname == "cn" else 5 
    agent_action_space = env.action_spaces[env.agents[0]] 

    config = read_config(args.config) 
    if not config:
        print("config required")
        return
    
    random_seed = args.randomseed 
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        # env.seed(random_seed) 
        env.reset(seed=random_seed) 
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    expname = args.expname if args.expname is not None else args.envname 
    
    writer = SummaryWriter(logdir=os.path.join(args.logdir, expname)) 
    local_memory = [Memory() for _ in range(args.nagents)]

    betas = (0.9, 0.999)
    local_state_dim = obs_dim  

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

    # logging variables 
    ep_reward = 0
    local_timestep = 0

    obs = env.reset() 
    obs = np.array([obs[agent] for agent in obs]).reshape(-1) 
    
    i_episode = 0
    episode_rewards = 0
    agents = [agent for agent in env.agents] 
    actor_loss = [0 for agent in agents]
    critic_loss = [0 for agent in agents]

    for timestep in count(1):
        action_array = [] 
        for i, agent in enumerate(agents): 
            local_state = obs 
            action, local_log_prob = local_agent.policy_old.act(local_state) 

            action_array.append(action) 
            local_memory[i].states.append(local_state)
            local_memory[i].actions.append(action)
            local_memory[i].logprobs.append(local_log_prob) 
        
        actions = {agent : action_array[i] for i, agent in enumerate(agents)}  

        next_obs, rewards, _, is_terminals, infos = env.step(actions) 
        next_obs = np.array([next_obs[agent] for agent in next_obs]).reshape(-1) 
    

        for i, agent in enumerate(agents):
            local_memory[i].rewards.append(rewards[agent])
            local_memory[i].is_terminals.append(is_terminals[agent])
            episode_rewards += rewards[agent]

        # update if its time
        if timestep % config["local"]["update_timestep"] == 0: 
            for i in range(args.nagents): 
                local_agent.update(local_memory[i], writer, i_episode)
            [mem.clear_memory() for mem in local_memory]

        obs = next_obs

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            i_episode += 1
            writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards/args.nagents, i_episode)
            obs = env.reset() 
            obs = np.array([obs[agent] for agent in obs]).reshape(-1) 
            print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards/args.nagents))
            episode_rewards = 0

        # save every 50 episodes
        if i_episode % args.saveinterval == 0:
            if not os.path.exists(os.path.join(args.savedir, str(i_episode)+"_"+expname)):
                os.makedirs(os.path.join(args.savedir, str(i_episode)+"_"+expname))
            torch.save(local_agent.policy.state_dict(),
                os.path.join(args.savedir, str(i_episode)+"_"+expname, "local_agent.pth")) 
        
        if i_episode == args.maxepisodes:
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/cn/hyperparams.yaml', help="config file name")
    parser.add_argument("--load", type=bool, default=False, help="load true / false") 

    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--envname", type=str, default="cn")
    parser.add_argument("--nagents", type=int, default=3)

    parser.add_argument("--maxepisodes", type=int, default=30_000) 
    parser.add_argument("--maxcycles", type=int, default=25) 

    parser.add_argument("--randomseed", type=int, default=10)
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--saveinterval", type=int, default=5000) 
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args)