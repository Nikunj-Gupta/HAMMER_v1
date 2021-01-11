import argparse
from itertools import count

from tensorboardX import SummaryWriter

from local_agents.ppo_discrete import PPO as LocalPolicy
from global_messenger.ppo import PPO as GlobalPolicy
from local_agents.ppo_discrete import Memory

from pettingzoo.mpe import simple_spread_v2
from utils import read_config
import os
import numpy as np
import torch
import json 


def run(args):

    env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=25) 
    env.reset()
    obs_space = env.observation_spaces 
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

    expname = args.expname if args.expname is not None else 'cn----L-lr-{}-updatestep-{}-epoch-{}----G-lr-{}-updatestep-{}-epoch-{}----nagents-{}-hammer-{}-meslen-{}'.format(config["local"]["lr"], config["local"]["update_timestep"], config["local"]["K_epochs"], config["global"]["lr"], config["global"]["update_timestep"], config["global"]["K_epochs"], args.nagents, args.hammer, args.meslen)
    
    writer = SummaryWriter(logdir=os.path.join(args.logdir, expname)) 
    local_memory = [Memory() for _ in range(args.nagents)]
    global_memory = Memory()
    MAIN = args.hammer


    betas = (0.9, 0.999)
    local_state_dim = obs_dim + args.meslen if MAIN else obs_dim  

    local_agent = LocalPolicy(
        state_dim=local_state_dim, 
        action_dim=action_dim,
        n_latent_var=config["local"]["n_latent_var"],
        lr=config["local"]["lr"],
        betas=betas,
        gamma=config["main"]["gamma"],
        K_epochs=config["local"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"]
    )

    global_agent = GlobalPolicy(
        state_dim=(obs_dim * args.nagents) + args.nagents,  # all local observations concatenated + all agents' previous actions
        action_dim=args.meslen, 
        n_agents=args.nagents,
        action_std=config["global"]["action_std"],
        lr=config["global"]["lr"],
        betas=betas,
        gamma = config["main"]["gamma"],
        K_epochs=config["global"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"],
        hidden_nodes=config["global"]["hidden_nodes"]
    )

    # logging variables
    ep_reward = 0
    local_timestep = 0
    global_timestep = 0

    obs = env.reset() 
    global_agent_state = [obs[i] for i in obs]
    global_agent_state = np.array(global_agent_state).reshape((-1,))
    global_agent_state = np.concatenate([global_agent_state, np.random.randint(0, action_dim, args.nagents)])
    i_episode = 0
    episode_rewards = 0
    agents = [agent for agent in env.agents] 
    actor_loss = [0 for agent in agents]
    critic_loss = [0 for agent in agents]


    for timestep in count(1):
        if MAIN: 
            global_agent_output, global_agent_log_prob = global_agent.select_action(global_agent_state) 

        for i, agent in enumerate(agents):
            local_state = np.concatenate([obs[agent], global_agent_output[i]]) if MAIN else obs[agent]
            action, local_log_prob = local_agent.policy_old.act(local_state)

            local_memory[i].states.append(local_state)
            local_memory[i].actions.append(action)
            local_memory[i].logprobs.append(local_log_prob)

        actions = {agent : local_memory[i].actions[-1] for i, agent in enumerate(agents)}
        next_obs, rewards, is_terminals, infos = env.step(actions)

        for i, agent in enumerate(agents):
            local_memory[i].rewards.append(rewards[agent])
            local_memory[i].is_terminals.append(is_terminals[agent])
            episode_rewards += rewards[agent]

        if MAIN: 
            global_memory.states.append(global_agent_state)
            global_memory.actions.append(global_agent_output)
            global_memory.logprobs.append(global_agent_log_prob)
            global_memory.rewards.append([rewards[agent] for agent in agents])
            global_memory.is_terminals.append([is_terminals[agent] for agent in agents])
            

        # update if its time
        if timestep % config["local"]["update_timestep"] == 0:
            local_agent.update(local_memory, writer, i_episode)
            [mem.clear_memory() for mem in local_memory]

        if MAIN and timestep % config["global"]["update_timestep"] == 0:
            global_agent.update(global_memory, writer, i_episode)
            global_memory.clear_memory()

        obs = next_obs

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            i_episode += 1
            writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards/args.nagents, i_episode)
            obs = env.reset()
            print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards/args.nagents))
            episode_rewards = 0

        # save every 50 episodes
        if i_episode % args.saveinterval == 0:
            if not os.path.exists(os.path.join(args.savedir, expname)):
                os.makedirs(os.path.join(args.savedir, expname))
            torch.save(local_agent.policy.state_dict(),
                    os.path.join(args.savedir, expname, "local_agent.pth"))
            torch.save(global_agent.policy.state_dict(),
                    os.path.join(args.savedir, expname, "global_agent.pth"))
        
        if i_episode == args.maxepisodes:
            break
        
        global_agent_state = np.array([obs[agent] for agent in agents]).reshape((-1,))
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

    parser.add_argument("--maxepisodes", type=int, default=30000) 

    parser.add_argument("--meslen", type=int, default=4, help="message length")
    parser.add_argument("--randomseed", type=int, default=10)
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--saveinterval", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args)
