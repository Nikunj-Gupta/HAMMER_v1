import argparse

from tensorboardX import SummaryWriter

from local_agents.ppo_discrete import PPO as LocalPolicy
from global_messenger.ppo import PPO as GlobalPolicy
from ppo import Memory

from ma_envs.make_env import make_env
from utils import read_config
import os
import numpy as np
import torch


def run(args):

    env = make_env(scenario_name="simple_spread", benchmark=False)
    print("Number of Agents: ", env.n)
    config = read_config(args.config) 
    if not config:
        print("config required")
        return
    
    random_seed = args.randomseed 
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)

    writer = SummaryWriter(logdir=os.path.join(args.logdir, args.expname)) 
    local_memory = [Memory() for _ in range(env.n)]
    global_memory = Memory()
    MAIN = args.hammer


    betas = (0.9, 0.999)
    local_state_dim = env.observation_space[0].shape[0] + args.meslen if MAIN else env.observation_space[0].shape[0] 

    local_agent = LocalPolicy(
        state_dim=local_state_dim, 
        action_dim=env.action_space[0].n,
        n_latent_var=config["local"]["n_latent_var"],
        lr=config["local"]["lr"],
        betas=betas,
        gamma=config["main"]["gamma"],
        K_epochs=config["local"]["K_epochs"],
        eps_clip=config["main"]["eps_clip"]
    )

    global_agent = GlobalPolicy(
        state_dim=(env.observation_space[0].shape[0] * env.n) + env.n,  # all local observations concatenated + all agents' previous actions
        action_dim=args.meslen, 
        n_agents=env.n,
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
    max_episodes = args.maxepisodes
    max_timesteps = args.maxtimesteps
    NUM_AGENTS = env.n

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()

        prev_actions = np.random.randint(0, env.action_space[0].n, env.n)
        global_agent_state = np.array(state).reshape((-1,))
        global_agent_state = np.concatenate([global_agent_state, prev_actions])

        for t in range(max_timesteps):
            local_timestep += 1
            global_timestep += 1

            if MAIN: 
                global_agent_output, global_agent_log_prob = global_agent.select_action(global_agent_state) 
                state = np.array([np.concatenate([state[i], global_agent_output[i]]) for i in range(NUM_AGENTS)])   

            # Running policy_old:
            action_masks = [np.zeros(env.action_space[0].n) for _ in range(NUM_AGENTS)]
            prev_actions = []
            for i in range(NUM_AGENTS):
                action, local_log_prob = local_agent.policy_old.act(state[i])

                action_masks[i][action] = 1.0 

                # Storing in states, actions, logprobs in local_memory:
                local_memory[i].states.append(state[i])
                local_memory[i].actions.append(action)
                local_memory[i].logprobs.append(local_log_prob)

                prev_actions.append(action) 

            next_state, reward, done, info = env.step(action_masks)
            ep_reward += np.mean(reward)

            global_agent_next_state = np.array(next_state).reshape((-1,))
            global_agent_next_state = np.concatenate([global_agent_next_state, prev_actions])

            for i in range(NUM_AGENTS):
                local_memory[i].rewards.append(reward[i])
                local_memory[i].is_terminals.append(done[i]) 
           
            if MAIN: 
                global_memory.states.append(global_agent_state)
                global_memory.actions.append(global_agent_output)
                global_memory.logprobs.append(global_agent_log_prob)
                global_memory.rewards.append([reward[agent] for agent in range(NUM_AGENTS)]) 
                global_memory.is_terminals.append([done[agent] for agent in range(NUM_AGENTS)]) 

            
            # update if its time
            if local_timestep % config["local"]["update_timestep"] == 0:
                local_agent.update(local_memory) 
                [mem.clear_memory() for mem in local_memory]

            if MAIN and global_timestep % config["global"]["update_timestep"] == 0:
                global_agent.update(global_memory)
                global_memory.clear_memory()

            if args.render:
                env.render()

            if all(done):
                break 

            state = next_state
            global_agent_state = global_agent_next_state


        if i_episode % args.saveinterval == 0:
            if not os.path.exists(os.path.join(args.savedir, args.expname)):
                os.makedirs(os.path.join(args.savedir, args.expname))
            torch.save(local_agent.policy.state_dict(), os.path.join(args.savedir, args.expname, "local_agent.pth"))
            torch.save(global_agent.policy.state_dict(), os.path.join(args.savedir, args.expname, "global_agent.pth"))

        writer.add_scalar('Episode Reward', ep_reward, i_episode)
        print("Episode ", i_episode, " Reward: ", ep_reward)
        ep_reward = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="config file name")
    parser.add_argument("--load", type=bool, default=False, help="load true / false") 

    parser.add_argument("--hammer", type=int, default=1, help="1 for hammer; 0 for IL")
    parser.add_argument("--expname", type=str, default=None)

    parser.add_argument("--maxepisodes", type=int, default=30000) 
    parser.add_argument("--maxtimesteps", type=int, default=25)

    parser.add_argument("--meslen", type=int, default=4, help="message length")
    parser.add_argument("--randomseed", type=int, default=10)
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--loginterval", type=int, default=20)
    parser.add_argument("--saveinterval", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args)
