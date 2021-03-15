import argparse, pprint
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from maddpg.utils.buffer import ReplayBuffer
from maddpg.algorithms.maddpg import MADDPG
from pettingzoo.mpe import simple_spread_v2


def run(config):
    torch.manual_seed(config.seed)

    np.random.seed(config.seed)
    env = simple_spread_v2.parallel_env(local_ratio=0.5)
    env.seed(config.seed)
    env.reset()
    logger = SummaryWriter(str(config.log_dir)+config.expname) 

    agent_alg=config.agent_alg 
    adversary_alg=config.adversary_alg 
    tau=config.tau 
    gamma=0.95 
    lr=config.lr 
    hidden_dim=config.hidden_dim 

    """ 
    Local Agents 
    """ 

    agent_init_params = []
    alg_types = [adversary_alg if agent.count('adversary') else agent_alg for agent in env.agents] 
    for acsp, obsp, algtype in zip(env.action_spaces.values(), env.observation_spaces.values(), alg_types): 
            num_in_pol = obsp.shape[0] + config.meslen 
            discrete_action = True
            get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            num_in_critic = obsp.shape[0] + config.meslen + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
    init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                    'hidden_dim': hidden_dim,
                    'alg_types': alg_types,
                    'agent_init_params': agent_init_params,
                    'discrete_action': discrete_action} 

    local_agents = MADDPG(**init_dict) 
    local_agents.init_dict = init_dict 

    local_replay_buffer = ReplayBuffer(
            config.buffer_length, 
            local_agents.nagents, 
            [obsp.shape[0]+config.meslen for obsp in env.observation_spaces.values()], 
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_spaces.values()]
        ) 



    """ 
    Global Agent --- HAMMER 
    """ 

    agent_init_params = []
    alg_types = [agent_alg] 
    acsp = config.meslen * env.num_agents 
    obsp = list(env.observation_spaces.values())[0].shape[0]*env.num_agents 
    num_in_pol = obsp 
    discrete_action = False
    num_out_pol = acsp
    num_in_critic = obsp + acsp 
    agent_init_params.append({'num_in_pol': num_in_pol,
                                'num_out_pol': num_out_pol,
                                'num_in_critic': num_in_critic})
    init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                    'hidden_dim': hidden_dim,
                    'alg_types': alg_types,
                    'agent_init_params': agent_init_params,
                    'discrete_action': discrete_action}

    global_agent = MADDPG(**init_dict) 
    global_agent.init_dict = init_dict 

    global_replay_buffer = ReplayBuffer(
            config.buffer_length, 
            global_agent.nagents, 
            [obsp], 
            [acsp]
        ) 
    





    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads): 
        obs = env.reset()

        # maddpg.prep_rollouts(device='cpu')
        local_agents.prep_rollouts(device='cpu') 
        global_agent.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps

        # maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        # maddpg.reset_noise()
        local_agents.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining) 
        local_agents.reset_noise() 
        global_agent.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining) 
        global_agent.reset_noise() 

        # print(list(np.array(list(obs.values())).reshape(-1))) 

        # concatenate local observations for Hammer's global agent, and converting it to torch Variable     
        global_torch_obs = [Variable(torch.Tensor(list(np.array(list(obs.values())).reshape(-1))).unsqueeze(dim=0), requires_grad=False) 
                for i in range(global_agent.nagents)] 
        # get message (global action) as torch Variable 
        global_agent_torch_action = global_agent.step(global_torch_obs, explore=True) 
        # convert global action to numpy array and rearrange to be per local agent 
        global_agent_action = [global_ac.data.numpy().reshape(local_agents.nagents, config.meslen) 
                                    for global_ac in global_agent_torch_action] 
        
        for i in range(local_agents.nagents): 
            local_obs = np.concatenate([list(obs.values())[i], global_agent_action[0][i]]) 
            obs[list(obs.keys())[i]] = local_obs 

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable 
            torch_obs = [Variable(torch.Tensor(list(obs.values())[i]).unsqueeze(dim=0),
                                  requires_grad=False)
                         for i in range(local_agents.nagents)] 
            
            # get actions as torch Variables
            torch_agent_actions = local_agents.step(torch_obs, explore=True) 

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be as per environment
            actions = {agent : np.argmax(ac[0]) for ac, agent in zip(agent_actions, env.agents)}
            
            # get next_obs, rewards and dones for local agents  
            next_obs, rewards, dones, infos = env.step(actions) 

            # calculate next_obs, rewards and dones for global agent 
            global_agent_reward = np.mean(list(rewards.values())) 
            global_agent_done = all(list(dones.values())) 
            
            global_replay_buffer.push(
                np.array([global_torch_obs[0].data.numpy()]), 
                [np.array([global_ac.reshape(-1)]) for global_ac in global_agent_action], 
                np.array([global_agent_reward]).reshape(1, -1), 
                np.array([[np.array(list(next_obs.values())).reshape(-1)]]), 
                np.array([global_agent_done]).reshape(1, -1) 
            )
            
            # Steps to update global and local agents' next_obs 
            global_torch_obs = [Variable(torch.Tensor(list(np.array(list(next_obs.values())).reshape(-1))).unsqueeze(dim=0), requires_grad=False) 
                for i in range(global_agent.nagents)] 
            # global agent's next action required for updating local agent's next_obs 
            # get message (global action) as torch Variable 
            global_agent_torch_action = global_agent.step(global_torch_obs, explore=True) 
            # convert global action to numpy array and rearrange to be per local agent 
            global_agent_action = [global_ac.data.numpy().reshape(local_agents.nagents, config.meslen) 
                                        for global_ac in global_agent_torch_action] 
            
            
            # update local agents' next_obs using global agent's next action 
            for i in range(local_agents.nagents): 
                local_obs = np.concatenate([list(next_obs.values())[i], global_agent_action[0][i]]) 
                next_obs[list(next_obs.keys())[i]] = local_obs 
            
            local_replay_buffer.push(
                np.array([[obs[i] for i in obs]]), 
                agent_actions, 
                np.array(list(rewards.values())).reshape(1, -1), 
                np.array([[next_obs[i] for i in next_obs]]), 
                np.array(list(dones.values())).reshape(1, -1)
            ) 

            obs = next_obs 

            t += config.n_rollout_threads
            if (len(local_replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads): 

                global_agent.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(global_agent.nagents):
                        sample = global_replay_buffer.sample(config.batch_size, to_gpu=False) 
                        global_agent.update(sample, a_i) 
                    global_agent.update_all_targets() 
                global_agent.prep_rollouts(device='cpu') 

                local_agents.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(local_agents.nagents):
                        sample = local_replay_buffer.sample(config.batch_size, to_gpu=False) 
                        local_agents.update(sample, a_i) 
                    local_agents.update_all_targets()
                local_agents.prep_rollouts(device='cpu') 
        ep_rews = local_replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads) 
        logger.add_scalar('mean_episode_reward', np.mean(ep_rews), ep_i) 
        
        if ep_i % config.save_interval < config.n_rollout_threads: 
            local_agents.save(os.path.join(config.log_dir, 'local_model.pt')) 
            global_agent.save(os.path.join(config.log_dir,'global_model.pt')) 

        print("Episodes %i-%i of %i --> Team Reward: %i " % (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.n_episodes, ep_rews[0])) 



    env.close()
    local_agents.save(os.path.join(config.log_dir, 'local_model.pt'))
    global_agent.save(os.path.join(config.log_dir, 'global_model.pt')) 
    logger.export_scalars_to_json(str(config.log_dir / 'summary.json'))
    logger.close()






    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int, help="Random seed") 
    parser.add_argument("--expname", default="testrun", type=str) 
    parser.add_argument("--log_dir", default="test/logs/", type=str) 

    parser.add_argument("--meslen", default=1, type=int, help="HAMMER message length")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=100000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="DDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="DDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--continuous_action",
                        action='store_false')

    config = parser.parse_args()

    run(config)
