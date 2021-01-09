import argparse
import pickle

import torch
import numpy as np
from tensorboardX import SummaryWriter

from ppo import Memory, PPO
from utils import read_config


def main(config=None):
    if not config:
        print("config required")
        return
    from ma_envs.make_env import make_env
    import os
    ############## Hyperparameters ##############
    env_name = config["env_name"]
    exp_name = config["exp_name"]
    writer = SummaryWriter(logdir=os.path.join(config["logdir"], exp_name))
    # creating environment
    # env = gym.make(env_name)
    env = make_env(scenario_name="simple_spread", benchmark=False)
    message_len = config["message_len"]
    state_dim = env.observation_space[0].shape[0] + message_len
    action_dim = env.action_space[0].n
    NUM_AGENTS = env.n
    render = False
    log_interval = 20  # print avg reward in the interval
    save_interval = 50 # save model in the interval
    max_episodes = config["max_episodes"]  # max training episodes
    max_timesteps = config["max_steps"]  # max timesteps in one episode
    n_latent_var = config["n_latent_var"]  # number of variables in hidden layer
    update_timestep = config["update_timestep"]  # update policy every n timesteps
    lr = config["lr"]
    betas = (0.9, 0.999)
    gamma = config["gamma"]  # discount factor
    K_epochs = config["K_epochs"]  # update policy for K epochs
    eps_clip = config["eps_clip"]  # clip parameter for PPO
    random_seed = config["random_seed"]
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    ep_rew_array = []
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        ep_rew_array.append(0)
        for t in range(max_timesteps):
            timestep += 1
            state = [np.concatenate([s, np.random.uniform(low=-1, high=1, size=message_len)]) for s in state]

            # Running policy_old:
            action_masks = [np.zeros(action_dim) for _ in range(NUM_AGENTS)]
            for i in range(NUM_AGENTS):
                action = ppo.policy_old.act(state[i], memory)
                action_masks[i][action] = 1.0

            state, reward, done, _ = env.step(action_masks)

            # Saving reward and is_terminal:
            for i in range(NUM_AGENTS):
                memory.rewards.append(reward[i])
                memory.is_terminals.append(done[i])

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += np.mean(reward)
            ep_rew_array[-1] += np.mean(reward)
            if render:
                env.render()
            if all(done):
                break

        # save every 50 episodes
        if i_episode % save_interval == 0:
            if not os.path.exists(os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/", exp_name)):
                os.makedirs(os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/", exp_name))
            torch.save(ppo.policy.state_dict(), os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/", exp_name, "checkpoint.pth"))

        # logging
        if i_episode % log_interval == 0:
            running_reward = int((running_reward / log_interval))

            writer.add_scalar('Running Reward', running_reward, i_episode)

            if not os.path.exists(os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/", exp_name)):
                os.makedirs(os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/", exp_name))
            with open(os.path.join("../../Desktop/others/ccdb_backup/plots/save-dir/", exp_name, 'rewards.pkl'), 'wb') as f:
                pickle.dump(ep_rew_array, f)

            print('Episode {} \t reward: {}'.format(i_episode, running_reward))
            running_reward = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Reinforcement Learning experiments: Config Generation")
    parser.add_argument("--config", type=str, default=None, help="config file name")
    args = parser.parse_args()
    main(config=read_config(args.config))

