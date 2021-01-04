import argparse

from tensorboardX import SummaryWriter

from ppo import PPO as LocalPolicy
from ppo_cont import PPO as GlobalPolicy
from ppo import Memory

from ma_envs.make_env import make_env
from utils import read_config
import os
import numpy as np
import torch


def run(config=None):

    env = make_env(scenario_name="simple_spread", benchmark=False)
    print("Number of Agents: ", env.n)
    if not config:
        print("config required")
        return

    random_seed = config["main"]["random_seed"]
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        # torch.manual_seed(random_seed)
        env.seed(random_seed)
        # np.random.seed(random_seed)

    writer = SummaryWriter(logdir=os.path.join(config["main"]["logdir"], config["main"]["exp_name"]))
    local_memory = Memory()
    global_memory = Memory()

    betas = (0.9, 0.999)
    local_agent = LocalPolicy(
        state_dim=env.observation_space[0].shape[0] + config["main"]["message_len"],
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
        action_dim=config["main"]["message_len"],
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
    max_episodes = config["main"]["max_episodes"]
    max_timesteps = config["main"]["max_timesteps"]
    NUM_AGENTS = env.n

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()

        prev_actions = np.random.randint(0, env.action_space[0].n, env.n)
        global_agent_state = np.array(state).reshape((-1,))
        global_agent_state = np.concatenate([global_agent_state, prev_actions])
        global_agent_state = [global_agent_state] * env.n

        for t in range(max_timesteps):
            local_timestep += 1
            global_timestep += 1
            global_agent_action = []
            for i in range(NUM_AGENTS):
                global_agent_action.append(global_agent.select_action(global_agent_state[i], global_memory))
            global_agent_action = np.array(global_agent_action)

            state = np.array([np.concatenate([state[i], global_agent_action[i]]) for i in range(NUM_AGENTS)])

            # Running policy_old:
            action_masks = [np.zeros(env.action_space[0].n) for _ in range(NUM_AGENTS)]
            prev_actions = []
            for i in range(NUM_AGENTS):
                action = local_agent.policy_old.act(state[i], local_memory)
                action_masks[i][action] = 1.0
                prev_actions.append(action)

            next_state, reward, done, info = env.step(action_masks)
            ep_reward += np.mean(reward)

            global_agent_next_state = np.array(next_state).reshape((-1,))
            global_agent_next_state = np.concatenate([global_agent_next_state, prev_actions])
            global_agent_next_state = [global_agent_next_state] * env.n

            for i in range(NUM_AGENTS):
                global_memory.rewards.append(reward[i])
                global_memory.is_terminals.append(done[i])
                local_memory.rewards.append(reward[i])
                local_memory.is_terminals.append(done[i])

            state = next_state
            global_agent_state = global_agent_next_state

            # update if its time
            if local_timestep % config["local"]["update_timestep"] == 0:
                local_agent.update(local_memory)
                local_memory.clear_memory()
                local_timestep = 0

            if global_timestep % config["global"]["update_timestep"] == 0:
                global_agent.update(global_memory)
                global_memory.clear_memory()
                global_timestep = 0

            if config["main"]["render"]:
                env.render()

            if all(done):
                break

        if i_episode % config["main"]["save_interval"] == 0:
            if not os.path.exists(os.path.join(config["main"]["save_dir"], config["main"]["exp_name"])):
                os.makedirs(os.path.join(config["main"]["save_dir"], config["main"]["exp_name"]))
            torch.save(local_agent.policy.state_dict(), os.path.join(config["main"]["save_dir"], config["main"]["exp_name"], "local_agent.pth"))
            torch.save(global_agent.policy.state_dict(), os.path.join(config["main"]["save_dir"], config["main"]["exp_name"], "global_agent.pth"))

        writer.add_scalar('Episode Reward', ep_reward, i_episode)
        print("Episode ", i_episode, " Reward: ", ep_reward)
        ep_reward = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="config file name")
    parser.add_argument("--load", type=bool, default=False, help="load true / false")
    args = parser.parse_args()
    run(config=read_config(args.config))
