import argparse
from itertools import count

from tensorboardX import SummaryWriter

from ppo_with_gradient import PPO

from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_reference_v2
from utils import read_config
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.cm as colormap 
import torch
import json 

from pathlib import Path

from guessing_obs_game import GuessingSumEnv

def plot(obs, actions, messages): 
    columns = ["index", "obs1", "obs2", "action1", "action2", "message_1", "message_2", "message_3", "message_4", "message_sum"]
    df = pd.DataFrame(columns=columns) 
    
    for i in range(len(obs)): 
        point = [0, obs[i]["Agent0"][0], obs[i]["Agent1"][0], actions[i]["Agent0"][0], actions[i]["Agent0"][1]] 
        point.extend(messages[i][0]) 
        point.append(sum(messages[i][0])) 
        df = df.append(pd.DataFrame([point], columns=columns)) 
    df["diff"] = abs(df["action2"] - df["obs2"]) 
    # print(df) 
    # print(df["diff"].mean())
    # print(df["diff"].std()) 
    # fig, axes = plt.subplots(2, 2, subplot_kw={'xlim': (0,0), 'ylim': (-4.0,4.0)}) 
    # df.plot.scatter(x="obs1", y="obs2", c="message_1", ax=axes[0,0], vmin=-0.6, vmax=0.6) 
    # df.plot.scatter(x="obs1", y="obs2", c="message_2", ax=axes[0,1], vmin=-0.6, vmax=0.6) 
    # df.plot.scatter(x="obs1", y="obs2", c="message_3", ax=axes[1,0], vmin=-0.6, vmax=0.6) 
    # df.plot.scatter(x="obs1", y="obs2", c="message_4", ax=axes[1,1], vmin=-0.6, vmax=0.6) 
    # df.plot.scatter(x="obs1", y="obs2", c="message_sum") 



    o = "obs1" 
    vmin = -0.8 
    vmax = 1.0 
    df.plot.scatter(x="index", y=o, c="message_sum") 
    # df.plot.scatter(x="index", y=o, c="message_1", ax=axes[0,0], vmin=vmin, vmax=vmax) 
    # df.plot.scatter(x="index", y=o, c="message_2", ax=axes[0,1], vmin=vmin, vmax=vmax) 
    # df.plot.scatter(x="index", y=o, c="message_3", ax=axes[1,0], vmin=vmin, vmax=vmax) 
    # df.plot.scatter(x="index", y=o, c="message_4", ax=axes[1,1], vmin=vmin, vmax=vmax) 

    # df.plot.scatter(x="obs1", y="action1", ax=axes[0,0]) 
    # df.plot.scatter(x="obs1", y="action2", ax=axes[0,1]) 
    # df.plot.scatter(x="obs2", y="action1", ax=axes[1,0]) 
    # df.plot.scatter(x="obs2", y="action2", ax=axes[1,1]) 
    # df.plot.scatter(x="action1", y="action2", c="message_3", ax=axes[1,0]) 
    # df.plot.scatter(x="action1", y="action2", c="message_4", ax=axes[1,1]) 
    # df.plot.scatter(x="action1", y="action2", c="message_sum", ax=axes[1,2]) 

    # df.plot.scatter(x="obs1", y="diff", ax=axes[1,2]) 
    plt.show() 

def run(args):
    
    SCALE = 10.0
    env = GuessingSumEnv(num_agents=args.nagents, scale=SCALE, discrete=False)

    env.reset()
    agents = env.agents

    obs_dim = 1
        
    action_dim = args.nagents

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

    betas = (0.9, 0.999)

    HAMMER = PPO(
        agents=agents,
        single_state_dim=obs_dim,
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
        dru_toggle=args.dru_toggle,
        sharedparams=1,
        is_discrete=0
    ) 

    HAMMER.policy_old.load_state_dict(torch.load(str(os.path.join(args.load, "local_agent.pth")))) 
    
    if args.dru_toggle: 
        print("Using DRU") 
    else: 
        print("Not Using DRU")

    # logging variables
    ep_reward = 0
    global_timestep = 0

    env.reset() 
    obs_set = [env.reset() for _ in range(1000)] 
    action_set = [] 
    message_set = [] 
    diff_set = [] 
    for obs in obs_set: 
        # print(obs) 
        action_array = [] 
        actions, messages = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory) 
        # print(actions) 
        action_set.append(actions) 
        next_obs, rewards, is_terminals, infos = env.step(actions) 
        if args.dru_toggle: 
            x=[]
            for i in messages: 
                temp = ""
                for n in i:                 
                    temp = temp+str(1) if n>0.5 else temp+str(0)
                x.append(temp) 
            messages = x 
        m_sums = [] 
        for i in messages: 
            m_sums.append(sum(i))
        # print(m_sums)
        diff = [
            abs(obs["Agent0"]) - abs(m_sums[1]), 
            abs(obs["Agent1"]) - abs(m_sums[0]), 
        ]
        diff_set.append(diff) 
        # print(diff) 
        # print(messages) 
        # print(rewards) 
        # print() 
        message_set.append(messages) 
    plot(obs=obs_set, actions=action_set, messages=message_set) 
    # print(np.mean(diff_set, axis=0)) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/guesser/hyperparams.yaml', help="config file name")
    parser.add_argument("--load", type=str, default="guesser-save-dir-new/50000_guesser--nagents2--dru0--meslen4--rs--999") 

    parser.add_argument("--nagents", type=int, default=2)
    parser.add_argument("--maxepisodes", type=int, default=1) 
    parser.add_argument("--dru_toggle", type=int, default=0) 
    parser.add_argument("--meslen", type=int, default=20, help="message length")
    parser.add_argument("--randomseed", type=int, default=999)

    args = parser.parse_args() 
    run(args=args)
