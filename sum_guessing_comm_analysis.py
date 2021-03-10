import argparse
from itertools import count

from tensorboardX import SummaryWriter

from ppo_with_gradient import PPO

from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_reference_v2
from utils import read_config
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.cm as colormap 
import torch
import json 
from scipy.stats import entropy  

from pathlib import Path

from guessing_sum_game import GuessingSumEnv

def plot(obs, actions, messages): 
    columns = ["index", "obs1", "obs2", "action1", "action2", "message"]
    df = pd.DataFrame(columns=columns) 
    
    for i in range(len(obs)): 
        point = [0, obs[i]["Agent0"][0], obs[i]["Agent1"][0], actions[i]["Agent0"][0], actions[i]["Agent1"][0]] 
        point.extend(messages[i][0]) 
        df = df.append(pd.DataFrame([point], columns=columns)) 
    df["obs_sum"] = df["obs1"] + df["obs2"] 
    df["action1_err"] = abs(df["action1"] - df["obs_sum"]) 
    df["action2_err"] = abs(df["action2"] - df["obs_sum"]) 
    # print(df) 
    
    # Mean and Standard Deviation 
    # print(df["action1_err"].mean(), df["action1_err"].std()) # 0.7698002555758323, 0.5571670516844823 in 10000 eps 
    # print(df["action2_err"].mean(), df["action2_err"].std()) # 0.7838607136086739, 0.5664230119884279 in 10000 eps 
    
    # Message trend for both obs1 and obs2 
    df.plot.scatter(x="obs1", y="obs2", c="message", colormap="viridis") 
    plt.gca().invert_yaxis() 
    plt.savefig('obs_vs_obs_vs_message.png') 
    
    # # Message trend for obs1 
    # df.plot.scatter(x="index", y="obs1", c="message", colormap="viridis") 
    # plt.savefig('sumgame_analysis/obs1_vs_message.png') 
    
    # # Message trend for obs2 
    # df.plot.scatter(x="index", y="obs2", c="message", colormap="viridis") 
    # plt.savefig('sumgame_analysis/obs2_vs_message.png') 
    
    
    
    # # Message trend for obs_sum 
    # df.plot.scatter(x="index", y="obs_sum", c="message", colormap="viridis") 
    # plt.savefig('sumgame_analysis/obssum_vs_message.png') 



    # # Message trend for both action1 and action2 
    # df.plot.scatter(x="action1", y="action2", c="message", colormap="viridis") 
    # plt.savefig('sumgame_analysis/action_vs_action_vs_message.png') 

    # # Message trend for action1 
    # df.plot.scatter(x="index", y="action1", c="message", colormap="viridis") 
    # plt.savefig('sumgame_analysis/action1_vs_message.png') 
    
    # # Message trend for action2 
    # df.plot.scatter(x="index", y="action2", c="message", colormap="viridis") 
    # plt.savefig('sumgame_analysis/action2_vs_message.png') 
    
    # print(entropy(pk=df["message"])) 

    # plt.show() 

def plot_discrete(obs, actions, messages): 
    # print(obs) 
    # print(actions) 
    x=[]
    for i in messages: 
        # print(i)
        temp = ""
        for k in i: 
            for n in k: 
                temp = temp+str(1) if n>=0.5 else temp+str(0) 
        x.append(temp) 
    messages = x 
    print(messages)

    # columns = ["index", "obs1", "obs2", "action1", "action2", "message"]
    # df = pd.DataFrame(columns=columns) 
    
    # for i in range(len(obs)): 
    #     point = [0, obs[i]["Agent0"][0], obs[i]["Agent1"][0], actions[i]["Agent0"][0], actions[i]["Agent1"][0]] 
    #     point.extend(messages[i][0]) 
    #     df = df.append(pd.DataFrame([point], columns=columns)) 
    # # df["obs_sum"] = df["obs1"] + df["obs2"] 
    # df["action1_err"] = abs(df["action1"] - df["obs_sum"]) 
    # df["action2_err"] = abs(df["action2"] - df["obs_sum"]) 

    # print(df) 

def run(args):
    
    SCALE = 1.0
    env = GuessingSumEnv(num_agents=args.nagents, scale=SCALE, discrete=False)

    env.reset()
    agents = env.agents

    obs_dim = 1
        
    action_dim = 1 

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
    HAMMER.load(args.load)
    
    if args.dru_toggle: 
        print("Using DRU") 
    else: 
        print("Not Using DRU")

    # logging variables
    ep_reward = 0
    global_timestep = 0

    env.reset() 
    # For continuous observations 
    obs_set = [env.reset() for _ in range(10000)] 

    # # For discrete observations 
    # obs_set = [{"Agent0":np.array([i]), "Agent1":np.array([i])} for i in range(10)] 

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
        # m_sums = [] 
        # for i in messages: 
        #     m_sums.append(sum(i))
        # print(messages) 
        # print(rewards) 
        # print() 
        message_set.append(messages) 
    plot(obs=obs_set, actions=action_set, messages=message_set) 
    
    # plot_discrete(obs=obs_set, actions=action_set, messages=message_set) 
    
    # print(np.mean(diff_set, axis=0)) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/guesser/hyperparams.yaml', help="config file name")
    parser.add_argument("--load", type=str, default="sumguesser-save-dir/1_guesser--nagents2--dru0--meslen1--rs--999") 

    parser.add_argument("--nagents", type=int, default=2)
    parser.add_argument("--maxepisodes", type=int, default=1) 
    parser.add_argument("--dru_toggle", type=int, default=0) 
    parser.add_argument("--meslen", type=int, default=1, help="message length")
    parser.add_argument("--randomseed", type=int, default=99)

    args = parser.parse_args() 
    run(args=args)
