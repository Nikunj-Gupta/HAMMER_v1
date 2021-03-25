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

def plot(obs, actions, messages, discrete_mes=False, nagents=3): 
    columns = ["index"] 
    for i in range(nagents): 
    
        columns.append("obs"+str(i+1)) 
        columns.append("action"+str(i+1)) 
        columns.append("message"+str(i+1)) 
    
    df = pd.DataFrame(columns=columns) 
    for k in range(len(obs)): 
        point=[0] 
        for i in range(nagents): 
            point.append(list(obs[k].values())[i][0]) 
            point.append(list(actions[k].values())[i][0]) 
            point.append(messages[k][i][0]) 
        df = df.append(pd.DataFrame([point], columns=columns)) 

    print(df) 
    df["obs_sum"] = df["obs1"] + df["obs2"] + df["obs3"] 
    df["obs12_sum"] = df["obs1"] + df["obs2"] 
    df["obs13_sum"] = df["obs1"] + df["obs3"] 
    df["obs32_sum"] = df["obs3"] + df["obs2"] 

    df["action1_err"] = abs(df["action1"] - df["obs_sum"]) 
    df["action2_err"] = abs(df["action2"] - df["obs_sum"]) 
    df["action3_err"] = abs(df["action3"] - df["obs_sum"]) 

    
    savedir = "sumgame_analysis/multiply/2agents" 
    savedir = "sumgame_analysis/3agents/" 
    if not os.path.exists(savedir): os.makedirs(savedir)
    # x="obs1"
    # y="obs2"
    # c="message1"
    # name="--".join([x,y,c]) 

    plots = [
        # {
        #     "x": "obs1", 
        #     "y": "obs2", 
        #     "c": "message1" 
        # }, 
        
        # {
        #     "x": "obs1", 
        #     "y": "obs2", 
        #     "c": "message2" 
        # }, 

        {
            "x": "obs2", 
            "y": "obs3", 
            "c": "message1" 
        }, 

        # {
        #     "x": "obs1", 
        #     "y": "obs3", 
        #     "c": "message1" 
        # }, 

        {
            "x": "obs1", 
            "y": "obs32_sum", 
            "c": "message1" 
        }, 

        # {
        #     "x": "obs2", 
        #     "y": "obs13_sum", 
        #     "c": "message2" 
        # }, 

        # {
        #     "x": "obs3", 
        #     "y": "obs12_sum", 
        #     "c": "message3" 
        # }, 
        # {
        #     "x": "obs1", 
        #     "y": "obs13_sum", 
        #     "c": "message1" 
        # }, 

        # {
        #     "x": "obs1", 
        #     "y": "obs12_sum", 
        #     "c": "message1" 
        # }, 


    ]

    for p in plots: 
        name = "--".join([p["x"],p["y"],p["c"]]) 
        df.plot.scatter(x=p["x"], y=p["y"], c=p["c"], colormap="viridis") 
        plt.savefig(os.path.join(savedir, name+".png")) 
        # plt.show() 

    
    
    """ 
    Previous 
    """
    # columns = ["index", "obs1", "obs2", "action1", "action2", "message1", "message2"]
    # df = pd.DataFrame(columns=columns) 

    # for i in range(len(obs)): 
    #     point = [0, obs[i]["Agent0"][0], obs[i]["Agent1"][0], actions[i]["Agent0"][0], actions[i]["Agent1"][0]] 
    #     point.extend(messages[i][0]) 
    #     point.extend(messages[i][1]) 
    #     df = df.append(pd.DataFrame([point], columns=columns)) 
    
    # df["obs_sum"] = df["obs1"] + df["obs2"] 
    # df["action1_err"] = abs(df["action1"] - df["obs_sum"]) 
    # df["action2_err"] = abs(df["action2"] - df["obs_sum"]) 

    # # print(df) 
    
    # # Mean and Standard Deviation 
    # print(df["action1_err"].mean(), df["action1_err"].std()) # 0.7698002555758323, 0.5571670516844823 in 10000 eps 
    # print(df["action2_err"].mean(), df["action2_err"].std()) # 0.7838607136086739, 0.5664230119884279 in 10000 eps 
    
    # # Message trend for both obs1 and obs2 
    # d = "sumgame_analysis/somemore_discrete/" 
    
    
    # if discrete_mes: 
    #     for m in ["message1", "message2"]: 
    #         df.loc[df[m] < 0.5, m] = 0 
    #         df.loc[df[m] >= 0.5, m] = 1 
    
    # df.plot.scatter(x="message1", y="message2", c="obs_sum", colormap="viridis") 
    # plt.savefig(d+'1.png') 
    
    # df.plot.scatter(x="message1", y="message2", c="action1", colormap="viridis") 
    # plt.savefig(d+'4.png') 

    # df.plot.scatter(x="message1", y="message2", c="action2", colormap="viridis") 
    # plt.savefig(d+'5.png') 

    # df.plot.scatter(x="obs1", y="message2", c="action1", colormap="viridis") 
    # plt.savefig(d+'2.png') 

    # df.plot.scatter(x="obs2", y="message1", c="action2", colormap="viridis") 
    # plt.savefig(d+'3.png') 

    # df.plot.scatter(x="obs2", y="message1", c="obs_sum", colormap="viridis") 
    # plt.savefig(d+'6.png') 

    # df.plot.scatter(x="obs1", y="message2", c="obs_sum", colormap="viridis") 
    # plt.savefig(d+'7.png') 

    # df.plot.scatter(x="obs1", y="obs2", c="obs_sum", colormap="viridis") 
    # plt.savefig(d+'8.png') 

    # df.plot.scatter(x="obs1", y="obs2", c="action1", colormap="viridis") 
    # plt.savefig(d+'9.png') 

    # df.plot.scatter(x="obs1", y="obs2", c="action2", colormap="viridis") 
    # plt.savefig(d+'10.png') 

    # plt.gca().invert_yaxis() 
    # for m in ["message1", "message2"]: 
    
        # df.plot.scatter(x="obs1", y="obs2", c=m, colormap="viridis") 
        
    #     # plt.gca().invert_yaxis() 
        # plt.savefig(d+'obs_vs_obs_vs_'+m+'-initial.png') 
        
    #     # Message trend for obs1 
    #     df.plot.scatter(x="index", y="obs1", c=m, colormap="viridis") 
    #     plt.savefig(d+'obs1_vs_'+m+'.png') 
        
    #     # Message trend for obs2 
    #     df.plot.scatter(x="index", y="obs2", c=m, colormap="viridis") 
    #     plt.savefig(d+'obs2_vs_'+m+'.png') 
        
        
        
    #     # Message trend for obs_sum 
    #     df.plot.scatter(x="index", y="obs_sum", c=m, colormap="viridis") 
    #     plt.savefig(d+'obssum_vs_'+m+'.png') 



    #     # Message trend for both action1 and action2 
    #     df.plot.scatter(x="action1", y="action2", c=m, colormap="viridis") 
    #     plt.savefig(d+'action_vs_action_vs_'+m+'.png') 

    #     # Message trend for action1 
    #     df.plot.scatter(x="index", y="action1", c=m, colormap="viridis") 
    #     plt.savefig(d+'action1_vs_'+m+'.png') 
        
    #     # Message trend for action2 
    #     df.plot.scatter(x="index", y="action2", c=m, colormap="viridis") 
    #     plt.savefig(d+'action2_vs_'+m+'.png') 
        
        # print(entropy(pk=df["message1"])) 


    # plt.show() 


def run(args):
    
    SCALE = 10.0
    env = GuessingSumEnv(num_agents=args.nagents, scale=SCALE, discrete=0)

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
        sharedparams=0,
        is_discrete=0
    ) 
    HAMMER.load(args.load)
    HAMMER.policy_old.action_var[0] = 1e-10
    log_dir = Path('test/eval')
    for i in count(0):
        temp = log_dir/('run{}'.format(i)) 
        if temp.exists():
            pass
        else:
            writer = SummaryWriter(logdir=temp)
            log_dir = temp
            break
    
    if args.dru_toggle: 
        print("Using DRU") 
    else: 
        print("Not Using DRU")

    # logging variables
    ep_reward = 0
    global_timestep = 0

    env.reset() 

    obs_set = [] 
    action_set = [] 
    message_set = [] 
    diff_set = [] 
    ep_rew = [] 
    obs = env.reset() 

    i_episode = -1
    episode_rewards = 0
    for timestep in count(1):

        action_array = [] 

        actions, messages = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory) 
        obs_set.append(obs)
        action_set.append(actions) 
        message_set.append(messages) 
        next_obs, rewards, is_terminals, infos = env.step(actions) 

        HAMMER.memory_record(rewards, is_terminals) 
        episode_rewards += np.mean(list(rewards.values())) 
        ep_rew.append(episode_rewards) 

        obs = next_obs

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            i_episode += 1
            writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards, i_episode)
            obs = env.reset() 
            print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards))
            episode_rewards = 0

        if i_episode == args.maxepisodes:
            break 


    print(np.mean(ep_rew)) 

    plot(obs=obs_set, actions=action_set, messages=message_set, discrete_mes=args.dru_toggle, nagents=args.nagents) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/guesser/hyperparams.yaml', help="config file name")
    parser.add_argument("--load", type=str, default="test/save-dir/guesser--nagents1--dru0--meslen0--rs--99/checkpoint_50000") 

    parser.add_argument("--nagents", type=int, default=1)
    parser.add_argument("--maxepisodes", type=int, default=10000) 
    parser.add_argument("--dru_toggle", type=int, default=0) 
    parser.add_argument("--meslen", type=int, default=0, help="message length")
    parser.add_argument("--randomseed", type=int, default=99)

    args = parser.parse_args() 
    run(args=args)
