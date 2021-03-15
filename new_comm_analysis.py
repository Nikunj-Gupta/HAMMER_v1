import argparse
from itertools import count

from tensorboardX import SummaryWriter

from ppo_with_gradient_old import PPO

from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe import simple_reference_v2
from utils import read_config
import os
import numpy as np, pandas as pd 
import torch
import json, matplotlib.pyplot as plt 
from sklearn.decomposition import PCA


from pathlib import Path

def preprocess_one_obs(obs, which=1, limit=10): 
    agent = "agent_" + str(which) 
    obs[agent][limit:] = [0.]*(len(obs["agent_0"])-(limit)) 
    return obs 


def preprocess_obs(obs, limit): 
    for i in obs: 
        obs[i] = obs[i][:limit] 
    return obs 

def analyze(messages, maxcycles=25, save_name=None): # messages vs timestep in episode (averaged over 1000 eps)
    for agent in range(3): 
        arr = [] 
        for count in range(0, len(messages), maxcycles): 
            temp=[] 
            for i in messages[count:count+maxcycles]: 
                temp.append(i[agent][0]) 
            arr.append(temp) 
        arr = np.array(arr).mean(axis=0) 
        # plt.subplot(1, 3, agent+1)
        plt.plot(arr, label="agent_"+str(agent)) 
        # plt.ylim((0.4, 0.6))

    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 6) 
    plt.legend() 

    if save_name:
        plt.savefig(os.path.join("hammer_commanalysis_plots", save_name+'--analyze1.png')) 
        plt.close()
    else: 
        plt.show() 


def analyze2(messages, save_name=None): 
    m = {"agent_0":[], "agent_1":[], "agent_2":[],}
    for agent in range(3): 
        temp=[] 
        for i in messages: 
            m["agent_"+str(agent)].append(i[agent][0]) 
        plt.scatter([agent]*len(m["agent_"+str(agent)]), m["agent_"+str(agent)], marker="o") 
    
    if save_name:
        plt.savefig(os.path.join("hammer_commanalysis_plots", save_name+'--analyze2.png')) 
        plt.close()
    else: 
        plt.show() 

def analyze3(messages, maxcycles=50, save_name=None): 
    arr=[] 
    for i in messages: 
        arr.append(i[0]) 
        arr.append(i[1]) 
        arr.append(i[2]) 

    # print(arr) 
    arr = np.array(arr)
    from sklearn.cluster import KMeans 
    kmeans = KMeans(n_clusters=3, random_state=0).fit(arr) 
    print("cluster centers: ", kmeans.cluster_centers_) 
    print("cluster labels: ", kmeans.labels_) 

    for agent in range(3): 
        arr = [] 
        for count in range(0, len(messages), maxcycles): 
            temp=[] 
            for i in messages[count:count+maxcycles]: 
                temp.append(i[agent][0]) 
            arr.append(temp) 
        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(arr) 
        # print(principalComponents) 
        plt.subplot(1, 3, agent+1)
        plt.scatter(principalComponents[:,0], principalComponents[:,1]) 
        # plt.xlim(-5, 5) 
        # plt.ylim(-5, 5) 


    if save_name: 
        plt.savefig(os.path.join("hammer_commanalysis_plots", save_name+'--analyze3.png')) 
        plt.close()
    else: 
        plt.show() 

def analyze4(messages, maxcycles=50, save_name=None): # correlation between messages for different local agents 
    arr={
        "agent_0":[], 
        "agent_1":[], 
        "agent_2":[], 

    } 
    for i in messages: 
        arr["agent_0"].append(i[0][0]) 
        arr["agent_1"].append(i[1][0]) 
        arr["agent_2"].append(i[2][0]) 

    df = pd.DataFrame(arr) 
    plt.matshow(df.corr())
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns)
    cb = plt.colorbar()

    cb.ax.tick_params()
    # plt.title('Correlation Matrix') 
    if save_name:
        plt.savefig(os.path.join("hammer_commanalysis_plots", save_name+'--analyze4.png')) 
        plt.close()
    else: 
        plt.show() 

def analyze5(messages, maxcycles=50, save_name=None): # scatter plot of messages at 0, 25, 50 timesteps of episodes 
    for x in [0, 25, 49]: 
        arr={
            "agent_0":[], 
            "agent_1":[], 
            "agent_2":[], 

        } 
        for i in range(x, len(messages), maxcycles): 
          
            arr["agent_0"].append(messages[i][0][0]) 
            arr["agent_1"].append(messages[i][1][0]) 
            arr["agent_2"].append(messages[i][2][0]) 

        df = pd.DataFrame(arr) 
        for agent in range(3): 
            plt.subplot(1, 3, agent+1) 
            plt.scatter([x]*len(df["agent_"+str(agent)]), df["agent_"+str(agent)]) 
    figure = plt.gcf() # get current figure
    figure.set_size_inches(15, 6) 
    plt.legend() 
    if save_name:
        plt.savefig(os.path.join("hammer_commanalysis_plots", save_name+'--analyze5.png')) 
        plt.close()
    else: 
        plt.show() 
    
def analyze6(messages, maxcycles=50, save_name=None): # histograms (distributions) of learned messages for all local agents 
    arr={
        "agent_0":[], 
        "agent_1":[], 
        "agent_2":[], 

    } 
    for i in range(len(messages)): 
        
        arr["agent_0"].append(messages[i][0][0]) 
        arr["agent_1"].append(messages[i][1][0]) 
        arr["agent_2"].append(messages[i][2][0]) 

    df = pd.DataFrame(arr) 
    print(df.describe()) 
 
    if save_name:
        df.plot.kde().get_figure().savefig(os.path.join("hammer_commanalysis_plots", save_name+'--analyze7.png')) 
        # df.plot.hist(stacked=True, bins=20, alpha=0.5).get_figure().savefig(os.path.join("hammer_commanalysis_plots", save_name+'--analyze6.png')) 
        plt.close()
    else: 
        plt.show() 

def analyze8(messages, positions, maxcycles=50, save_name=None): 
    arr={
        "agent_0":[], 
        "agent_1":[], 
        "agent_2":[], 

    } 
    for i in range(len(messages)): 
        
        arr["agent_0"].append([positions[i][0][0], positions[i][0][1], messages[i][0][0]]) 
        arr["agent_1"].append([positions[i][1][0], positions[i][1][1], messages[i][1][0]]) 
        arr["agent_2"].append([positions[i][2][0], positions[i][2][1], messages[i][2][0]]) 

    for agent in range(3): 
        plt.subplot(1, 3, agent+1) 
        df = pd.DataFrame(arr['agent_'+str(agent)], columns=["x", "y", "message"]) 
        df.plot.scatter(x="x", y="y", c="message", s=50).get_figure().savefig(os.path.join("hammer_commanalysis_plots", save_name+"--agent_"+str(agent)+'--analyze8.png')) 
 
def analyze9(messages, agent_positions, landmark_positions, maxcycles=50, save_name=None): 
    arr={
        "agent_0":[], 
        "agent_1":[], 
        "agent_2":[], 

    } 
    for i in range(len(messages)): 
        for agent in range(3): 
            arr["agent_"+str(agent)].append([landmark_positions[i][agent][0], \
                landmark_positions[i][agent][1], \
                landmark_positions[i][agent][2], \
                landmark_positions[i][agent][3], \
                landmark_positions[i][agent][4], \
                landmark_positions[i][agent][5], \
                    messages[i][agent][0]]) 


    for agent in range(3): 
        df = pd.DataFrame(arr['agent_'+str(agent)], columns=["l1_x", "l1_y", "l2_x", "l2_y", "l3_x", "l3_y", "message"]) 
        for landmark in range(3):     
            df.plot.scatter(x="l"+str(landmark+1)+"_x", \
                y="l"+str(landmark+1)+"_y", \
                    c="message", s=50).get_figure().savefig(os.path.join(\
                        "hammer_commanalysis_plots", save_name+"--agent_"+str(agent)+"--landmark_"+str(landmark)+'--analyze9.png')) 
        df=[]

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
        print("Using Partial Observations") 
    
    if args.heterogeneity: 
        print("Using Heterogeneous Local Agents") 

    if args.heterogeneity: 
        assert args.limit == 10 
        obs_dim = len(preprocess_one_obs(env.reset(), limit=args.limit)["agent_0"]) 
    elif args.partialobs:
        obs_dim = len(preprocess_obs(env.reset(), limit=args.limit)["agent_0"]) 
    else:
        obs_dim = env.observation_spaces[env.agents[0]].shape[0]
        
    action_dim = env.action_spaces[env.agents[0]].n if args.envname == "cn" else 5 

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
    
    # writer = SummaryWriter(logdir=os.path.join(args.logdir, expname)) 

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
        dru_toggle=args.dru_toggle, 
        is_discrete=1, 
    ) 

    HAMMER.policy_old.load_state_dict(torch.load(str(os.path.join(args.load, "local_agent.pth")))) 


    if args.dru_toggle: 
        print("Using DRU") 
    else: 
        print("Not Using DRU")

    # logging variables
    ep_reward = 0
    global_timestep = 0

    if args.heterogeneity: 
        obs = preprocess_one_obs(env.reset(), limit=args.limit) 
    elif args.partialobs: 
        obs = preprocess_obs(env.reset(), limit=args.limit)
    else:  
        obs = env.reset() 

    i_episode = 0
    episode_rewards = 0 

    all_messages = [] 
    all_distances = [] 
    all_agent_positions = [] 
    all_landmark_relative_positions = [] 

    for timestep in count(1): 
        # env.render() 
        action_array = [] 
        actions, messages = HAMMER.policy_old.act(obs, HAMMER.memory, HAMMER.global_memory)
        all_messages.append(messages) 

        all_agent_positions.append([[obs[agent][2], obs[agent][3]] for agent in agents])
        all_landmark_relative_positions.append([[obs[agent][4], obs[agent][5], obs[agent][6], obs[agent][7], obs[agent][8], obs[agent][9]] \
            for agent in agents])

        next_obs, rewards, is_terminals, infos = env.step(actions) 

        HAMMER.memory_record(rewards, is_terminals)
        episode_rewards += list(rewards.values())[0] 

        if args.partialobs: 
            next_obs = preprocess_obs(next_obs, limit=args.limit) 
        elif args.heterogeneity: 
            next_obs = preprocess_one_obs(next_obs, limit=args.limit) 
        obs = next_obs 

        # If episode had ended
        if all([is_terminals[agent] for agent in agents]):
            i_episode += 1
            # writer.add_scalar('Avg reward for each agent, after an episode', episode_rewards, i_episode) 
            if args.heterogeneity: 
                obs = preprocess_one_obs(env.reset(), limit=args.limit) 
            elif args.partialobs: 
                obs = preprocess_obs(env.reset(), limit=args.limit)
            else: 
                obs = env.reset() 
            print('Episode {} \t  Avg reward for each agent, after an episode: {}'.format(i_episode, episode_rewards))
            episode_rewards = 0

        if i_episode == args.maxepisodes: 
            print(len(all_messages)) 
            # analyze(all_messages, maxcycles=args.maxcycles, save_name=args.expname) 
            # analyze2(all_messages, save_name=args.expname) 
            # analyze(all_messages, maxcycles=args.maxcycles) 
            # analyze2(all_messages) 
            # analyze3(all_messages, maxcycles=args.maxcycles, save_name=args.expname) 
            # analyze4(all_messages, save_name=args.expname) 
            # analyze5(all_messages, save_name=args.expname)
            # analyze6(all_messages, save_name=args.expname) 
            # analyze8(all_messages, all_agent_positions, save_name=args.expname) 
            analyze9(all_messages, all_agent_positions, all_landmark_relative_positions, save_name=args.expname) 

            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/cn/hyperparams.yaml', help="config file name") 
    parser.add_argument("--load", type=str, default=None, help="load path") 


    parser.add_argument("--expname", type=str, default=None) 
    parser.add_argument("--envname", type=str, default='cn') 
    parser.add_argument("--nagents", type=int, default=3) 

    parser.add_argument("--maxepisodes", type=int, default=10) 
    parser.add_argument("--partialobs", type=int, default=0) 

    parser.add_argument("--heterogeneity", type=int, default=0) 
    parser.add_argument("--limit", type=int, default=10) # 11 for sr, 10 for cn
    parser.add_argument("--maxcycles", type=int, default=25) 

    parser.add_argument("--dru_toggle", type=int, default=1) 

    parser.add_argument("--meslen", type=int, default=1, help="message length")
    parser.add_argument("--randomseed", type=int, default=10)

    parser.add_argument("--saveinterval", type=int, default=50_000) 
    parser.add_argument("--logdir", type=str, default="logs/", help="log directory path")
    parser.add_argument("--savedir", type=str, default="save-dir/", help="save directory path")
    

    args = parser.parse_args() 
    print(args.savedir)
    run(args=args)
