import argparse 
from pettingzoo.mpe import simple_spread_v2
from pprint import pprint 
from time import sleep 

parser = argparse.ArgumentParser()
parser.add_argument("--nagents", type=int, default=3)
args = parser.parse_args() 



env = simple_spread_v2.parallel_env(N=args.nagents, local_ratio=0.5, max_cycles=100) 
pprint(env.reset()) 
# env.render() 
# sleep(100)
obs_space = env.observation_spaces 
print(obs_space)
obs_dim = env.observation_spaces[env.agents[0]].shape[0]
action_dim = env.action_spaces[env.agents[0]].n
agent_action_space = env.action_spaces[env.agents[0]]

# for i in range(100): 
#     env.reset()
#     env.render()