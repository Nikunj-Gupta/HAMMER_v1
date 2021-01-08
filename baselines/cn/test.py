# from ray import tune
# from ray.rllib.agents.ppo import PPOTrainer
# # tune.run(PPOTrainer, config={"env": "CartPole-v0", "framework":"torch"})  
#                                                                             # "log_level": "INFO" for verbose,
#                                                                             # "framework": "tfe"/"tf2" for eager,
#                                                                             # "framework": "torch" for PyTorch


# from ray.tune.registry import register_env
# from pettingzoo.mpe import simple_spread_v2
# from ray.rllib.env import PettingZooEnv
# env_creator = simple_spread_v2.parallel_env(N=5, local_ratio=0.5, max_cycles=25)
# register_env('CN', PettingZooEnv(env_creator({"N": 5, "local_ratio":0.5, "max_cycles":25})))

from copy import deepcopy
from numpy import float32

import os
# from supersuit import normalize_obs_v0, dtype_v0, color_reduction_v0

import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import PettingZooEnv
from pettingzoo.mpe import simple_spread_v2 
from gym.spaces import Discrete 

from ray.tune.registry import register_env

if __name__ == "__main__":
    """For this script, you need:
    1. Algorithm name and according module, e.g.: "PPo" + agents.ppo as agent
    2. Name of the aec game you want to train on, e.g.: "pistonball".
    3. num_cpus
    4. num_rollouts
    Does require SuperSuit
    """ 
    
    
    alg_name = "PPO" #  
    # function that outputs the environment you wish to register.
    def env_creator(config):
        env = simple_spread_v2.env(N=config.get("N", 5), local_ratio=config.get("local_ratio", 0.5), )
        # env = dtype_v0(env, dtype=float32)
        # env = color_reduction_v0(env, mode="R")
        # env = normalize_obs_v0(env)
        return env

    num_cpus = 1
    num_rollouts = 2

    # 1. Gets default training configuration and specifies the POMgame to load.
    config = deepcopy(get_agent_class(alg_name)._default_config) 
    config["framework"]="torch"

    # 2. Set environment config. This will be passed to
    # the env_creator function via the register env lambda below.
    config["env_config"] = {"N":5, "local_ratio": 0.5}

    # 3. Register env
    register_env("cooperativenavigation",
                 lambda config: PettingZooEnv(env_creator(config)))

    # 4. Extract space dimensions
    test_env = PettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    # 5. Configuration for multiagent setup with policy sharing:
    config["multiagent"] = {
        "policies": {
            # the first tuple value is None -> uses default policy
            "av": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: "av"
    }

    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["log_level"] = "DEBUG"
    config["num_workers"] = 1
    # Fragment length, collected at once from each worker and for each agent!
    config["rollout_fragment_length"] = 30
    # Training batch size -> Fragments are concatenated up to this point.
    config["train_batch_size"] = 200
    # After n steps, force reset simulation
    config["horizon"] = 200
    # Default: False
    config["no_done_at_end"] = False
    # Info: If False, each agents trajectory is expected to have
    # maximum one done=True in the last step of the trajectory.
    # If no_done_at_end = True, environment is not resetted
    # when dones[__all__]= True.

    # 6. Initialize ray and trainer object
    ray.init(num_cpus=num_cpus + 1)
    trainer = get_agent_class(alg_name)(env="cooperativenavigation", config=config)

    # 7. Train once
    trainer.train()

    test_env.reset()
