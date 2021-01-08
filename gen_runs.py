import os, glob


# ==============================================================================
# Hammer and IL on Cooperative Navigation 
# ==============================================================================

# code = "../../../main_complete_state_and_prev_actions.py"
# config = "../../../configs/2021/cn/hyperparams.yaml" 

# fixed_text = "#!/bin/bash\n"\
#                     "#SBATCH --nodes=1\n"\
#                     "#SBATCH --gres=gpu:2\n"\
#                     "#SBATCH --ntasks-per-node=32\n"\
#                     "#SBATCH --mem=127000M\n"\
#                     "#SBATCH --time=00-30:00\n"\
#                     "#SBATCH --account=def-mtaylor3\n"\
#                     "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 

# for seed in range(9, 11): 
#     for meslen in range(4, 6): 
#         expname = "cn-hammer-nagents-3-rs-"+str(seed)+"-meslen-"+str(meslen) 
#         command = " ".join([
#             "python", code, 
#             "--config", config, 
#             "--hammer", str(hammer), 
#             "--expname", expname, 
#             "--maxepisodes", "100000", 
#             "--randomseed", str(seed), 
#             "--meslen", str(meslen)
#         ]) 
#         with open(os.path.join("runs/2021/cn", expname + ".sh"), "w") as f:
#             f.write(fixed_text + command)


# ==============================================================================
# IL on Cooperative Navigation 
# ==============================================================================


# fixed_text = "#!/bin/bash\n"\
#                     "#SBATCH --nodes=1\n"\
#                     "#SBATCH --gres=gpu:2\n"\
#                     "#SBATCH --ntasks-per-node=32\n"\
#                     "#SBATCH --mem=127000M\n"\
#                     "#SBATCH --time=00-06:00\n"\
#                     "#SBATCH --account=def-mtaylor3\n"\
#                     "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 


# for seed in range(9, 11): 
#     expname = "cn-il-nagents-3--rs-"+str(seed) 
#     command = " ".join([
#         "python", code, 
#         "--config", config, 
#         "--hammer", "0", 
#         "--expname", expname, 
#         "--maxepisodes", "100000", 
#         "--randomseed", str(seed)
#     ]) 

#     with open(os.path.join("runs/2021/cn", expname + ".sh"), "w") as f:
#         f.write(fixed_text + command)

# ==============================================================================
# Hammer on Multi-Agent Walker  
# ==============================================================================

code = "../../../multiwalker.py"
config = "../../../configs/2021/mw/hyperparameters.yaml" 

fixed_text = "#!/bin/bash\n"\
                    "#SBATCH --nodes=1\n"\
                    "#SBATCH --gres=gpu:2\n"\
                    "#SBATCH --ntasks-per-node=32\n"\
                    "#SBATCH --mem=127000M\n"\
                    "#SBATCH --time=00-30:00\n"\
                    "#SBATCH --account=def-mtaylor3\n"\
                    "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 

for seed in range(9, 11): 
    for meslen in range(4, 6): 
        for nagent in range(3, 6, 2): 
            expname = "mw-hammer-rs-"+"-nagents-"+str(nagent)+str(seed)+"-meslen-"+str(meslen) 
            command = " ".join([
                "python", code, 
                "--config", config, 
                "--hammer", "1", 
                "--expname", expname, 
                "--nwalkers", str(nagent), 
                "--maxepisodes", "100000", 
                "--randomseed", str(seed), 
                "--meslen", str(meslen)
            ]) 
            with open(os.path.join("runs/2021/mw", expname + ".sh"), "w") as f:
                f.write(fixed_text + command)



# ==============================================================================
# IL on Multi-Agent Walker  
# ==============================================================================

for seed in range(9, 11): 
    for nagent in range(3, 6, 2): 
        expname = "mw-il-nagents-"+str(nagent)+"-rs-"+str(seed) 
        command = " ".join([
            "python", code, 
            "--config", config, 
            "--hammer", "0", 
            "--expname", expname, 
            "--nwalkers", str(nagent), 
            "--maxepisodes", "100000", 
            "--randomseed", str(seed), 
        ]) 
        with open(os.path.join("runs/2021/mw", expname + ".sh"), "w") as f:
            f.write(fixed_text + command)
