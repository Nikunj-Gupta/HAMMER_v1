import os, glob


fixed_text = "#!/bin/bash\n"\
                    "#SBATCH --nodes=1\n"\
                    "#SBATCH --gres=gpu:2\n"\
                    "#SBATCH --ntasks-per-node=32\n"\
                    "#SBATCH --mem=127000M\n"\
                    "#SBATCH --time=2:00:00\n"\
                    "#SBATCH --account=def-mtaylor3\n"\
                    "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 


# Hammer on Cooperative Navigation 
code = "../../../main_complete_state_and_prev_actions.py"
config = "../../../configs/2021/cn/hyperparams.yaml" 

for seed in range(9, 11): 
    for meslen in range(4, 6): 
        expname = "cn-hammer-rs-"+str(seed)+"-meslen-"+str(meslen) 
        command = " ".join([
            "python", code, 

            "--config", config, 
            "--hammer", "1", 
            "--expname", expname, 
            "--maxepisodes", "100000", 
            "--randomseed", str(seed), 
            "--meslen", str(meslen)
        ]) 
        with open(os.path.join("runs/2021/cn", expname + ".sh"), "w") as f:
            f.write(fixed_text + command)

# IL on Cooperative Navigation 
for seed in range(9, 11): 
    expname = "cn-il-rs-"+str(seed) 
    command = " ".join([
        "python", code, 

        "--config", config, 
        "--hammer", "0", 
        "--expname", expname, 
        "--maxepisodes", "100000", 
        "--randomseed", str(seed)
    ]) 

    with open(os.path.join("runs/2021/cn", expname + ".sh"), "w") as f:
        f.write(fixed_text + command)
