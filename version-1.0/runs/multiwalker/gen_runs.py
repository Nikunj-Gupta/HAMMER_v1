import os, glob
dir = "."
filename = "multiwalker-"
exp = "../../multiwalker.py"
config_dir = "../../configs/random_seed_runs/multiwalker/"
files = glob.glob(config_dir+"*.yaml")

for file in files:
    with open(os.path.join(dir, filename+file.split('/')[-1].split('.')[0]+'.sh'), "w") as f:
        f.write("#!/bin/bash\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --gres=gpu:2\n"
                "#SBATCH --ntasks-per-node=32\n"
                "#SBATCH --mem=127000M\n"
                "#SBATCH --time=2:00:00\n"
                "#SBATCH --account=def-mtaylor3\n"

                "tensorboard --logdir=logs/ --host 0.0.0.0 &\n")
        string = "python " + exp + " " + "--config " + file
        f.write(string)



