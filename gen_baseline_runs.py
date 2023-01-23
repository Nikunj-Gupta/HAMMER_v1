
import os, yaml 
from pathlib import Path 
from itertools import count 
dumpdir = "runs/jan_2023/cn_hammer_v1/" 
if not os.path.isdir(dumpdir):
    os.mkdir(dumpdir)
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --nodes=1\n"\
             "#SBATCH --cpus-per-task=16 \n"\
             "#SBATCH --time=24:00:00\n"\
             "#SBATCH --mem=40GB\n"\
             "\n"\
             "source ../venvs/hammer/bin/activate\n"\
             "\n"\
             "module load python/intel/3.8.6\n"\
             "module load openmpi/intel/4.0.5\n"\
             "\n" 

# for experiment in ["randommes", "coil"]: 
for experiment in ["hammer-v1"]: 
    for nagents in [3, 5, 7, 10]: 
        for seed in [984, 999, 674, 354, 234, 879, 347, 639, 465, 999, 10]: 
            expname = "hammer--cn" 
            expname = '--'.join([expname, experiment]) 
            expname = '--'.join([expname, "n"+str(nagents)]) 
            expname = '--'.join([expname, "rs"+str(seed)]) 

            
            if experiment=="randommes": 
                script = "hammer-cn.py" 
                command = " ".join([
                    "time python", script, 
                    "--config configs/2021/cn/hyperparams.yaml", 
                    "--nagents ", str(nagents), 
                    "--expname", expname, 
                    "--discretemes 0", 
                    "--meslen 1 ", 
                    "--randomseed", str(seed), 
                    "--randommes 1 ", 
                    "--hammer 1 ", 
                ])
            if experiment=="hammer-v1": 
                script = "hammer-cn.py" 
                command = " ".join([
                    "time python", script, 
                    "--config configs/2021/cn/hyperparams.yaml", 
                    "--nagents ", str(nagents), 
                    "--expname", expname, 
                    "--discretemes 0", 
                    "--meslen 1 ", 
                    "--randomseed", str(seed), 
                    "--randommes 0 ", 
                    "--hammer 1 ", 
                ])
            if experiment=="coil": 
                script = "centralized_training.py" 
                command = " ".join([
                    "time python", script, 
                    "--config configs/2021/cn/hyperparams.yaml", 
                    "--nagents ", str(nagents), 
                    "--expname", expname, 
                    "--randomseed", str(seed), 
                ])
            # print(expname) 
            # print(command) 
            # print() 
            with open(os.path.join(dumpdir, expname + ".sh"), "w") as f:
                f.write(fixed_text + command) 
