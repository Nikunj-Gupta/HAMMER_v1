import os 
dumpdir = "runs/jan_2023/cn/" 

fixed_text = "#!/bin/bash\n"\
             "#SBATCH --mem=5000M\n"\
             "#SBATCH --time=00-16:00\n"\
             "#SBATCH --account=def-mtaylor3\n"\
             "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 

for experiment in ["randommes", "coil"]: 
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
