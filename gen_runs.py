import os 


codes = {
    "CN": {
        "script": "../../../main_complete_state_and_prev_actions.py", 
        "config": "../../../configs/2021/cn/hyperparams.yaml", 
        "dumpdir": "runs/2021/test", 
        "maxepisodes": 100000
    }, 

    "MW": {
        "script": "../../../multiwalker.py", 
        "config": "../../../configs/2021/mw/hyperparameters.yaml", 
        "dumpdir": "runs/2021/test", 
        "maxepisodes": 100000 
    }
}



fixed_text = "#!/bin/bash\n"\
                    "#SBATCH --nodes=1\n"\
                    "#SBATCH --gres=gpu:2\n"\
                    "#SBATCH --ntasks-per-node=32\n"\
                    "#SBATCH --mem=127000M\n"\
                    "#SBATCH --time=00-30:00\n"\
                    "#SBATCH --account=def-mtaylor3\n"\
                    "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 



for code in codes:  
    for hammer in [0, 1]: 
        for nagents in [3, 5]: 
            for seed in range(9, 11): 
                for meslen in range(4, 6): 
                    expname = "hammer" if hammer else "IL" 
                    expname = code + "-" + expname + "-nagents-" + str(nagents) + "-rs-" + str(seed)+ "-meslen-" + str(meslen) 
                    command = " ".join([

                        "python", codes[code]['script'], 
                        "--config", codes[code]['config'], 
                        "--hammer", str(hammer), 
                        "--expname", expname, 
                        "--nagents", str(nagents), 
                        "--maxepisodes", str(codes[code]["maxepisodes"]), 
                        "--randomseed", str(seed), 
                        "--meslen", str(meslen)
                    ]) 
                    with open(os.path.join(codes[code]["dumpdir"], expname + ".sh"), "w") as f:
                        f.write(fixed_text + command)
