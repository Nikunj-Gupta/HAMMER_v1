import os 
dumpdir = "runs/2021/" 

codes = { 
    # "Hammer_previous": { 
    #     "script": "../../../hammer-cn.py", 
    #     "config": "../../../configs/2021/cn/hyperparams.yaml", 
    #     "dumpdir": dumpdir, 
    #     "args": {

    #     }
    # }, 
    # "centralized_training": { 
    #     "script": "../../../centralized_training.py", 
    #     "config": "../../../configs/2021/cn/hyperparams.yaml", 
    #     "dumpdir": dumpdir, 
    #     "args": {

    #     }
    # }, 
    "Hammerwithgradients_mw": { 
        "script": "../../../hammer-with-gradient-mw.py", 
        "config": "../../../configs/2021/cn/hyperparams.yaml", 
        "dumpdir": dumpdir, 
        "args": {

        }
    }, 
    "il_multiwalker": { 
        "script": "../../../il_multiwalker.py", 
        "config": "../../../configs/2021/cn/hyperparams.yaml", 
        "dumpdir": dumpdir, 
        "args": {
        }
    }, 

}
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --mem=5000M\n"\
             "#SBATCH --time=00-48:00\n"\
             "#SBATCH --account=def-mtaylor3\n"\
             "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 

for code in codes: 
    for seed in [999, 674, 354, 234, 10]: # [984, 999, 674, 354, 234, 879, 347, 639, 465, 999, 10]  
        for dru in [0, 1]: 
            # for discretemes in [0, 1]: 
                discretemes = 0 
                partialobs=0 
                heterogeneity = 0 
                randommes = 0 

                meslen = 1 
                envname="mw"
                limit = 10 if envname == "cn" else 11 
                nagents = 3 if envname=="cn" else 2 

                expname = '--'.join([code, envname]) 

                if "Hammer" in code: expname = '--'.join([expname, "meslen"+str(meslen)]) 
                if "gradients" in code: 
                    if dru: 
                       expname = '--'.join([expname, "dru"]) 
                    else: 
                        expname = '--'.join([expname, "nodru"]) 
                elif "Hammer_previous" in code: expname = '--'.join([expname, "discretemes"+str(discretemes)]) 
                
                expname = '--'.join([expname, "rs"+str(seed)]) 
                command = " ".join([
                    "time python", codes[code]['script'], 
                    "--config", codes[code]['config'], 
                    "--expname", expname, 
                    "--maxepisodes", str(50_000), 
                    # "--partialobs", str(partialobs), 
                    # "--heterogeneity", str(heterogeneity), 
                    # "--randommes", str(randommes), 
                    "--randomseed", str(seed), 
                    # "--nagents", str(nagents)
                ]) 

                if "Hammer" in code: command = " ".join([command,  "--meslen", str(meslen)]) 
                # if "gradients" in code: command = " ".join([command,  "--envname", str(envname)]) 
                if "gradients" in code: command = " ".join([command,  "--dru_toggle", str(dru)]) 
                if "ILsharedparams" in code: command = " ".join([command,  "--envname", str(envname)]) 
                if "Hammer_previous" in code: command = " ".join([command,  "--discretemes", str(discretemes)]) 
                if partialobs or heterogeneity: command = " ".join([command,  "--limit", str(limit)]) 
                
                # print(expname) 
                # print(command) 
                # print() 
                with open(os.path.join(codes[code]["dumpdir"], expname + ".sh"), "w") as f:
                    f.write(fixed_text + command) 

""" 
MADDPG Runs 
""" 

# for seed in [10, 465, 685, 354, 234, 879, 347, 639, 984, 999]: # [984, 999, 674] 
#     expname = "hammer-maddpg-rs"+str(seed) 
#     command = " ".join([
#         "time python ../../../hammer-maddpg.py ", 
#         "--model_name", expname, 
#         "--seed", str(seed), 
#     ]) 
#     with open(os.path.join("runs/2021/maddpg", expname + ".sh"), "w") as f: 
#         f.write(fixed_text + command) 