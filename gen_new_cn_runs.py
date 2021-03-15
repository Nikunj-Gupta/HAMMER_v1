import os 
dumpdir = "runs/march2021/cn/" 

codes = { 
    "Hammerwithgradients": {
        "script": "../../../hammer-with-gradient.py", 
        "config": "../../../configs/2021/cn/hyperparams.yaml", 
        "dumpdir": dumpdir, 
        "args": {
        }
    }, 
}
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --mem=5000M\n"\
             "#SBATCH --time=00-16:00\n"\
             "#SBATCH --account=def-mtaylor3\n"\
             "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 

for code in codes: 
    for seed in [999]:#, 674, 354, 234, 10]: # [984, 999, 674, 354, 234, 879, 347, 639, 465, 999, 10]  
        for dru in [0, 1]: 
            for meslen in [0, 1, 2, 3, 4, 5]: 
                partialobs = 0 
                heterogeneity = 0 
                sharedparams = 1 
                envname = "cn" 
                if meslen: 
                    expname = '--'.join([
                        code, 
                        "meslen"+str(meslen), 
                        ]) 
                else: 
                    expname = "IL" 

                if meslen != 0: expname = '--'.join([expname, "dru"+str(dru)]) 
                expname = '--'.join([expname, "rs"+str(seed)]) 

                if dru: 
                    meslen = meslen*5 

                command = " ".join([
                    "time python", codes[code]['script'], 
                    "--config", codes[code]['config'], 
                    "--expname", expname, 
                    "--dru_toggle", str(dru), 
                    "--meslen", str(meslen), 
                    "--randomseed", str(seed), 
                ]) 
                

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