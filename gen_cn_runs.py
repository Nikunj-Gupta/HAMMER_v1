import os 
codes = {
    "CN": {
        "script": "../../../hammer-cn.py", 
        "config": "../../../configs/2021/cn/hyperparams.yaml", 
        "dumpdir": "runs/2021/maxcycles_100", 
        "maxepisodes": 100000
    }, 
}

fixed_text = "#!/bin/bash\n"\
             "#SBATCH --mem=5000M\n"\
             "#SBATCH --time=00-20:00\n"\
             "#SBATCH --account=def-mtaylor3\n"\
             "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 


for hammer in [0, 1]: 
    for discretemes in [0, 1]: 
        for prevactions in [0, 1]: 
            for heterogeneity in [0]: 
                for partialobs in [0, 1]: 
        
                    for sharedparams in [0]: 
                        for meslen in range(1, 11): 
                            for seed in [10, 465, 685, 354, 234, 879, 347, 639, 984, 999]: # [984, 999, 674] 
                                if heterogeneity and partialobs: 
                                    continue 
                                expname = "hammer" if hammer else "IL" 
                                if hammer: expname += "-prevactionsyes-" if prevactions else "-prevactionsno-"
                                expname += "-partialobsyes-" if partialobs else "-partialobsno-"
                                expname += "-sharedparamsyes-" if sharedparams else "-sharedparamsno-" 
                                expname += "-heterogeneityyes-" if heterogeneity else "-heterogeneityno-" 
                                if hammer: expname += "-discretemes-" if discretemes else "-continuousmes-" 
                                expname += "-rs" + str(seed) + "-"
                                if hammer: expname += "-meslen" + str(meslen) + "-" 
                                
                                code = "CN" 
                                command = " ".join([
                                    "time python", codes[code]['script'], 
                                    "--config", codes[code]['config'], 
                                    "--maxepisodes", str(codes[code]["maxepisodes"]), 
                                    "--expname", expname, 
                                    "--hammer", str(hammer), 
                                    "--randomseed", str(seed), 
                                    "--maxcycles", str(100), 
                                    "--limit", str(10), 
                                    "--discretemes", str(discretemes), 
                                    "--meslen", str(meslen), 
                                    "--prevactions", str(prevactions), 
                                    "--partialobs", str(partialobs), 
                                    "--sharedparams", str(sharedparams), 
                                    "--heterogeneity", str(heterogeneity) 

                                ]) 
                                with open(os.path.join(codes[code]["dumpdir"], expname + ".sh"), "w") as f:
                                    f.write(fixed_text + command)
