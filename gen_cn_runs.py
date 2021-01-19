import os 
codes = {
    "CN": {
        "script": "../../../hammer-cn.py", 
        "config": "../../../configs/2021/cn/hyperparams.yaml", 
        "dumpdir": "runs/2021/temp2", 
        "maxepisodes": 50000
    }, 
}

fixed_text = "#!/bin/bash\n"\
             "#SBATCH --mem=5000M\n"\
             "#SBATCH --time=00-10:00\n"\
             "#SBATCH --account=def-mtaylor3\n"\
             "tensorboard --logdir=logs/ --host 0.0.0.0 &\n" 

for hammer in [0, 1]: 
    for prevactions in [0]: 
        for partialobs in [0, 1]: 
            for sharedparams in [0, 1]: 
                for meslen in [1, 4]: 
                    for seed in [465, 685]: 
                        expname = "hammer" if hammer else "IL" 
                        if hammer: expname += "-prevactionsyes-" if prevactions else "-prevactionsno-"
                        expname += "-partialobsyes-" if partialobs else "-partialobsno-"
                        expname += "-sharedparamsyes-" if sharedparams else "-sharedparamsno-" 
                        if hammer: expname += "-meslen" + str(meslen) 
                        code = "CN" 
                        command = " ".join([
                            "python", codes[code]['script'], 
                            "--config", codes[code]['config'], 
                            "--maxepisodes", str(codes[code]["maxepisodes"]), 
                            "--expname", expname, 
                            "--hammer", str(hammer), 
                            "--randomseed", str(seed), 
                            "--meslen", str(meslen), 
                            "--prevactions", str(prevactions), 
                            "--partialobs", str(partialobs), 
                            "--sharedparams", str(sharedparams) 
                        ]) 
                        with open(os.path.join(codes[code]["dumpdir"], expname + ".sh"), "w") as f:
                            f.write(fixed_text + command)
