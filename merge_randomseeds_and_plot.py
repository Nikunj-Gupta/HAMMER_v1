import os, glob, yaml, pprint, numpy as np 
import matplotlib.pyplot as plt 
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd, numpy as np 

COMPRESSED_HISTOGRAMS = 'compressedHistograms'
HISTOGRAMS = 'histograms'
IMAGES = 'images'
AUDIO = 'audio'
SCALARS = 'scalars'
GRAPH = 'graph'
RUN_METADATA = 'run_metadata'

SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 500,
    IMAGES: 4,
    AUDIO: 4,
    SCALARS: 100000,
    HISTOGRAMS: 1,
}



def get_values(filename, scalar="Avg_reward_for_each_agent__after_an_episode", smooth=0, ignore_start=0, ignore_end=None): 
    ea = event_accumulator.EventAccumulator(filename, size_guidance=SIZE_GUIDANCE)
    ea.Reload()
    ea_scalar = ea.Scalars(tag=scalar) 
    ea_scalar = pd.DataFrame(ea_scalar) 
    if ignore_start: 
        ea_scalar = ea_scalar[ignore_start:]
    if ignore_end: 
        ea_scalar = ea_scalar[:-ignore_end]
    if smooth: 
        ea_scalar["value"] = ea_scalar["value"].rolling(smooth).mean() 
    return ea_scalar


def myregex(log_dir="/home/nikunj/work/HAMMER/runs/2021/meslen_analysis/logs", name="", conditions=[], dump_path=None): 
    res = {name:[]} 
    logs = glob.glob(os.path.join(log_dir, "*")) 
    count = 0 
    for l in logs: 
        count+=1 
        log = [x for x in l.split('-') if x] 
        exp = log[0].split('/')[-1] 
        log = log[1:] 
        check =  all(item in log for item in conditions) 
        check_il = all(item in log for item in conditions[:4]) if exp == "IL" else False 
    
        if check or check_il: 
            log.insert(0, exp) 
            res[name].append({exp+str(count):{"label":"--".join(log),"filename":glob.glob(os.path.join(l, "*"))[0] } }) 
    if dump_path: 
        with open(dump_path, 'w', encoding = "utf-8") as yaml_file:
            yaml_file.write( yaml.dump(res) ) 
    else: 
        return res 

def smooth(array, window=3): 
    cumsum_vec = np.cumsum(np.insert(array, 0, 0)) 
    ma_vec = (cumsum_vec[window:] - cumsum_vec[:-window]) / window 
    return ma_vec 

def random_seeds(name, res, dump=False): 
    random_seeds = {} 
    exp = list(res.keys())[0] 
    for i in res[exp]: 
        label = i[list(i.keys())[0]]['label'] 
        label = label.split('--') 
        if label[0]=='hammer': label[-1], label[-2] = label[-2], label[-1]
        txt = "--".join(label[:-1]) 
        filename = i[list(i.keys())[0]]['filename'] 
        try: 
            random_seeds[txt].append(filename) 
        except: 
            random_seeds[txt] = [filename] 

    concat_array = [] 
    run = 0 
    plots = {name: []} 
    for i in random_seeds: 
        run+=1 
        for j in random_seeds[i]: 
            concat_array.append(list(get_values(filename=j, smooth=0)['value']))
        # res = smooth(np.array(concat_array).mean(axis=0), window=2000) 
        res = np.array(concat_array).mean(axis=0) 
        random_seeds[i] = res 
        # break 
    if dump: 
        np.save(os.path.join('/home/nikunj/work/HAMMER/configs/2021/', name+'.npy'), random_seeds) 
    else: 
        return random_seeds 

def newplot(filename="/home/nikunj/work/HAMMER/configs/2021/res.npy", conditions=[], names=[], smoothing=1, lims=None): 
    res = np.load(filename, allow_pickle=True) 
    res = res.item() 
    ignorestart = lims[0] if lims else 0 
    ignoreend = 100001-lims[1] if lims else 1 
    for i in res: 
        if conditions!=[]: 
            for j in conditions: 
                if j in i: 
                    print(i) 
                    plt.plot(smooth(res[i][ignorestart:-ignoreend], window=smoothing), label=i) 
        else: 
            plt.plot(smooth(res[i][ignorestart:-ignoreend], window=smoothing), label=i) 
            

    if lims: 
        plt.axis((0, lims[1], lims[2], lims[3]))
    plt.legend() 
    plt.show() 


if __name__ == "__main__": 
    # res = myregex(
    #     log_dir="/home/nikunj/work/HAMMER/runs/2021/modified_partialobs/logs", 
    #     name="partialobs_4", 
    #     conditions=['partialobsyes', 'sharedparamsno', 'heterogeneityno'], 
    #     ) 

    # random_seeds("partialobs_4", res, dump=True) 


    # for i in range(1, 11): 
    #     newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/original.npy", 
    #         conditions=["IL", "meslen"+str(i)], 
    #         smoothing=1000
    #     )

    newplot(
        filename="/home/nikunj/work/HAMMER/configs/2021/heterogeneity.npy", 
        conditions=[
            "IL",
            # "prevactionsyes--partialobsno--sharedparamsno--heterogeneityyes--discretemes--meslen10", 
            "prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--discretemes--meslen10"
            ], 
        smoothing=1500, 
    )

    newplot(
        filename="/home/nikunj/work/HAMMER/configs/2021/partialobs.npy", 
        conditions=[
            "IL", 
            "prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--discretemes--meslen4", 
            # "prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--discretemes--meslen8"
            ], 
        smoothing=1500, 
        # lims=[0, 100000, -35, -23.5]
    )
    newplot(
        filename="/home/nikunj/work/HAMMER/configs/2021/original.npy", 
        conditions=[
            "IL", 
                # "prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--discretemes--meslen2", 
                "prevactionsno--partialobsno--sharedparamsno--heterogeneityno--discretemes--meslen4",
                # "prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--discretemes--meslen5",
            ], 
        smoothing=1500, 
        # lims=[0, 100000, -35, -23.5]
    )



    # newplot(
    #     filename="/home/nikunj/work/HAMMER/configs/2021/partialobs_4.npy", 
    #     conditions=[
    #         "IL", 
    #         "prevactionsyes--partialobsyes--sharedparamsno--heterogeneityno--continuousmes--meslen1"
    #         ], 
    #     smoothing=10000, 
    #     ignorestart=0, 
    #     ignoreend=1
    # )


    # newplot(
    #     filename="/home/nikunj/work/HAMMER/configs/2021/heterogeneity.npy", 
    #     conditions=[
    #         "IL",
    #         # "prevactionsyes--partialobsno--sharedparamsno--heterogeneityyes--discretemes--meslen10", 
    #         "prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--discretemes--meslen10"
    #         ], 
    #     smoothing=2000, 
    #     ignorestart=0, 
    #     ignoreend=1
    # )


    # newplot(filename="/home/nikunj/work/HAMMER/configs/2021/partialobs.npy", 
    #         conditions=[
    #             'IL', 
    #             'prevactionsyes--partialobsyes--sharedparamsno--heterogeneityno--continuousmes--meslen1', 
    #             'prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--discretemes--meslen4', 
    #             'prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--discretemes--meslen8' 
    #             ], 
    #         smoothing=5000, 
    #         ignorestart=5000, 
    #         ignoreend=1
    #         ) 



