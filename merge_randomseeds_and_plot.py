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


def myregex(log_dir="/home/nikunj/work/HAMMER/runs/2021/meslen_analysis/logs", name="", conditions=[], rs=None, dump_path=None): 
    res = {name:[]} 
    logs = glob.glob(os.path.join(log_dir, "*")) 
    count = 0 
    for l in logs: 
        count+=1 
        log = [x for x in l.split('-') if x] 
        exp = log[0].split('/')[-1] 
        log = log[1:] 
        check =  all(item in log for item in conditions) 
        position = -1 if exp=="IL" else -2 
        rs_check = any(str(seed) in log[position] for seed in rs) if rs else True 
        check_il = all(item in log for item in conditions[:3]) if exp == "IL" else False 
    
        if (check or check_il) and rs_check: 
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
        if 'hammer' in label[0]: label[-1], label[-2] = label[-2], label[-1] 
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
        np.save(os.path.join('/home/nikunj/work/HAMMER/configs/2021/newnpy/', name+'.npy'), random_seeds) 
    else: 
        return random_seeds 

def plot(res, smoothing=1, ignorestart=0, ignoreend=1): 
    for i in res: 
        x = i[list(i.keys())[0]] 
        plt.plot(smooth(np.array(get_values(x['filename'])['value']), window=smoothing), label=x['label']) 
            

    # if lims: 
    #     plt.axis((0, lims[1], lims[2], lims[3]))
    plt.legend() 
    plt.show() 


def get_res(filename="/home/nikunj/work/HAMMER/configs/2021/res.npy", conditions=[]): 
    res = np.load(filename, allow_pickle=True) 
    res = res.item() 
    newres = {}  
    for i in res: 
        for j in conditions: 
            if j in i: 
                if "meslen10" in i: continue 
                newres[i] = res[i] 
    return newres 


def plot_onemore(res, smoothing=1, lims=None): 
    ignorestart = lims[0] if lims else 0 
    ignoreend = 100001-lims[1] if lims else 1 
    for i in res: 
        print(i) 
        plt.plot(smooth(res[i][ignorestart:-ignoreend], window=smoothing), label=i) 
    if lims: 
        plt.axis((0, lims[1], lims[2], lims[3])) 
    plt.legend()
    plt.show()

def newplot(filename="/home/nikunj/work/HAMMER/configs/2021/res.npy", conditions=[], names=[], smoothing=1, lims=None): 
    
    res = np.load(filename, allow_pickle=True) 
    res = res.item() 
    ignorestart = lims[0] if lims else 0 
    ignoreend = 100001-lims[1] if lims else 1 
    for i in res: 
        if conditions!=[]: 
            for j in conditions: 
                if j in i: 
                    if "meslen10" in i: continue 
                    print(i) 
                    plt.plot(smooth(res[i][ignorestart:-ignoreend], window=smoothing), label=i) 
        else: 
            plt.plot(smooth(res[i][ignorestart:-ignoreend], window=smoothing),label=i) 
            
    if lims: 
        plt.axis((0, lims[1], lims[2], lims[3])) 
    plt.legend()
    plt.show()
    
def publication_plot( 
        res, 
        # filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/original.npy", 
        # conditions=[], 
        title="", 
        labels=[], 
        smoothing=1, 
        lims=None, 
        save=False
    ): 

    params = {
        'axes.labelsize': 18, 
        'axes.titlesize':'xx-large', 
        'legend.fontsize': 14,
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'text.usetex': False,
        'figure.figsize': [7.5, 6]
    }

    from pylab import plot, rcParams, legend, axes, grid

    rcParams.update(params)

    # res = np.load(filename, allow_pickle=True) 
    # res = res.item() 
    ignorestart = lims[0] if lims else 0 
    ignoreend = 100001-lims[1] if lims else 1 
    colors = ['#006BB2', '#B22400', '#FFC325', '#A9A9A9', '#A8A8A8'] 
    counter = 0 
    for i in res: 
        # if conditions!=[]: 
        #     for j in conditions: 
        #         if j in i: 
        #             if "meslen10" in i: continue 
                    print(i) 
                    plt.plot(smooth(res[i][ignorestart:-ignoreend], window=smoothing), color=colors[counter], label=labels[counter]) 
                    counter+=1 
        # else: 
        #     plt.plot(smooth(res[i][ignorestart:-ignoreend], window=smoothing), label=i) 
            

    if lims: 
        plt.axis((0, lims[1], lims[2], lims[3])) 
    
    legend = legend(loc="lower right") 
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9') 
    # plt.yticks(np.arange(lims[2], lims[3], step=2)) 

    plt.title(title) 
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Returns per Agent")
    plt.legend(loc="lower right")
    grid()
    if save:
        plt.savefig(os.path.join("/home/nikunj/Desktop/plots", title+'.png')) 
        plt.close()
    plt.show()

def maddpg(name="maddpg", rs=[10, 984, """879, 234"""], dump=False): 
    log_dir = "/home/nikunj/work/HAMMER/runs/2021/maddpg/models/simple_spread" 
    logs = glob.glob(os.path.join(log_dir, "*")) 
    

    res = {} 
    for l in logs: 
        key = l.split('/')[-1] 
        value = glob.glob(os.path.join(os.path.join(l, "run1/logs/event*")))[0]
        rs_check = any(str(seed) in l for seed in rs) if rs!=[] else True 
        if rs_check: 
            res[key] = value
    
    concat_array = [] 
    newres = {}
    for i in res: 
        print(i)
        x = list(get_values(filename=res[i], smooth=0, scalar="agent0/mean_episode_rewards")['value']) 
        plt.plot(smooth(x, window=1000), label=i) 
        concat_array.append(x) 
    newres["maddpg"] = np.array(concat_array).mean(axis=0) 
    if dump: 
        np.save(os.path.join('/home/nikunj/work/HAMMER/configs/2021/newnpy/', name+'.npy'), newres) 
    else: 
        plt.legend() 
        plt.show() 
        # plt.savefig(os.path.join("/home/nikunj/Desktop/plots", save))
        # plt.close()
        print(newres)
        return newres  

if __name__ == "__main__": 
    """ 
    Original
    """ 

    # seeds = [465, 685, 354, 234, 879, 347, 639, 984, 999, 674]
    # # for s in seeds: 
    # res = myregex(
    #     log_dir="/home/nikunj/work/HAMMER/randomseeds/logs", 
    #     name="original_bestseeds", 
    #     conditions=['partialobsno', 'sharedparamsno', 'heterogeneityno'], 
    #     rs=[465] 
    #     ) 

    # # newres = {"original_bestseeds": []} 
    # # for i in res['original_bestseeds']: 
    # #     if ("prevactionsno--partialobsno--sharedparamsno--heterogeneityno--discretemes" in i[list(i.keys())[0]]['label'] and \
    # #                         "--meslen4" in i[list(i.keys())[0]]['label']) or "IL" in i[list(i.keys())[0]]['label']: 
    # #         newres['original_bestseeds'].append(i) 
    # # plot(newres['original_bestseeds'], smoothing=5000)

    # random_seeds("original_bestseeds", res, dump=True) 
    # print("Original Done!") 

    # for i in range(1, 11): 
    #     newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/original_10seeds.npy", 
    #         conditions=["IL", "meslen"+str(i)], 
    #         smoothing=1000
    #     ) 
    # newplot(
    #     filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/original_bestseeds.npy", 
    #     conditions=["IL", "prevactionsno--partialobsno--sharedparamsno--heterogeneityno--discretemes--meslen6"], 
    #     smoothing=750, 
    #     # lims=[0,50000,-35,-25]
    # ) 
    
    """ 
    Partial Obs 
    """ 

    # res = myregex(
    #     log_dir="/home/nikunj/work/HAMMER/randomseeds/logs", 
    #     name="partialobs", 
    #     conditions=['partialobsyes', 'sharedparamsno', 'heterogeneityno'], 
    #     rs=[347, 639, 685, 354] 
    #     ) 
    
    # # newres = {"partialobs": []}
    # # for i in res['partialobs']: 
    # #     if "IL" in i[list(i.keys())[0]]['label']: 
    # #         newres['partialobs'].append(i) 
    # # plot(newres['partialobs'], smoothing=1000)

    # random_seeds("partialobs", res, dump=True) 
    # print("PartialObs Done!") 
    
    """ 
    Heterogeneity 
    """ 

    # res = myregex(
    #     log_dir="/home/nikunj/work/HAMMER/randomseeds/logs", 
    #     name="heterogeneity", 
    #     conditions=['partialobsno', 'sharedparamsno', 'heterogeneityyes'], 
    #     rs=[465, 984, 639, 685] 
    #     ) 

    # # # newres = {"heterogeneity": []}
    # # # for i in res['heterogeneity']: 
    # # #     if "IL" in i[list(i.keys())[0]]['label']: 
    # # #         newres['heterogeneity'].append(i) 
    # # # plot(newres['heterogeneity'], smoothing=1000)


    # random_seeds("heterogeneity", res, dump=True) 
    # print("Heterogeneity Done!") 

    # for i in range(1, 11): 
    #     newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/heterogeneity.npy", 
    #         conditions=["IL", "meslen"+str(i)], 
    #         smoothing=1000
    #     ) 

    # newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/heterogeneity.npy", 
    #         conditions=[
    #             "IL", 
    #             # "prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--discretemes--meslen2", 
    #             # "prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--continuousmes--meslen4", 
    #             "prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--discretemes--meslen4", 
    #             "prevactionsyes--partialobsno--sharedparamsno--heterogeneityyes--continuousmes--meslen9" 
    #             ], 
    #         smoothing=3000, 
    #         lims=[0, 70000, -35.5, -24] 
    #     )


    # newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/partialobs.npy", 
    #         conditions=[
    #             "IL", 
    #             # "prevactionsyes--partialobsyes--sharedparamsno--heterogeneityno--discretemes--meslen4", 
    #             "prevactionsyes--partialobsyes--sharedparamsno--heterogeneityno--continuousmes--meslen6", 
    #             "prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--discretemes--meslen8", 
    #             ], 
    #         smoothing=3000, 
    #         lims=[0, 50000, -35, -24]
    #     )


    # newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/npy/test2.npy", 
    #         conditions=[
    #             "IL", 
    #             "prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen1", 
    #             # "prevactionsno--partialobsno--sharedparamsno--heterogeneityno--discretemes--meslen4", 
    #             # "prevactionsno--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen6", 

    #             ], 
    #         smoothing=1500, 
    #         lims=[0, 50000, -36, -25]
    #     )

    """ 
    MaxCycles 100 
    """ 

    """ 
    Original
    """ 

    # meslen = "meslen10"
    # res = myregex(
    #     log_dir="/home/nikunj/work/HAMMER/runs/2021/maxcycles_100/logs", 
    #     name="original-maxcycles100", 
    #     conditions=['partialobsno', 'sharedparamsno', 'heterogeneityno', meslen], 
    #     # rs=[465, 234, 685, 354] 
    #     ) 

    # # newres = {"original-maxcycles100": []}
    # # for i in res['original-maxcycles100']: 
    # #     if "IL" in i[list(i.keys())[0]]['label']: 
    # #         newres['original-maxcycles100'].append(i) 
    # # plot(newres['original-maxcycles100'], smoothing=1000)

    # random_seeds("original-maxcycles100", res, dump=True)  
    # print("Original Done!") 

    # # for i in range(1, 11): 
    # newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/original-maxcycles100.npy", 
    #         conditions=["IL", meslen], 
    #         smoothing=1000
    #     ) 


    """ 
    Partial Obs 
    """ 
    # meslen = "meslen8"
    # res = myregex(
    #     log_dir="/home/nikunj/work/HAMMER/runs/2021/maxcycles_100/logs", 
    #     name="partialobs-maxcycles100", 
    #     conditions=['partialobsyes', 'sharedparamsno', 'heterogeneityno', meslen], 
    #     # rs=[465, 234, 685, 354] 
    #     ) 

    # # newres = {"original": []}
    # # for i in res['original']: 
    # #     if "IL" in i[list(i.keys())[0]]['label']: 
    # #         newres['original'].append(i) 
    # # plot(newres['original'], smoothing=1000)

    # random_seeds("partialobs-maxcycles100", res, dump=True) 
    # print("partialobs Done!") 

    # # for i in range(1, 11): 
    # newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/partialobs-maxcycles100.npy", 
    #         # conditions=[meslen], 
    #         smoothing=1000, 
    #         # lims=[0, 15000, -140, -60] 
    #     ) 
    

    """ 
    Random Message 
    """ 
    # meslen = "meslen2"
    # res = myregex(
    #     log_dir="/home/nikunj/work/HAMMER/runs/2021/randommes/logs", 
    #     name="randommes", 
    #     conditions=['partialobsno', 'sharedparamsno', 'heterogeneityno'], 
    #     # rs=[10] 
    #     ) 
    # pprint.pprint(random_seeds("randommes", res)) 

    # pprint.pprint(res)
    # # newres = {"randommes": []}
    # # for i in res['randommes']: 
    # #     # if "IL" in i[list(i.keys())[0]]['label']: 
    # #     print(i)
    # #     newres['randommes'].append(i) 
    # # plot(newres['randommes'], smoothing=1000)

    # random_seeds("randommes", res, dump=True) 
    # print("randommes Done!") 

    # for i in range(1, 11): 
    #     newplot(
    #             filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/randommes.npy", 
    #             conditions=["meslen"+str(i)], 
    #             smoothing=1000, 
    #             # lims=[0, 15000, -140, -60] 
    #         ) 
    
    # newplot(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/randommes.npy", 
    #         conditions=["prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen4"], 
    #         smoothing=1000, 
    #         # lims=[0, 15000, -140, -60] 
    #     ) 


    """ 
    Publication Plots 
    """ 
    # final = get_res(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/original_bestseeds.npy", 
    #         conditions=[
    #             "IL", 
    #             # "prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen1", 
    #             "prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen4", 
    #             "prevactionsno--partialobsno--sharedparamsno--heterogeneityno--discretemes--meslen4", 
    #             ]
    #         ) 

    # randommes = get_res(filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/randommes.npy", 
    #         conditions=["prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen4"], 
    #         ) 



    # final.update(randommes) 

    # publication_plot(
    #         res=final, 
    #         title="HAMMER on Cooperative Navigation (3 agents)", 
    #         labels=["Independent Learners", "HAMMER-Discrete-4", "HAMMER-Continuous-4", "HAMMER-Random-4"], 
    #         smoothing=1000, 
    #         lims=[0, 50000, -36, -25] 
    #     ) 
    
    # partialobs = get_res(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/partialobs.npy", 
    #         conditions=[
    #             "IL", 
    #             # "prevactionsyes--partialobsyes--sharedparamsno--heterogeneityno--discretemes--meslen4", 
    #             "prevactionsyes--partialobsyes--sharedparamsno--heterogeneityno--continuousmes--meslen6", 
    #             "prevactionsno--partialobsyes--sharedparamsno--heterogeneityno--discretemes--meslen8", 
    #             ] 
    #     ) 

    # publication_plot(
    #         res=partialobs, 
    #         title="HAMMER on Modified Cooperative Navigation (3 agents)", 
    #         labels=["HAMMER-Discrete-8", "HAMMER-Continuous-6", "Independent Learners"], 
    #         smoothing=2000, 
    #         lims=[0, 50000, -35.5, -25] 
    #     ) 
    
    # heterogeneity = get_res( 
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/heterogeneity.npy", 
    #         conditions=[
    #             "IL", 
    #             # "prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--discretemes--meslen2", 
    #             # "prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--continuousmes--meslen4", 
    #             "prevactionsno--partialobsno--sharedparamsno--heterogeneityyes--discretemes--meslen4", 
    #             "prevactionsyes--partialobsno--sharedparamsno--heterogeneityyes--continuousmes--meslen9" 
    #             ], 

    #     ) 

    # publication_plot(
    #         res=heterogeneity, 
    #         title="HAMMER on a Heterogeneous Setting \nin Cooperative Navigation (3 agents)", 
    #         labels=["Independent Learners", "HAMMER-Discrete-4", "HAMMER-Continuous-9"], 
    #         smoothing=1500, 
    #         lims=[0, 70000, -35, -24] 
    #     ) 


   

    # # publication_plot(
    # #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/original.npy", 
    # #         conditions=[
    # #             "IL", 
    # #             "prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen1", 
    # #             "prevactionsno--partialobsno--sharedparamsno--heterogeneityno--discretemes--meslen4", 
    # #             # "prevactionsno--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen6", 

    # #             ], 
            
    # #         title="HAMMER on Cooperative Navigation (3 agents)", 
    # #         labels=["HAMMER-Discrete-4", "Independent Learners", "HAMMER-Continuous-1"], 
    # #         smoothing=2000, 
    # #         lims=[0, 50000, -36, -24] 
    # #     ) 


    # """ 
    # HAMMER vs MADDPG 
    # """ 
    # # maddpg(dump=True) 

    # final = get_res(
    #         filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/original.npy", 
    #         conditions=[
    #             "IL",
    #             # "prevactionsyes--partialobsno--sharedparamsno--heterogeneityno--continuousmes--meslen4", 
    #             "prevactionsno--partialobsno--sharedparamsno--heterogeneityno--discretemes--meslen4", 
    #             ]
    #         ) 

    # filename="/home/nikunj/work/HAMMER/configs/2021/newnpy/maddpg.npy" 
    # res = np.load(filename, allow_pickle=True) 
    # res = res.item()  

    # final.update(res) 

    # publication_plot(
    #         res=final, 
    #         title="HAMMER vs MADDPG \non Cooperative Navigation (3 agents)", 
    #         labels=["HAMMER", "IL", "MADDPG"], 
    #         smoothing=500, 
    #         lims=[0, 50000, -35, -24] 
    #     ) 

    """ 
    SharedParams 
    """ 
    """ 
    PartialObs 
    """ 

    seeds = [465, 685, 354, 234, 879, 347, 639, 984, 999, 674]
    for s in seeds: 
        res = myregex(
            log_dir="/home/nikunj/work/HAMMER/randomseeds/logs", 
            name="original_bestseeds", 
            conditions=['partialobsno', 'sharedparamsno', 'heterogeneityno'], 
            rs=[s] 
            ) 

    newres = {"original_bestseeds": []} 
    for i in res['original_bestseeds']: 
        if ("prevactionsno--partialobsno--sharedparamsno--heterogeneityno--discretemes" in i[list(i.keys())[0]]['label'] and \
                            "--meslen4" in i[list(i.keys())[0]]['label']) or "IL" in i[list(i.keys())[0]]['label']: 
            newres['original_bestseeds'].append(i) 
    plot(newres['original_bestseeds'], smoothing=5000)
