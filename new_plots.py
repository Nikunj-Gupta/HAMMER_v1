import os, glob, yaml, pprint, numpy as np 
import matplotlib.pyplot as plt 
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd, numpy as np 

SIZE_GUIDANCE = {
    'compressedHistograms': 100_000, 
    'images': 4, 
    'audio': 4, 
    'scalars': 500_000, 
    'histograms': 1, 
}


def get_values(filename, scalar="Avg_reward_for_each_agent__after_an_episode"): 
    ea = event_accumulator.EventAccumulator(filename, size_guidance=SIZE_GUIDANCE)
    ea.Reload()
    ea_scalar = ea.Scalars(tag=scalar) 
    ea_scalar = pd.DataFrame(ea_scalar) 
    return ea_scalar



def smooth(array, window=3): 
    cumsum_vec = np.cumsum(np.insert(array, 0, 0)) 
    ma_vec = (cumsum_vec[window:] - cumsum_vec[:-window]) / window 
    return ma_vec 




def merge(log_dir, conditions=[], smoothing_window=None, save_name=None): 
    logs = glob.glob(os.path.join(log_dir, "*/*"), recursive=True) 
    seeds = {} 
    exps = {} 
    for log in logs: 
        exp = log.split("/")[-2] 
        if all(c in exp for c in conditions):
            seed = exp.split("--")[-1] 
            seeds[seed] = [] 

            exp = "--".join(exp.split("--")[:-1]) 
            exps[exp] = [] 

    for exp in exps: 
        for log in logs: 
            if exp in log.split("/")[-2]: 
                exps[exp].append(log)
    # pprint.pprint(seeds) 
    # pprint.pprint(exps) 

    res = {} 

    for exp in exps: 
        array = [] 
        for filename in exps[exp]: 
            array.append(list(get_values(filename=filename)['value'])) 
        if smoothing_window:
            array = [smooth(a, window=smoothing_window) for a in array] 
        mean_array = np.array(array).mean(axis=0) 
        std_array = np.array(array).std(axis=0) 

        res[exp] = {"mean": mean_array, "std": std_array} 
        # break 
    
    if save_name: 
        np.save(os.path.join(save_name), res) 
        print(save_name+" saved!") 
    else: 
        print(res) 

def publication_plot( 
        res, 
        start=None, 
        end=None, 
        smoothing_window=None, 
        title="", 
        labels={}, 
        colors=[], 
        save=False 
    ): 

    params = {
        'axes.labelsize': 28, 
        'axes.titlesize': 32, 
        'legend.fontsize': 14,
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'text.usetex': True, 
        'figure.figsize': [10, 8]
    }

    from pylab import plot, rcParams, legend, axes, grid

    rcParams.update(params)

    values = labels if labels else res 
    for i in values: 
        print(i) 
        mean_array = res[i]["mean"] 
        std_array = res[i]["std"] 
        if not start: 
            start=0 
        if not end: 
            end = len(mean_array) 
        mean_array = mean_array[start:end]
        std_array = std_array[start:end]
        if smoothing_window: 
            mean_array = smooth(array=mean_array, window=smoothing_window) 
            std_array = smooth(array=std_array, window=smoothing_window) 
        if "discretemes1" not in i: 
            if labels!={}: 
                plt.plot(mean_array, color=labels[i]["color"], label=labels[i]["label"]) 
            else: 
                plt.plot(mean_array, label=i) 
        

        plt.fill_between(np.arange(1, len(mean_array)+1), 
                    mean_array - std_array, 
                    mean_array + std_array, color=labels[i]["color"], 
                    alpha=0.2)
            

    legend = legend() 
    # legend = legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True) 
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9') 

    plt.title(title) 
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Returns per Agent") 
    plt.xticks([i for i in range(0, 600_000, 100_000) ]) 

    if end: 
            plt.xlim((start, end))

    # plt.ylim((-36, -21)) # CN 
    # plt.ylim((-35, -28)) # Modified CN 
    plt.ylim((-34, -25.5)) # Heterogeneous CN 

    plt.legend(loc="upper left")
    grid()
    if save:
        plt.savefig(os.path.join("hammer_plots", title+'.png')) 
        plt.close()
    plt.show()

if __name__ == "__main__": 
    log_dir = "/home/nikunj/work/HAMMER/runs/2021/message_gradients_rs/logs/" 


    # merge(log_dir=log_dir, conditions=["partialobs0"], smoothing_window=500, save_name="rs_npy/cn") 
    # merge(log_dir=log_dir, conditions=["partialobs1"], smoothing_window=500, save_name="rs_npy/modified_cn") 
    # merge(log_dir=log_dir, conditions=["heterogeneity1"], smoothing_window=500, save_name="rs_npy/heterogeneity") 
    # merge(log_dir=log_dir, conditions=["centralized_training"], smoothing_window=500, save_name="rs_npy/centralized_training") 
    # merge(log_dir=log_dir, conditions=["randommes1"], smoothing_window=500, save_name="rs_npy/randommes") 

    """ 
    CN --- HAMMER + IL + other Baselines (Random Message, Centralized Training) 
    """ 
    res = np.load("rs_npy/cn.npy", allow_pickle=True).item() 
    res.update(np.load("rs_npy/centralized_training.npy", allow_pickle=True).item()) 
    res.update(np.load("rs_npy/randommes.npy", allow_pickle=True).item()) 

    # "#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
    labels = {
        "Hammerwithgradients--cn--partialobs0--meslen1--dru": {"label": "HAMMERv3", "color": "#377eb8"}, 
        "Hammerwithgradients--cn--partialobs0--meslen1--nodru": {"label": "HAMMERv2", "color": "#4daf4a"}, 
        "Hammer_previous--cn--partialobs0--meslen1--discretemes0": {"label": "HAMMERv1", "color": "#984ea3"}, 
        # "Hammer_previous--cn--partialobs0--meslen1--discretemes1": {"label": "", "color": ""}, 
        "ILsharedparams--cn--partialobs0": {"label": "IL", "color": "#e41a1c"}, 
        "Hammer_previous--cn--randommes1--meslen1--discretemes0": {"label": "IL + Random messages", "color": "#ff7f00"}, 
        "centralized_training--cn": {"label": "Centralized Training for IL", "color": "#ffff33"}, 
    }
    
    
    
    # publication_plot(
    #     res=res, 
    #     start=None, 
    #     end=450_000, 
    #     smoothing_window=1500, 
    #     title="All HAMMER Variations on \nCooperative Navigation", 
    #     labels=labels, 
    #     colors = [], #['#006BB2', '#006BB2', '#B22400', '#FFC325', '#A9A9A9', '#A8A8A8'], 
    #     save=False 
    # )

    # """ 
    # Modified CN 
    # """ 

    # publication_plot(
    #     res=np.load("rs_npy/modified_cn.npy", allow_pickle=True).item(), 
    #     start=None, 
    #     end=450_000, 
    #     smoothing_window=2000, 
    #     title="All HAMMER Variations on \nModified Cooperative Navigation", 
    #     labels={
    #         "Hammerwithgradients--cn--partialobs1--meslen1--dru": {"label": "HAMMERv3", "color": "#377eb8"}, 
    #         "Hammerwithgradients--cn--partialobs1--meslen1--nodru": {"label": "HAMMERv2", "color": "#4daf4a"}, 
    #         "Hammer_previous--cn--partialobs1--meslen1--discretemes0": {"label": "HAMMERv1", "color": "#984ea3"}, 
    #         # "Hammer_previous--cn--partialobs0--meslen1--discretemes1": {"label": "", "color": ""}, 
    #         "ILsharedparams--cn--partialobs1": {"label": "IL", "color": "#e41a1c"}, 
    #     }, 
    #     colors = [], #['#006BB2', '#006BB2', '#B22400', '#FFC325', '#A9A9A9', '#A8A8A8'], 
    #     save=False 
    # )


    """ 
    Heterogeneity in CN 
    """ 

    publication_plot(
        res=np.load("rs_npy/heterogeneity.npy", allow_pickle=True).item(), 
        start=None, 
        end=450_000, 
        smoothing_window=5000, 
        title="All HAMMER Variations on \nHeterogeneous Settings in Cooperative Navigation", 
        labels={
            "Hammerwithgradients--cn--heterogeneity1--meslen1--dru": {"label": "HAMMERv3", "color": "#377eb8"}, 
            "Hammerwithgradients--cn--heterogeneity1--meslen1--nodru": {"label": "HAMMERv2", "color": "#4daf4a"}, 
            "Hammer_previous--cn--heterogeneity1--meslen1--discretemes0": {"label": "HAMMERv1", "color": "#984ea3"}, 
            # "Hammer_previous--cn--partialobs0--meslen1--discretemes1": {"label": "", "color": ""}, 
            "ILsharedparams--cn--heterogeneity1": {"label": "IL", "color": "#e41a1c"}, 
        }, 
        colors = [], #['#006BB2', '#006BB2', '#B22400', '#FFC325', '#A9A9A9', '#A8A8A8'], 
        save=True 
    )
