import glob
import os 
import argparse 
from utils import read_config

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd 
import matplotlib.pyplot as plt 

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
    SCALARS: 50000,
    HISTOGRAMS: 1,
}


def get_values(filename, scalar="Avg_reward_for_each_agent__after_an_episode", smooth=2000, ignore_start=0, ignore_end=None): 
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



def plot(config): 
    for fig in config: 
        ax = plt.gca() 
        for line in config[fig]: 
            print(line) 
            curve = config[fig][line] 
            df = get_values(filename=curve["filename"]) 
            df.plot(kind="line", x='step', y='value', ax=ax, label=curve["label"]) 
        plt.show() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/2021/plots.yaml', help="config file name") 
    args = parser.parse_args() 

    config = read_config(args.config) 

    plot(config) 



