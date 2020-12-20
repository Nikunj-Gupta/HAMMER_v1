import glob
import os

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
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
    SCALARS: 25000,
    HISTOGRAMS: 1,
}


def get_ea(filename):
    ea = event_accumulator.EventAccumulator(filename, size_guidance=SIZE_GUIDANCE)
    ea.Reload()
    x = ea.Scalars('Running_Reward')
    return x


def smooth(ea_scalar, factor=None, sum_factor=None):
    ea_scalar = pd.DataFrame(ea_scalar)
    ea_scalar = ea_scalar["value"]
    if sum_factor:
        ea_scalar = ea_scalar.groupby(ea_scalar.index // sum_factor).sum()
        print(ea_scalar.describe())

    if factor:
        return ea_scalar.rolling(factor).mean()
    return ea_scalar


exp = "multiwalker-2"
path = os.path.join("logs", exp)
files = glob.glob(path+"/event*")
filename = files[0]
ea_scalar = get_ea(filename)
arr = smooth(ea_scalar, sum_factor=50)
print(arr)
