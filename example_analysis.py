# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:39:45 2023 by Guido Meijer
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from brainbox.io.one import SpikeSortingLoader
from brainbox.processing import bincount2D
from psychedelic_functions import paths
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Get paths
path_dict = paths()

# Query recordings
pids = list(one.search_insertions(project='psychedelics', query_type='remote'))

# Loop over recordings
for i, pid in enumerate(pids):
        
    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    if len(spikes) == 0:
        continue
    
    # Get spike raster
    iok = ~np.isnan(spikes.depths)
    R, times, depths = bincount2D(spikes.times[iok], spikes.depths[iok], 0.01, 20, weights=None)
        
    # Plot figure
    f, ax1 = plt.subplots(1, 1, figsize=(5, 2.5), dpi=300)
    ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R) * 2,
              extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='lower')
    plt.tight_layout()
    sns.despine(trim=True, offset=4)
    
    plt.savefig(join(path_dict['fig_path'], f'raster_{pid}.jpg'), dpi=600)
    plt.close(f)
    