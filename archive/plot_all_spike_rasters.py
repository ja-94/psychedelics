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
from psychedelic_functions import paths, query_recordings
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Get paths
path_dict = paths()

# Query recordings
rec = query_recordings(aligned=False)

# Loop over recordings
for i, pid in enumerate(rec['pid']):
        
    # Load in spikes
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    if len(spikes) == 0:
        continue
    
    # Only keep IBL good neurons
    spikes.times = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]
    spikes.depths = spikes.depths[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]
    
    if spikes.times.shape[0] == 0:
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
    
    plt.savefig(join(path_dict['fig_path'], 'SpikeRasters', f'raster_{pid}.jpg'), dpi=600)
    plt.close(f)
    