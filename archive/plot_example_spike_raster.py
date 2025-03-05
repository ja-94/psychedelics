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
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Get paths
path_dict = paths()

PID = '540ea29f-610e-47b5-9d2b-cc2425c63ec7'
        
# Load in spikes
sl = SpikeSortingLoader(pid=PID, one=one, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting()
clusters = sl.merge_clusters(spikes, clusters, channels)

# Only keep IBL good neurons
spikes.times = spikes.times[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]
spikes.depths = spikes.depths[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]
spikes.clusters = spikes.clusters[np.isin(spikes.clusters, clusters.cluster_id[clusters.label == 1])]

# Convert to mm
spikes.depths = spikes.depths / 1000

# Get spike raster
iok = ~np.isnan(spikes.depths)
R, times, depths = bincount2D(spikes.times[iok], spikes.depths[iok], xbin=0.01, ybin=0.02, weights=None)
    
# %% Plot figure
f, ax1 = plt.subplots(1, 1, figsize=(5, 2.5), dpi=300)
ax1.imshow(R, aspect='auto', cmap='binary', vmin=0, vmax=np.std(R),
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='upper')
ax1.set(ylim=[0, 4], ylabel='Depth (mm)')
#ax1.set_title('Awake', color=colors['awake'], fontweight='bold')
ax1.set(xticks=[ax1.get_xlim()[0] + 60, ax1.get_xlim()[0] + 660])
ax1.text(ax1.get_xlim()[0] + 350, 4.3, '10 min', ha='center', va='center')
ax1.axes.get_xaxis().set_visible(False)
ax1.invert_yaxis()

for i in np.arange(1, channels['acronym'].shape[0], 50):
    ax1.text(ax1.get_xlim()[-1]+50, channels['axial_um'][channels['acronym'].shape[0] - i] / 1000,
             channels['acronym'][i], fontsize=8)

plt.tight_layout()
sns.despine(trim=True, offset=2)
plt.savefig(join(path_dict['fig_path'], 'example_raster.pdf'))