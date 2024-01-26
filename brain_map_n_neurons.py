#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:19:58 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from iblatlas.plots import plot_scalar_on_slice
from psychedelic_functions import paths
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas(res_um=10)

# Settings
AP = [2, -1.5, -3.5]

# Paths
path_dict = paths()

# Load in results
all_neurons = pd.read_csv(join(path_dict['save_path'], 'n_neurons.csv'))
#all_neurons = pd.read_csv(join(path_dict['data_path'], 'n_neurons.csv'))
all_neurons = all_neurons[all_neurons['region'] != 'root']
all_neurons = all_neurons[all_neurons['region'] != 'void']

# %%

CMAP = 'turbo' 

# Plot brain map slices
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 4), dpi=150)

plot_scalar_on_slice(all_neurons['region'].values, np.log10(all_neurons['n_neurons'].values), ax=ax1,
                     slice='coronal', coord=AP[0]*1000, background='boundary', brain_atlas=ba,
                     mapping='Beryl', cmap=CMAP, clevels=[0, 2])
ax1.axis('off')
#ax1.set(title=f'+{np.abs(AP[0])} mm AP')

plot_scalar_on_slice(all_neurons['region'].values, np.log10(all_neurons['n_neurons'].values), ax=ax2,
                     slice='coronal', coord=AP[1]*1000, background='boundary', brain_atlas=ba,
                     mapping='Beryl', cmap=CMAP, clevels=[0, 2])
ax2.axis('off')
#ax2.set(title=f'-{np.abs(AP[1])} mm AP')

plot_scalar_on_slice(all_neurons['region'].values, np.log10(all_neurons['n_neurons'].values), ax=ax3,
                     slice='coronal', coord=AP[2]*1000, background='boundary', brain_atlas=ba,
                     mapping='Beryl', cmap=CMAP, clevels=[0, 2])
ax3.axis('off')
#ax3.set(title=f'-{np.abs(AP[2])} mm AP')

sns.despine()

f.subplots_adjust(right=0.85)
# lower left corner in [0.88, 0.3]
# axes width 0.02 and height 0.4
cb_ax = f.add_axes([0.88, 0.42, 0.01, 0.2])
cbar = f.colorbar(mappable=ax1.images[0], cax=cb_ax)
cbar.ax.set_ylabel('Total number of\nrecorded neurons (log)', rotation=270, labelpad=25)
cbar.ax.set_yticks([0, 1, 2])
cbar.ax.set_yticklabels([1, 10, 100])


