# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:28:11 2024

By Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from psychedelic_functions import paths, combine_regions, figure_style
colors, dpi = figure_style()

# Settings
MIN_MOD_NEURONS = 10
MIN_N_NEURONS = 0
MIN_NEURONS_PER_MOUSE = 5
MIN_REC = 3

# Load in data
path_dict = paths()
neuron_df = pd.read_csv(join(path_dict['save_path'], 'neuron_modulation.csv'))
neuron_df['region'] = combine_regions(neuron_df['allen_acronym'])

# Drop root and void
neuron_df = neuron_df.reset_index(drop=True)
neuron_df = neuron_df.drop(index=[i for i, j in enumerate(neuron_df['region']) if 'root' in j])

mod_neurons = neuron_df[neuron_df['significant']]
mod_neurons = mod_neurons.groupby('region').filter(lambda x: len(x) >= MIN_MOD_NEURONS)

# Calculate summary statistics
summary_df = neuron_df.groupby(['region']).sum(numeric_only=True)
summary_df['n_neurons'] = neuron_df.groupby(['region']).size()
summary_df = summary_df.reset_index()
summary_df['perc_mod'] =  (summary_df['significant'] / summary_df['n_neurons']) * 100
summary_df = summary_df[summary_df['n_neurons'] >= MIN_N_NEURONS]

# Summary statistics per mouse
per_mouse_df = neuron_df.groupby(['region', 'subject']).sum(numeric_only=True)
per_mouse_df['n_neurons'] = neuron_df.groupby(['region', 'subject']).size()
per_mouse_df['perc_mod'] = (per_mouse_df['significant'] / per_mouse_df['n_neurons']) * 100
per_mouse_df = per_mouse_df[per_mouse_df['n_neurons'] >= MIN_NEURONS_PER_MOUSE]
per_mouse_df = per_mouse_df.groupby('region').filter(lambda x: len(x) >= MIN_REC)
per_mouse_df = per_mouse_df.reset_index()

# Get ordered regions per mouse
ordered_regions_pm = per_mouse_df.groupby('region').mean(numeric_only=True).sort_values(
    'perc_mod', ascending=False).reset_index()

# %% Plot percentage modulated neurons per region

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=dpi)
sns.barplot(x='perc_mod', y='region', data=per_mouse_df,
            order=ordered_regions_pm['region'],
            color=[0.6, 0.6, 0.6], ax=ax1, errorbar='se')
#sns.swarmplot(x='perc_mod', y='region', data=per_mouse_df,
#              order=ordered_regions_pm['region'], ax=ax1, size=2, legend=None)
ax1.set(xlabel='Modulated neurons (%)', ylabel='', xlim=[0, 100], xticks=np.arange(0, 101, 20))
#ax1.legend(frameon=False, bbox_to_anchor=(0.8, 1.1), prop={'size': 5}, title='Mouse',
#           handletextpad=0.1)

#plt.tight_layout()
plt.subplots_adjust(left=0.45, bottom=0.2, right=0.95)
sns.despine(trim=True)
plt.savefig(join(path_dict['fig_path'], 'perc_modulated_neurons_per_region.jpg'), dpi=600)

# %%

PROPS = {'boxprops':{'facecolor':'none', 'edgecolor':'none'}, 'medianprops':{'color':'none'},
         'whiskerprops':{'color':'none'}, 'capprops':{'color':'none'}}
ORDER = mod_neurons.groupby('region').mean(numeric_only=True)['mod_index'].sort_values(
    ascending=False).reset_index()['region']

f, ax1 = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=dpi)
sns.stripplot(x='mod_index', y='region', ax=ax1, data=mod_neurons, order=ORDER,
              size=2, color='grey', zorder=1)
sns.boxplot(x='mod_index', y='region', ax=ax1, data=mod_neurons, showmeans=True,
            order=ORDER, meanprops={"marker": "|", "markeredgecolor": "red", "markersize": "8"},
            fliersize=0, zorder=2, **PROPS)
ax1.plot([0, 0], ax1.get_ylim(), ls='--', color='black', zorder=0)
ax1.set(ylabel='', xlabel='Modulation index', xlim=[-1.05, 1.05], xticklabels=[-1, -0.5, 0, 0.5, 1])
#ax1.spines['bottom'].set_position(('data', np.floor(ax1.get_ylim()[0]) - 0.4))
#plt.tight_layout()
plt.subplots_adjust(left=0.45, bottom=0.2, right=0.95)
sns.despine(trim=True)
plt.savefig(join(path_dict['fig_path'], 'modulation_per_neuron_per_region.jpg'), dpi=600)
