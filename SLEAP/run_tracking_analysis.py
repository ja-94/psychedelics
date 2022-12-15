# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:05:50 2022

@author: Guido
"""

import numpy as np
import seaborn as sns
from os.path import join, split
from psychedelic_functions import paths, load_tracking, load_subjects
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import math

# Settings
DOSAGES = ['Low_Dose', 'Medium_Dose', 'High_Dose']
TITLES = ['Low dose (75 ug/kg)', 'Medium dose (150 ug/kg)', 'High dose (300 ug/kg)']
NODE = 'nose'
FRAME_RATE = 30  # frames/s
BIN_SIZE = 2  # minutes
PLOT = False
BASELINE = [10, 20]

# Get paths
path_dict = paths()
data_dir = join(path_dict['data_path'], 'OpenField', 'Tracking')

# Load subject data
subjects = load_subjects()


def distance(row):
    x1, y1, x2, y2 = row['X'], row['Y'], row['X2'], row['Y2']
    if np.isnan(x2) or np.isnan(y2):
        return 0
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Loop over different dosages
dist_df = pd.DataFrame()
for i, this_dose in enumerate(DOSAGES):
    # Get list of subjects
    sub_dirs = glob(join(data_dir, this_dose, '*'))
    # Loop over subjects
    for j, this_sub in enumerate(sub_dirs):
        # Get list of sessions
        ses_paths = glob(join(this_sub, '*.h5'))
        if len(ses_paths) == 0:
            continue

        # Loop over sessions
        ses_df = pd.DataFrame()
        if PLOT:
            f, axs = plt.subplots(1, len(ses_paths), figsize=(2*len(ses_paths), 2.5), dpi=300)
            if len(ses_paths) == 1:
                axs = [axs]
        for k, this_ses in enumerate(ses_paths):

            # Load in tracking data of this session
            tracking = load_tracking(this_ses)

            # Generate time axis
            time_ax = np.linspace(0, tracking['tracks'].shape[0] / FRAME_RATE,
                                  tracking['tracks'].shape[0])

            # Get index to node to use
            node_ind = [i for i, node in enumerate(tracking['node_names']) if NODE in node][0]

            # Add to dataframe
            ses_df['X'] = tracking['tracks'][:, node_ind, 0]
            ses_df['Y'] = tracking['tracks'][:, node_ind, 1]
            ses_df['time'] = time_ax / 60

            # Calculate distance travelled
            ses_df['X2'] = ses_df['X'].shift(-1)
            ses_df['Y2'] = ses_df['Y'].shift(-1)
            ses_df['distance'] = ses_df.apply(distance, axis=1)
            ses_df['distance_bl'] = (
                ses_df['distance']
                - ses_df.loc[(ses_df['time'] >= BASELINE[0]) & (ses_df['time'] <= BASELINE[1]),
                             'distance'].mean())
            ses_df['binned_time'] = pd.cut(ses_df['time'], np.arange(0, 61, BIN_SIZE))

            # Get mean distance travelled per timebin
            binned_df = ses_df.groupby('binned_time').mean(numeric_only=True)['distance_bl']
            binned_df = binned_df.to_frame().reset_index(drop=True)
            binned_df['time'] = np.arange(0, 60, BIN_SIZE)
            binned_df['subject'] = split(this_sub)[-1]
            binned_df['dose'] = this_dose
            binned_df['date'] = split(this_ses)[-1][:8]
            binned_df['administration'] = subjects.loc[
                subjects['subject'] == split(this_sub)[-1], 'administration'].values[0]

            # Add to df
            dist_df = pd.concat((dist_df, binned_df))

            # Plot
            if PLOT:
                axs[k].plot(np.arange(0, 60, BIN_SIZE), binned_df['distance'])
                axs[k].set(ylabel='Distance travelled', xlabel='Time (m)', yticks=[0, 5, 10],
                        xticks=[0, 30, 60], title=split(this_sub)[-1])
        if PLOT:
            f.suptitle(f'{TITLES[i]}')
            plt.tight_layout()
            sns.despine(trim=True)


# %% Plot

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=300)
sns.lineplot(data=dist_df[dist_df['administration'] == 'catheter'], x='time', y='distance_bl',
             hue='dose', ax=ax1, errorbar='se')
ax1.set(ylabel='Distance travelled', xlabel='Time (m)', ylim=[-2.5, 7.5])
ax1.legend(title='', frameon=False, prop={'size': 7})

sns.lineplot(data=dist_df[dist_df['administration'] == 'ip'], x='time', y='distance_bl',
             hue='dose', ax=ax2, errorbar='se')
ax2.set(ylabel='', xlabel='Time (m)', ylim=[-2.5, 7.5])
ax2.legend(title='', frameon=False, prop={'size': 7})

plt.tight_layout()
sns.despine(trim=True)
