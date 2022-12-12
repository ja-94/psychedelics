# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:05:50 2022

@author: Guido
"""

import numpy as np
import seaborn as sns
from os.path import join, split
from psychedelic_functions import paths, load_tracking, load_subjects, smooth_interpolate_signal_sg
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import math
import ssm

# Settings
DOSAGES = ['Low_Dose', 'Medium_Dose']
K = 8  # number of behavioral states to infer
FRAME_RATE = 30  # frames/s
BIN_SIZE = 2  # minutes
PLOT = False
BASELINE = [10, 20]

# Get paths
path_dict = paths()
fig_dir = join(path_dict['fig_path'], 'HMM')
data_dir = join(path_dict['data_path'], 'OpenField', 'Tracking')
results_dir = join(path_dict['data_path'], 'OpenField', 'BehavioralClassification')

# Load subject data
subjects = load_subjects()


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
        for k, this_ses in enumerate(ses_paths):

            # Load in tracking data of this session
            tracking = load_tracking(this_ses)

            # Generate time axis
            time_ax = np.linspace(0, tracking['tracks'].shape[0] / FRAME_RATE,
                                  tracking['tracks'].shape[0])
            
            # Reshape tracking data so that x and y are each their own column
            reshaped_tracks = np.reshape(tracking['tracks'], (tracking['tracks'].shape[0],
                                                              tracking['tracks'].shape[1]*2))
            
            # Smooth traces and interpolate NaNs
            print('Smoothing and interpolating traces..')
            smooth_tracks = np.empty(reshaped_tracks.shape)
            for tt in range(reshaped_tracks.shape[1]):
               smooth_tracks[:, tt] = smooth_interpolate_signal_sg(reshaped_tracks[:, tt],  window=5)
            
            # Make an HMM and sample from it
            print('Fitting HMM..')
            arhmm = ssm.HMM(K, smooth_tracks.shape[1], observations='ar')  # use an auto-regressive HMM
            arhmm.fit(smooth_tracks)
            zhat = arhmm.most_likely_states(smooth_tracks)
                        
            # Get transition matrix
            transition_mat = arhmm.transitions.transition_matrix
            
            # Plot this session
            #cmap = sns.color_palette("tab10", K)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=300)
            ax1.imshow(zhat[None,time_ax <= 60], aspect="auto", cmap='tab10', extent=[0, 60, 0, 800], vmin=0, vmax=K-1)
            ax1.plot(time_ax[time_ax <= 60], reshaped_tracks[time_ax <= 60, 0], zorder=1, color='k', lw=0.5)
            ax1.set(xlabel='Time (s)', ylabel='Nose position')
            
            im = ax2.imshow(transition_mat, cmap='gray')
            ax2.set(title="Transition Matrix")
            
            plt.tight_layout()
            sns.despine(trim=True)
            
            ax1.plot
            

           

