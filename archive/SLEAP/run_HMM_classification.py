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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
DO_PCA = True
PCA_DIMS = 5
BASELINE = [10, 20]

# Get paths
path_dict = paths()
fig_dir = join(path_dict['fig_path'], 'HMM')
data_dir = join(path_dict['data_path'], 'OpenField', 'Tracking')
results_dir = join(path_dict['data_path'], 'OpenField', 'BehavioralClassification')

# Load subject data
subjects = load_subjects()

<<<<<<< Updated upstream
=======

def distance(row):
    x1, y1, x2, y2 = row['X'], row['Y'], row['X2'], row['Y2']
    if np.isnan(x2) or np.isnan(y2):
        return 0
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


>>>>>>> Stashed changes
# Loop over different dosages
behav_df = pd.DataFrame()
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

            # Get session params
            subject = split(this_sub)[-1]
            date = split(this_ses)[-1][:8]

            # Load in tracking data of this session
            tracking = load_tracking(this_ses)

            # Generate time axis
            time_ax = np.linspace(0, tracking['tracks'].shape[0] / FRAME_RATE,
                                  tracking['tracks'].shape[0])
<<<<<<< Updated upstream

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

=======
            
            """
            # Reshape tracking data so that x and y are each their own column
            reshaped_tracks = np.reshape(tracking['tracks'], (tracking['tracks'].shape[0],
                                                              tracking['tracks'].shape[1]*2))
            """
            
            # Smooth traces and interpolate NaNs
            print('Smoothing and interpolating traces..')
            smooth_tracks = np.empty(tracking['tracks'].shape)
            for nn in range(tracking['tracks'].shape[1]):
                for xy in range(tracking['tracks'].shape[2]):
                    smooth_tracks[:, nn, xy] = smooth_interpolate_signal_sg(tracking['tracks'][:, nn, xy],
                                                                            window=5)
            
            # TO DO
            # Get distance 
            for nn in range(tracking['tracks'].shape[1]):
                math.sqrt(((smooth_tracks[1:, nn, 0] - smooth_tracks[:-1, nn, 0]) ** 2) + ((smooth_tracks[1:, nn, 1] - smooth_tracks[:-1, nn, 1]) ** 2))
                
                math.dist(smooth_tracks[0:1, nn, 0], smooth_tracks[0:1, nn, 1])
            
            # Do PCA
            pca = PCA(n_components=PCA_DIMS)
            ss = StandardScaler(with_mean=True, with_std=True)
            norm_tracks = ss.fit_transform(smooth_tracks)
            pca_tracks = pca.fit_transform(smooth_tracks)
            
            # Make an HMM and sample from it
            print('Fitting HMM..')
            if DO_PCA:
                arhmm = ssm.HMM(K, pca_tracks.shape[1], observations='ar')  # use an auto-regressive HMM
                arhmm.fit(pca_tracks)
                zhat = arhmm.most_likely_states(pca_tracks)
            else:
                arhmm = ssm.HMM(K, smooth_tracks.shape[1], observations='ar')  # use an auto-regressive HMM
                arhmm.fit(smooth_tracks)
                zhat = arhmm.most_likely_states(smooth_tracks)
                        
>>>>>>> Stashed changes
            # Get transition matrix
            transition_mat = arhmm.transitions.transition_matrix

            # Plot this session
            #cmap = sns.color_palette("tab10", K)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2), gridspec_kw={'width_ratios':[4,1]}, dpi=500)
            ax1.imshow(zhat[None,time_ax <= 60], aspect="auto", cmap='Set2', extent=[0, 60, 0, 800], vmin=0, vmax=K-1)
            ax1.plot(time_ax[time_ax <= 60], smooth_tracks[time_ax <= 60, 0], zorder=1, color='k', lw=0.5)
            ax1.set(xlabel='Time (s)', ylabel='Nose position')

            im = ax2.imshow(transition_mat, cmap='gray')
            ax2.set(title="Transition Matrix")

            plt.tight_layout()
            sns.despine(trim=True)
<<<<<<< Updated upstream
            plt.savefig(join(fig_dir, f'{subject}_{date}_{this_dose}_K{K}.jpg'), dpi=600)
            
            # Add to dataframe
            behav_df = pd.concat((behav_df, pd.DataFrame(data={
                'subject': subject, 'date': date, 'dose': this_dose, 'time': time_ax, 'behavior': zhat})))

        # Save
        behav_df.to_csv(join(results_dir, f'behavioral_classification_K{K}.csv'))



=======
            
            
>>>>>>> Stashed changes


