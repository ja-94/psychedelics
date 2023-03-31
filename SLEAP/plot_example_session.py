#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:06:53 2022
By: Guido Meijer and Joana Catarino
"""

import numpy as np
from os.path import join
from psychedelic_functions import paths, load_tracking
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
SESSION = '20230103_05494_S'
FRAME_RATE = 30
MAP_TIME = 60 * 3
NODE = 'nose'
path_dict = paths()
path = join(path_dict['data_path'], 'OpenField', 'Tracking', 'Super_Low_Dose', 'ZFM-05494')

# Load in SLEAP tracking
tracking = load_tracking(join(path, SESSION + '.h5'))
tracks_array = tracking['tracks']

# Generate time axis
time_ax = np.linspace(0, tracks_array.shape[0] / FRAME_RATE, tracks_array.shape[0])

# Get node to plot
node_ind = [i for i, node in enumerate(tracking['node_names']) if NODE in node][0]

# Plot occupancy map (min=3 + min=10-13 + last 3 min)

    # colors: Catheter = standard  IP = 'purple'
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 2), dpi=300)
plt.suptitle('Open Field  ZFM-05494  Super Low Dose')

ax1.plot(tracks_array[time_ax < MAP_TIME, node_ind, 0],
         tracks_array[time_ax < MAP_TIME, node_ind, 1])
ax1.set(title='First 3 min')
ax1.axis('off')

ax2.plot(tracks_array[(time_ax > (10*60)) & (time_ax < (10*60) + MAP_TIME), node_ind, 0],
         tracks_array[(time_ax > (10*60)) & (time_ax < (10*60) + MAP_TIME), node_ind, 1])
ax2.set(title='Min 10 - 13')
ax2.axis('off')

ax3.plot(tracks_array[time_ax > (time_ax[-1] - MAP_TIME), node_ind, 0],
         tracks_array[time_ax > (time_ax[-1] - MAP_TIME), node_ind, 1])
ax3.set(title='Last 3 min')
ax3.axis('off')

plt.tight_layout()


# Save Figure 
plt.savefig(join('/home/joana/Desktop/LSD_project/Figures_OF_tracking', f'{SESSION}_OccupancyMap.png'))
plt.savefig(join('/home/joana/Desktop/LSD_project/Figures_OF_tracking', f'{SESSION}_OccupancyMap.pdf'))


#%% velocity - tests

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from os.path import join
from psychedelic_functions import paths, load_tracking


# Settings
SESSION = '20221215_05494_H'
FRAME_RATE = 30
MAP_TIME = 60 * 10
NODE = 'nose'
path_dict = paths()
path = join(path_dict['data_path'], 'OpenField', 'Tracking', 'High_Dose', 'ZFM-05494')


# Load in SLEAP tracking
tracking = load_tracking(join(path, SESSION + '.h5'))
tracks_array = tracking['tracks']

# Generate time axis
time_ax = np.linspace(0, tracks_array.shape[0] / FRAME_RATE, tracks_array.shape[0])

# Get node to plot
node_ind = [i for i, node in enumerate(tracking['node_names']) if NODE in node][0]

# Get node location
node_loc = tracks_array[:,node_ind,:] #node_ind = 0 so it means we are looking at nose location


#Fill missing values

def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

node_loc_fill = fill_missing(node_loc) # This is not adding much. The length of the array is the same as without filling missing values 


# Get velocity
def smooth_diff(node_loc, win=20, poly=3):
    """
    node_loc is a [frames, 2] array
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    node_loc_vel = np.zeros_like(node_loc)
    
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)
    
    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel


node_vel = smooth_diff(node_loc)


# PLot track + velocity

fig = plt.figure(figsize=(15,7), dpi=300)
plt.suptitle('Open Field  ZFM-05494  High_Dose')
ax1 = fig.add_subplot(211)
ax1.plot(node_loc[:, 0], label='x', color = '#51AACB') # blue
ax1.plot(-1*node_loc[:, 1], label='y', color = '#A881E8') # purple
plt.axvline(x = (len(node_loc))/2, color = '#FC6A31', linewidth=2.5) # Still ned to fix the line to represent the half-time of the sessions (more less 30 min)
ax1.legend()
ax1.set_xticks([])
ax1.set_title('Nose')

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.imshow(node_vel[:,np.newaxis].T, aspect='auto', vmin=0, vmax=10) # use cmap='inferno' to select a color
plt.axvline(x = (len(node_loc))/2, color = '#FC6A31', linewidth=2.5, label = 'Session half-time')
ax2.legend()
ax2.set_yticks([])
ax2.set_title('Velocity')

plt.tight_layout()

plt.savefig(join('/home/joana/Desktop/LSD_project/Figures_OF_tracking', f'{SESSION}_Track_Velocity.png'))
plt.savefig(join('/home/joana/Desktop/LSD_project/Figures_OF_tracking', f'{SESSION}_Track_Velocity.pdf'))


# Plot track colored by magnitude of speed 

time_ax = np.linspace(0, tracks_array.shape[0] / FRAME_RATE, tracks_array.shape[0])

node_loc_first_10 = tracks_array[time_ax < 10*60, node_ind, :] # Here we are plotting only the first 10 min of the session
node_vel_first_10 = smooth_diff(node_loc_first_10) 
node_loc_last_10 = tracks_array[time_ax > time_ax[-1] - (10*60), node_ind, :] # Here we are plotting only the last 10 min of the session 
node_vel_last_10 = smooth_diff(node_loc_last_10)
kp1 = node_vel_first_10
kp2 = node_vel_last_10 
vmin=0
vmax=10

fig = plt.figure(figsize=(15,6), dpi=300)
subtitle_string = 'Tracks colored by magnitude of speed  ZFM-05494  High_Dose'
plt.suptitle(subtitle_string, fontsize=17, color='#2F4858', ha='center')

ax1 = fig.add_subplot(121)
ax1.scatter(node_loc_first_10[:, 0], node_loc_first_10[:, 1], c=kp1, s=4, vmin=vmin, vmax=vmax)
ax1.set_xlim(0,1024)
ax1.set_xticks([])
ax1.set_ylim(0,1024)
ax1.set_yticks([])
ax1.axis('off')
ax1.set_title('First 10 min of session', fontsize=14)

ax2 = fig.add_subplot(122)
sct_plt = ax2.scatter(node_loc_last_10[:, 0], node_loc_last_10[:, 1], c=kp2, s=4, vmin=vmin, vmax=vmax)
ax2.set_xlim(0,1024)
ax2.set_xticks([])
ax2.set_ylim(0,1024)
ax2.set_yticks([])
ax2.set_title('Last 10 min of session', fontsize=14)
ax2.axis('off')

fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.9, 0.16, 0.019, 0.72])
cbar = fig.colorbar(mappable=sct_plt, cax=cax)

plt.subplots_adjust(wspace=0)

plt.savefig(join('/home/joana/Desktop/LSD_project/Figures_OF_tracking', f'{SESSION}_Track_with_Velocity.png'))
plt.savefig(join('/home/joana/Desktop/LSD_project/Figures_OF_tracking', f'{SESSION}_Track_with_Velocity.pdf'))


# Plot Velocity profiles 

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
plt.suptitle('Velocity Profiles  ZFM-05494  High_Dose')

sns.lineplot(data=node_vel_first_10, ax=ax1, color='#51AACB')
ax1.set(ylabel='Velocity', xlabel='Time (ms)', ylim=[0, 45])
ax1.set(title='First 10 min of session')

sns.lineplot(data=node_vel_last_10, ax=ax2, color='#51AACB')
ax2.set(ylabel='Velocity', xlabel='Time (ms)', ylim=[0, 45])
ax2.set(title='Last 10 min of session')

plt.margins(0)

plt.tight_layout()











