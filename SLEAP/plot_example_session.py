#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:06:53 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
from psychedelic_functions import paths, load_tracking
import matplotlib.pyplot as plt

# Settings
SESSION = '20221130_05494_L'
FRAME_RATE = 30
MAP_TIME = 60 * 3
NODE = 'nose'
path_dict = paths()
path = join(path_dict['data_path'], 'OpenField', 'Tracking', 'Low_Dose', 'ZFM-05494')

# Load in SLEAP tracking
tracking = load_tracking(join(path, SESSION + '.h5'))
tracks_array = tracking['tracks']

# Generate time axis
time_ax = np.linspace(0, tracks_array.shape[0] / FRAME_RATE, tracks_array.shape[0])

# Get node to plot
node_ind = [i for i, node in enumerate(tracking['node_names']) if NODE in node][0]

# Plot occupancy map
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.25, 2), dpi=300)
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