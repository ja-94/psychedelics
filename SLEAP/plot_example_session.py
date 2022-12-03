#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:06:53 2022
By: Guido Meijer
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import h5py

# Settings
PATH = '/home/guido/Dropbox/Work/Data/Psychedelics/OpenField/Tracking/Low_Dose/ZFM-05488/'
SESSION = '20221130_05488_L'
FRAME_RATE = 30
MAP_TIME = 200

# Load in SLEAP tracking
with h5py.File(PATH + SESSION + '.h5', 'r') as f:
    node_names = f['node_names'][:]
    tracks_array = f['tracks'][:]
tracks_array = np.transpose(np.squeeze(tracks_array))

# Generate time axis
time_ax = np.linspace(0, tracks_array.shape[0] / FRAME_RATE, tracks_array.shape[0])

# Plot occupancy map
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2), dpi=300)
ax1.plot(tracks_array[time_ax < MAP_TIME, 4, 0], tracks_array[time_ax < MAP_TIME, 4, 1])
ax1.set(title='Control')
ax1.axis('off')

ax2.plot(tracks_array[time_ax > (time_ax[-1] - MAP_TIME), 4, 0],
         tracks_array[time_ax > (time_ax[-1] - MAP_TIME), 4, 1])
ax2.set(title='LSD')
ax2.axis('off')

plt.tight_layout()