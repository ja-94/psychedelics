#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:18:07 2023

@author: joana
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from os.path import join
from psychedelic_functions import paths, load_tracking

# Select sessions for all doses 
SESSION_Control = '20221223_05568_C'
SESSION_SuperLow = '20230119_05568_S'
SESSION_Low = '20221202_05568_L'
SESSION_Medium = '20221209_05568_M'
SESSION_High = '20221216_05568_H'

#Give Path directory for all doses 
path_dict_Control = paths()
path_Control = join(path_dict_Control['data_path'], 'OpenField', 'Tracking', 'Control', 'ZFM-05568')

path_dict_SuperLow = paths()
path_SuperLow = join(path_dict_SuperLow['data_path'], 'OpenField', 'Tracking', 'Super_Low_Dose', 'ZFM-05568')

path_dict_Low = paths()
path_Low = join(path_dict_Low['data_path'], 'OpenField', 'Tracking', 'Low_Dose', 'ZFM-05568')

path_dict_Medium = paths()
path_Medium = join(path_dict_Medium['data_path'], 'OpenField', 'Tracking', 'Medium_Dose', 'ZFM-05568')

path_dict_High = paths()
path_High = join(path_dict_High['data_path'], 'OpenField', 'Tracking', 'High_Dose', 'ZFM-05568')

# Change title of the plot
subtitle_string = 'Tracks colored by magnitude of speed  ZFM-05568 - 40-50mins'

#General Settings
FRAME_RATE = 30


# IMPORTANT!!! CHANGE PLT.SAVE NAME AT THE END OF THE SCRIPT  !!!!


#%%

# ---------------------- CONTROL SESSION ----------------------

# Settings
NODE_Control = 'nose'

# Load in SLEAP tracking
tracking = load_tracking(join(path_Control, SESSION_Control + '.h5'))
tracks_array_Control = tracking['tracks']

# Generate time axis
time_ax_Control = np.linspace(0, tracks_array_Control.shape[0] / FRAME_RATE, tracks_array_Control.shape[0])

# Get node to plot
node_ind_Control = [i for i, node in enumerate(tracking['node_names']) if NODE_Control in node][0]

# Get node location
node_loc = tracks_array_Control[:,node_ind_Control,:] #node_ind = 0 so it means we are looking at nose location

# Get node location for a specific time during the session
#node_loc_Control = tracks_array_Control[time_ax_Control > time_ax_Control[-1] - (10*60), node_ind_Control, :] Last 10 min

node_loc_Control = tracks_array_Control[(time_ax_Control > (40*60)) & (time_ax_Control < (50*60)), node_ind_Control, :] # From min40 to min50


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

node_loc_fill = fill_missing(node_loc_Control) 


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

# Get velocity for this specific session 
node_vel_Control = smooth_diff(node_loc_Control) 

#%%

# --------------------------------- SUPER LOW DOSE SESSION -----------------------------------

# Settings
NODE_SuperLow = 'nose'

# Load in SLEAP tracking
tracking = load_tracking(join(path_SuperLow, SESSION_SuperLow + '.h5'))
tracks_array_SuperLow = tracking['tracks']

# Generate time axis
time_ax_SuperLow = np.linspace(0, tracks_array_SuperLow.shape[0] / FRAME_RATE, tracks_array_SuperLow.shape[0])

# Get node to plot
node_ind_SuperLow = [i for i, node in enumerate(tracking['node_names']) if NODE_SuperLow in node][0]

# Get node location
node_loc = tracks_array_SuperLow[:,node_ind_SuperLow,:] #node_ind = 0 so it means we are looking at nose location

# Get node location for a specific time during the session
#node_loc_SuperLow = tracks_array_SuperLow[time_ax_SuperLow > time_ax_SuperLow[-1] - (10*60), node_ind_SuperLow, :]
node_loc_SuperLow = tracks_array_SuperLow[(time_ax_SuperLow > (40*60)) & (time_ax_SuperLow < (50*60)), node_ind_SuperLow, :]

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

node_loc_fill = fill_missing(node_loc_SuperLow) 


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

# Get velocity for this specific session 
node_vel_SuperLow = smooth_diff(node_loc_SuperLow) 


#%% 

#---------------------------------- LOW DOSE SESSION --------------------------------------------

# Settings
NODE_Low = 'nose'

# Load in SLEAP tracking
tracking = load_tracking(join(path_Low, SESSION_Low + '.h5'))
tracks_array_Low = tracking['tracks']

# Generate time axis
time_ax_Low = np.linspace(0, tracks_array_Low.shape[0] / FRAME_RATE, tracks_array_Low.shape[0])

# Get node to plot
node_ind_Low = [i for i, node in enumerate(tracking['node_names']) if NODE_Low in node][0]

# Get node location
node_loc = tracks_array_Low[:,node_ind_Low,:] #node_ind = 0 so it means we are looking at nose location

# Get node location for a specific time during the session
#node_loc_Low = tracks_array_Low[time_ax_Low > time_ax_Low[-1] - (10*60), node_ind_Low, :]
node_loc_Low = tracks_array_Low[(time_ax_Low > (40*60)) & (time_ax_Low < (50*60)), node_ind_Low, :]

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

node_loc_fill = fill_missing(node_loc_Low) 


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

# Get velocity for this specific session 
node_vel_Low = smooth_diff(node_loc_Low) 


#%%

# ------------------------------------- MEDIUM DOSE SESSION --------------------------------------------

# Settings
NODE_Medium = 'nose'

# Load in SLEAP tracking
tracking = load_tracking(join(path_Medium, SESSION_Medium + '.h5'))
tracks_array_Medium = tracking['tracks']

# Generate time axis
time_ax_Medium = np.linspace(0, tracks_array_Medium.shape[0] / FRAME_RATE, tracks_array_Medium.shape[0])

# Get node to plot
node_ind_Medium = [i for i, node in enumerate(tracking['node_names']) if NODE_Medium in node][0]

# Get node location
node_loc = tracks_array_Medium[:,node_ind_Medium,:] #node_ind = 0 so it means we are looking at nose location

# Get node location for a specific time during the session
#node_loc_Medium = tracks_array_Medium[time_ax_Medium > time_ax_Medium[-1] - (10*60), node_ind_Medium, :]
node_loc_Medium = tracks_array_Medium[(time_ax_Medium > (40*60)) & (time_ax_Medium < (50*60)), node_ind_Medium, :]

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

node_loc_fill = fill_missing(node_loc_Medium) 


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

# Get velocity for this specific session 
node_vel_Medium = smooth_diff(node_loc_Medium) 


#%%

# ------------------------------------- HIGH DOSE SESSION ---------------------------------


# Settings
NODE_High = 'nose'

# Load in SLEAP tracking
tracking = load_tracking(join(path_High, SESSION_High + '.h5'))
tracks_array_High = tracking['tracks']

# Generate time axis
time_ax_High = np.linspace(0, tracks_array_High.shape[0] / FRAME_RATE, tracks_array_High.shape[0])

# Get node to plot
node_ind_High = [i for i, node in enumerate(tracking['node_names']) if NODE_High in node][0]

# Get node location
node_loc = tracks_array_High[:,node_ind_High,:] #node_ind = 0 so it means we are looking at nose location

# Get node location for a specific time during the session
#node_loc_High = tracks_array_High[time_ax_High > time_ax_High[-1] - (10*60), node_ind_High, :]
node_loc_High = tracks_array_High[(time_ax_High > (40*60)) & (time_ax_High < (50*60)), node_ind_High, :]

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

node_loc_fill = fill_missing(node_loc_High) 


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

# Get velocity for this specific session 
node_vel_High = smooth_diff(node_loc_High) 


#%%

# ----------------- General Plot  -------------------------------------

# Plot track colored by magnitude of speed 

kp1 = node_vel_Control
kp2 = node_vel_SuperLow
kp3 = node_vel_Low
kp4 = node_vel_Medium
kp5 = node_vel_High
vmin=0
vmax=10

fig = plt.figure(figsize=(25,6), dpi=300)
plt.suptitle(subtitle_string, fontsize=17, color='#2F4858', ha='center')

ax1 = fig.add_subplot(151)
ax1.scatter(node_loc_Control[:, 0], node_loc_Control[:, 1], c=kp1, s=4, vmin=vmin, vmax=vmax)
ax1.axis('off')
ax1.set_title('Control', fontsize=14)

ax2 = fig.add_subplot(152)
sct_plt = ax2.scatter(node_loc_SuperLow[:, 0], node_loc_SuperLow[:, 1], c=kp2, s=4, vmin=vmin, vmax=vmax)
ax2.set_title('Super Low Dose', fontsize=14)
ax2.axis('off')

ax3 = fig.add_subplot(153)
sct_plt = ax3.scatter(node_loc_Low[:, 0], node_loc_Low[:, 1], c=kp3, s=4, vmin=vmin, vmax=vmax)
ax3.set_title('Low Dose', fontsize=14)
ax3.axis('off')

ax4 = fig.add_subplot(154)
sct_plt = ax4.scatter(node_loc_Medium[:, 0], node_loc_Medium[:, 1], c=kp4, s=4, vmin=vmin, vmax=vmax)
ax4.set_title('Medium Dose', fontsize=14)
ax4.axis('off')

ax5 = fig.add_subplot(155)
sct_plt = ax5.scatter(node_loc_High[:, 0], node_loc_High[:, 1], c=kp5, s=4, vmin=vmin, vmax=vmax)
ax5.set_title('High Dose', fontsize=14)
ax5.axis('off')

fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.91, 0.15, 0.012, 0.72])
cbar = fig.colorbar(mappable=sct_plt, cax=cax)
cbar.set_label('Velocity', rotation=270, fontsize=12)
cbar.ax.get_yaxis().labelpad = 15

plt.subplots_adjust(wspace=0.05)

plt.savefig(join('/home/joana/Desktop/LSD_project/Figures_OF_tracking', 'ZFM-05568_Track_with_Velocity_AllSessions_40-50.png'))
plt.savefig(join('/home/joana/Desktop/LSD_project/Figures_OF_tracking', 'ZFM-05568_Track_with_Velocity_AllSessions_40-50.pdf'))



