#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:04:09 2023

@author: joana
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from psychedelic_functions import paths, get_dlc_XYs, get_raw_smooth_pupil_diameter
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
BIN_SIZE = 2  # minutes

# Get paths
path_dict = paths()

# Query recordings
eids = one.search(projects=['psychedelics'], task_protocol='_iblrig_tasks_passiveChoiceWorldIndependent')

# Loop over recordings
for i, eid in enumerate(eids):
        
   # Get session details
   ses_details = one.get_details(eid)
   subject = ses_details['subject']
   date = ses_details['start_time'][:10]
   print(f'Starting {subject}, {date}')
   
   # Load in timestamps and DLC data
   video_times, XYs = get_dlc_XYs(one, eid)
   if video_times is None:
       continue
   
   # Get smoothed pupil diameter
   print('Smoothing pupil diameter trace')
   raw_diameter, diameter = get_raw_smooth_pupil_diameter(XYs)
   
   # Calculate percentage change versus the lowest 2 percentiles
   diameter_perc = ((diameter - np.percentile(diameter[~np.isnan(diameter)], 2))
                    / np.percentile(diameter[~np.isnan(diameter)], 2)) * 100
   
   # Downsample trace 
   sampling_rate = 1 / np.mean(np.diff(video_times))
   binsize = int((BIN_SIZE * 60) * sampling_rate)
   end = binsize * int(diameter_perc.shape[0] / binsize)
   diameter_binned = np.nanmean(diameter_perc[:end].reshape(-1, binsize), 1)
   time_binned = np.arange(BIN_SIZE / 2, BIN_SIZE * diameter_binned.shape[0], BIN_SIZE)
   
   # Plot this session
   
   f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3), gridspec_kw={'width_ratios':[4.5, 0.08, 4.5]}, dpi=300)
   subtitle_string = f'{subject}, {date}'
   plt.suptitle(subtitle_string, fontsize=12, color='#2F4858', ha='center', y=1.06)
   
   ax1.plot(video_times / 60, diameter_perc, color = '#38a3a5')
   ax1.set(xlabel='Time (min)', ylabel='Pupil diameter (%)')
   ax1.set_xlabel('Time (min)', fontsize=10, labelpad=14)
   ax1.tick_params(labelsize=9)
   
   ax2.set(visible=False)
   
   ax3.plot(time_binned, diameter_binned, marker='o', ms= 6, color = '#38a3a5')
   ax3.set(xlabel='Time (min)', ylabel='Pupil diameter (%)')
   ax3.yaxis.set_ticks(np.arange(0, 200, 50))
   ax3.set_xlabel('Time (min)', fontsize=10, labelpad=14)
   
   plt.subplots_adjust(hspace=0.05)
   
   sns.despine(trim=True)
   #plt.tight_layout()
   
  
'''    
palette = sns.color_palette('husl', 5)


# Plot distance travelled (Catheter vs IP) - All session 
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3), gridspec_kw={'width_ratios':[4.5, 0.08, 4.5]}, dpi=500)

sns.lineplot(data=dist_df[dist_df['administration'] == 'catheter'], x='time', y='distance_bl',
             hue='dose', ax=ax1, errorbar='se', palette=palette)
ax1.set(ylabel='Distance traveled', xlabel='Time (m)', ylim=[-4, 8], )
ax1.legend(title='', frameon=False, prop={'size':9}, loc='upper center', bbox_to_anchor=(1.1, -0.35), ncol=5)
ax1.set_xlabel('Time (m)', fontsize=10, labelpad=12)
ax1.tick_params(labelsize=9)
ax1.set_title('Administration: Catheter', fontsize=10, pad=11)

ax2.set(visible=False)

sns.lineplot(data=dist_df[dist_df['administration'] == 'ip'], x='time', y='distance_bl',
             hue='dose', ax=ax3, errorbar='se', palette=palette)
ax3.set(ylabel='Distance traveled', xlabel='Time (m)', ylim=[-4, 8])
ax3.set_xlabel('Time (m)', fontsize=10, labelpad=12)
ax3.tick_params(labelsize=9)
ax3.legend([], frameon=False)
ax3.set_title('Administration: IP', fontsize=10, pad=11)

sns.despine(trim=True)


# Just to keep

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2), dpi=150)
ax1.plot(video_times / 60, diameter_perc)
ax1.set(xlabel='Time (min)', ylabel='Pupil diameter (%)', title=f'{subject}, {date}')

ax2.plot(time_binned, diameter_binned, marker='o')
ax2.set(xlabel='Time (min)', ylabel='Pupil diameter (%)')

sns.despine(trim=True)
plt.tight_layout()