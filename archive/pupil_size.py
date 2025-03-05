# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:39:45 2023 by Guido Meijer
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
   f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2), dpi=150)
   ax1.plot(video_times / 60, diameter_perc)
   ax1.set(xlabel='Time (min)', ylabel='Pupil diameter (%)', title=f'{subject}, {date}')
   
   ax2.plot(time_binned, diameter_binned, marker='o')
   ax2.set(xlabel='Time (min)', ylabel='Pupil diameter (%)')
   
   sns.despine(trim=True)
   plt.tight_layout()
   
   