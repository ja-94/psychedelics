# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:35:33 2023

@author: Joana
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:04:09 2023

@author: joana

script to plot the pupil only for specific animals
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


# Query recordings
eids = one.search(projects=['psychedelics'], task_protocol='_iblrig_tasks_passiveChoiceWorldIndependent')

eid = eids[0]
admin_time = 17.53
second_passive = (admin_time + 30)

# Get session details
ses_details = one.get_details(eid)
subject = ses_details['subject']
date = ses_details['start_time'][:10]
print(f'Starting {subject}, {date}')

# Load in timestamps and DLC data
video_times, XYs = get_dlc_XYs(one, eid)

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

f, (ax1) = plt.subplots(1, 1, figsize=(6, 3), dpi=900)
subtitle_string = f'{subject}, {date}'
plt.suptitle(subtitle_string, fontsize=12, color='#2F4858', ha='center', y=1.06)

ax1.plot(time_binned, diameter_binned, marker='o', ms= 6, color = '#38a3a5')
ax1.set(xlabel='Time (min)', ylabel='Pupil diameter (%)')
ax1.yaxis.set_ticks(np.arange(0, 250, 50))
ax1.xaxis.set_ticks(np.arange(0,80,10))
plt.axvline(x = admin_time, color = 'red', label = 'LSD administration (150ug/Kg)', linestyle='solid')
plt.axvline(x = second_passive, color = '#AD7BD9', label = 'Start of second passive protocol', linestyle='dashed')
ax1.set_xlabel('Time (min)', fontsize=10, labelpad=14)

plt.subplots_adjust(hspace=0.05)

sns.despine(trim=True)

plt.savefig('/home/guido/Desktop/Joana/Figures/Mice LSD project/Pupil_figures_passive_mark/'f'{subject}_{date}.png', bbox_inches='tight')
plt.savefig('/home/guido/Desktop/Joana/Figures/Mice LSD project/Pupil_figures_passive_mark/'f'{subject}_{date}.pdf', bbox_inches='tight')

