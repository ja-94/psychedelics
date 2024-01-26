# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:05:39 2024

@author: Guido
"""

import numpy as np
import pandas as pd
from os.path import join
from psychedelic_functions import remap, paths, query_recordings
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Query recordings
rec = query_recordings(aligned=False)

# Loop over recordings
for i, pid in enumerate(rec['pid']):
    print(f'Recording {i} of {rec.shape[0]}')
        
    # Load in neural data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
            
    # No neural data
    if len(spikes) == 0:
        continue
    
    # No histology
    if 'acronym' not in clusters.keys():
        continue
    
    
    