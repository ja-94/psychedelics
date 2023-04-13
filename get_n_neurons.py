# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:03:05 2023 by Guido Meijer
"""

import numpy as np
import pandas as pd
from psychedelic_funcitons import remap
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Query pids
pids = list(one.search_insertions(project='psychedelics', query_type='remote'))

neurons_df = pd.DataFrame()
for i, pid in enumerate(pids):
        
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
    
    # Get recording info
    ses_details = one.get_details(one.pid2eid(pid)[0])
    subject = ses_details['subject']
    date = ses_details['date']
    probe = one.pid2eid()[1]
    
    # Remap Allen acronyms to Beryl
    clusters['beryl_acronyms'] = remap(clusters['acronyms'])
    
    # Get number of neurons per region
    n_neurons = np.unique(clusters['beryl_acronyms'], return_counts=True)
    
    # Add to dataframe
    this_df = pd.DataFrame(data=n_neurons)
    this_df['subject'] = subject
    this_df['date'] = date
    neurons_df = pd.concat((neurons_df, this_df))
    
    
    