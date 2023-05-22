# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:03:05 2023 by Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
from psychedelic_functions import remap, paths
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE
from ibllib.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Get paths
#path_dict = paths()

# Query pids
pids = list(one.search_insertions(project='psychedelics', query_type='remote'))

neurons_df = pd.DataFrame()
for i, pid in enumerate(pids):
    print(f'Recording {i} of {len(pids)}')
        
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
    probe = one.pid2eid(pid)[1]
    print(f'{subject} {date}')
    
    # Only keep good neurons
    clusters.acronym = clusters.acronym[clusters.label == 1]
    
    # Remap Allen acronyms to Beryl
    clusters['beryl_acronyms'] = remap(clusters['acronym'])
    
    # Get number of neurons per region
    n_neurons = np.unique(clusters['beryl_acronyms'], return_counts=True)
    
    # Add to dataframe
    this_df = pd.DataFrame(data={'region': n_neurons[0], 'n_neurons': n_neurons[1]})
    this_df['subject'] = subject
    this_df['date'] = date
    neurons_df = pd.concat((neurons_df, this_df))
    
# Save dataframe
neurons_df.to_csv('C:/Users/Asus/int-brain-lab/psychedelics_project_folder/Figures/n_neurons.csv')
    
    
    