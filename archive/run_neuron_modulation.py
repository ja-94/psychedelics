# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:05:39 2024

@author: Guido
"""

import numpy as np
import pandas as pd
from os.path import join
from psychedelic_functions import paths, query_recordings
from brainbox.io.one import SpikeSortingLoader
from sklearn.utils import shuffle
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()
path_dict = paths()

# Settings
TIME_WIN = 10  # min
PAD = 5  # min away from beginning and end of recording
QC = True
N_SHUFFLE = 500
OVERWRITE = False

# Query recordings
rec = query_recordings(aligned=False)

if OVERWRITE:
    neuron_df = pd.DataFrame()
else:
    neuron_df = pd.read_csv(join(path_dict['save_path'], 'neuron_modulation.csv'))
    rec = rec[~rec['pid'].isin(neuron_df['pid'])]

# Loop over recordings
for i, pid in enumerate(rec['pid']):
    print(f'\nRecording {i} of {rec.shape[0]}')
        
    # Get recording info
    eid, probe = one.pid2eid(pid)
    ses_info = one.get_details(eid)
    
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
    
    # Only keep good neurons
    if QC:
        use_neurons = np.where(clusters['label'] == 1)[0]
        spikes['times'] = spikes['times'][np.isin(spikes['clusters'], use_neurons)]
        spikes['clusters'] = spikes['clusters'][np.isin(spikes['clusters'], use_neurons)]
        clusters['acronym'] = clusters.acronym[use_neurons]
    else:
        use_neurons = clusters['cluster_id']
    if spikes['times'].shape[0] == 0:
        continue
    
    rec_length = spikes['times'][-1]
    for n, neuron_id in enumerate(use_neurons):
        if np.mod(n, 25) == 0:
            print(f'Neuron {n} of {len(use_neurons)}')
        
        # Calculate modulation index 
        these_spikes = spikes['times'][spikes['clusters'] == neuron_id]
        pre_count = np.sum((these_spikes >= PAD * 60) & (these_spikes <= (PAD + TIME_WIN) * 60))
        post_count = np.sum((these_spikes >= rec_length - ((PAD + TIME_WIN) * 60))
                            & (these_spikes <= rec_length - (PAD * 60)))
        mod_index = (post_count - pre_count) / (post_count + pre_count)
    
        # Shuffle inter-spike intervals and create null distribution of modulation index
        mod_idx_shuf = np.zeros(N_SHUFFLE)
        for ii in range(N_SHUFFLE):
            shuf_spikes = np.r_[these_spikes[0], shuffle(np.diff(these_spikes))].cumsum()
            shuf_pre_count = np.sum((shuf_spikes >= PAD * 60) & (shuf_spikes <= (PAD + TIME_WIN) * 60))
            shuf_post_count = np.sum((shuf_spikes >= rec_length - ((PAD + TIME_WIN) * 60))
                                & (shuf_spikes <= rec_length - (PAD * 60)))
            mod_idx_shuf[ii] = (shuf_post_count - shuf_pre_count) / (shuf_post_count + shuf_pre_count)
        mod_idx_shuf = mod_idx_shuf[~np.isnan(mod_idx_shuf)]
        if mod_idx_shuf.shape[0] == 0:
            shuf_quant = [np.nan, np.nan]
        else:
            shuf_quant = np.quantile(mod_idx_shuf, [0.025, 0.975])
        neuron_sig = (mod_index < shuf_quant[0]) | (mod_index > shuf_quant[1])
        
        # Add to dataframe
        neuron_df = pd.concat((neuron_df, pd.DataFrame(index=[neuron_df.shape[0]], data={
            'mod_index': mod_index, 'lower_quantile': shuf_quant[0], 'upper_quantile': shuf_quant[1],
            'significant': neuron_sig, 'neuron_id': neuron_id, 'allen_acronym': clusters['acronym'][n],
            'subject': ses_info['subject'], 'date': ses_info['date'], 'eid': eid, 'pid': pid})))
        
    # Save output
    neuron_df.to_csv(join(path_dict['save_path'], 'neuron_modulation.csv'), index=False)

    