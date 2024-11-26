import numpy as np
import pandas as pd
from tqdm import tqdm
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader

from one.api import ONE

one = ONE()
df_pids = pd.read_csv('rec.csv', index_col='Unnamed: 0')

df_spikes = pd.DataFrame()
for pid in tqdm(df_pids['pid'].values):
    # Load in spike times and cluster info
    loader = SpikeSortingLoader(pid=pid, one=one, atlas=AllenAtlas(res_um=50))
    spikes, clusters, channels = loader.load_spike_sorting()
    # Merge QC metrics into clusters dict
    clusters = loader.merge_clusters(spikes, clusters, channels)
    if clusters is None:
        continue
    clusters['uuids'] = clusters['uuids'].values  # take values out of dataframe
    # Take only "good clusters"
    good_cluster_mask = clusters['label'] == 1  # TODO: find out what other label values mean
    if not good_cluster_mask.any():
        continue
    good_clusters = {key:val[good_cluster_mask] for key, val in clusters.items()}
    # Unpack dict of arrays into list of dicts
    cluster_infos = [{key:val[i] for key, val in good_clusters.items()} for i, cid in enumerate(good_clusters['cluster_id'])]
    # print(f"Good clusters: {good_cluster_mask.sum()} / {len(good_cluster_mask)}")
    # Build dataframe from list for this probe
    df_probe = pd.DataFrame(cluster_infos)
    df_probe['pid'] = pid
    df_probe['eid'], df_probe['probe'] = one.pid2eid(pid)
    df_probe['spike_times'] = df_probe['cluster_id'].apply(lambda x: spikes.times[spikes.clusters == x])
    # Concatenate data from this probe
    df_spikes = pd.concat([df_spikes, df_probe])
        
df_spikes.to_pickle('data/spike_times.pkl')