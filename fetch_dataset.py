import pandas as pd
from tqdm import tqdm
from one.api import ONE

import psyfun.io as io

one = ONE()

print("Querying database...")
df_recordings = io.query_recordings(one)
pids = df_recordings.index
eids = df_recordings['eid'].unique()

print("Fetching protocol timings...")
df_timings = pd.DataFrame()
for eid in tqdm(eids):
    df_timings = pd.concat([df_timings, io.fetch_protocol_timings(one, eid)])
df_timings.to_csv('metadata/timings.csv')

print("Fetching spike sorting data...")
df_spikes = pd.DataFrame()
for pid in tqdm(pids):
    df_spikes = pd.concat([df_spikes, io.fetch_spikes(one, pid)])
df_spikes.to_pickle('data/spike_times.pkl')


