import numpy as np
import pandas as pd
from tqdm import tqdm
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader

from one.api import ONE

one = ONE()

df_pids = pd.read_csv('metadata/rec.csv', index_col='Unnamed: 0')

df_spikes = pd.DataFrame()
for pid in tqdm(df_pids['pid'].values):
    df_spikes = pd.concat([df_spikes, fetch_spikes(pid)])
df_spikes.to_pickle('data/spike_times.pkl')