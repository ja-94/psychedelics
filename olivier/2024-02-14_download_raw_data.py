# %%
#source /home/olivier/PycharmProjects/ibl-task-forces/.venv/bin/activate
from pathlib import Path
import tqdm

import numpy as np
import pandas as pd
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir='/mnt/s0/psychedelics/one_cache')
path_psychedelics = Path('/mnt/s0/psychedelics')

df_insertions = pd.read_parquet(path_psychedelics / 'insertions.pqt')
for pid, rec in tqdm.tqdm(df_insertions.iterrows()):
    ssl = SpikeSortingLoader(pid=pid, one=one)
    raw_ap = ssl.raw_electrophysiology(band='ap', stream=False)
    raw_lf = ssl.raw_electrophysiology(band='lf', stream=False)

