import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import matplotlib as mpl
from matplotlib import pyplot as plt

from one.api import ONE
from psyfun import io, plots, atlas
from psyfun.config import paths, qc_datasets, cmaps

# Instantiate database connection
one = ONE()

# Query the database for all sessions associated with this project
df_sessions = io.fetch_sessions(one, save=True)

# Query the database for all probe insertions associated with this project
df_insertions = io.fetch_insertions(one, save=False)

# Load session and insertion info from file if already downloaded
# df_sessions = pd.read_csv(paths['sessions'])
# df_insertions = pd.read_csv(paths['insertions'])

# Choose to save unit info
uinfo_file = paths['spikes']  # download spike times and save to file
# Choose to save spike times as well as cluster info
spike_file = paths['spikes']  # download spike times and save to file
# Download cluster info and spike times from server
df_units = io.fetch_unit_info(one, df_insertions, uinfo_file=uinfo_file, spike_file=spike_file)