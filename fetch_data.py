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

# Query the database for all sessions associated with this project (metadata)
# df_sessions = io.fetch_sessions(one, save=True)

# Query the database for all probe insertions associated with this project (metadata)
# df_insertions = io.fetch_insertions(one, save=False)

# Load session and insertion info from file if already downloaded
# df_sessions = pd.read_csv(paths['sessions'])
# df_insertions = pd.read_csv(paths['insertions'])

# Choose to save unit info
# uinfo_file = paths['units']  # download spike times and save to file
# Choose to save spike times as well as cluster info
# spike_file = paths['spikes']  # download spike times and save to file
# Download cluster info and spike times from server (if spike_file is empty no spikes are downloaded, 
# df_units has no spike times because they are heavy so we are doing loading on demand for spikes)
# df_units = io.fetch_unit_info(one, df_insertions, uinfo_file=uinfo_file, spike_file=spike_file)

# Download task start times for all BWM data
# df_bwm = io.fetch_BWM_task_starts(one)
# Or load task start times from saved file
# df_bwm = pd.read_csv('metadata/BWM_task_starts.csv')
# Apply cutoff to select ones we can use as controls
# df_controls = df_bwm.query('task_start > @cutoff')

# Fetch probe insertion info for these control recordings
# df_insertions_bwm = io.fetch_BWM_insertions(one, df_controls)
# Or load task start times from saved file
df_insertions_bwm = pd.read_csv(paths['BWM_insertions'])
df_uinfo = io.fetch_unit_info(one, df_insertions_bwm, uinfo_file=paths['BWM_units'], spike_file=paths['BWM_spikes'])