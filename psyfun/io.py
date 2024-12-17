import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings

from one.alf.exceptions import *
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas

def load_metadata():
    """
    Loads recording metadata .csv as a pandas DataFrame. Converts date column 
    to a datetime object and administration time to seconds.
    """
    df_meta = pd.read_csv('./metadata/metadata.csv')
    df_meta['date'] = df_meta['date'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y').date())
    hms2sec = lambda hms: np.sum(np.array([int(val) for val in hms.split(':')]) * np.array([3600, 60, 1]))
    df_meta['administration_time'] = df_meta['administration_time'].apply(hms2sec)
    return df_meta

def query_recordings(one, qc='50', aligned=True, save=True):
    """
    Query alyx database for recordings (pids) from the psychedelics project.

    Parameters
    ----------
    one : one.api.OneAlyx
        Alyx database connection instance.
    qc : str
        Number specifying passing quality control level.
    aligned : bool
        If True, will only fetch recording sessions where anatomical alignment
        has been performed.
    save : bool
        If True, dataframe with recording infor will be saved as csv.

    Returns
    -------
    df_recordings : pd.DataFrame
        Data frame of information for each probe insertion matching the query.
    """

    django_str = f'session__qc__lt,{qc},'
    if aligned:
        django_str += 'json__extended_qc__tracing_exists,True'
    pids, infos = one.search_insertions(project='psychedelics', django=django_str, details=True)

    name_map = {'id': 'pid', 'session': 'eid', 'probe': 'name'}
    df_recordings = pd.DataFrame(infos).rename(columns=name_map)
    df_recordings = df_recordings.set_index('pid')
    df_recordings['subject'] = df_recordings['session_info'].apply(lambda x: x['subject']) 
    df_recordings['date'] = df_recordings['session_info'].apply(lambda x: datetime.fromisoformat(x['start_time']).date().isoformat())
    
    if save:
        df_recordings.to_csv('metadata/recordings.csv')
        
    return df_recordings

def load_recordings():
    return pd.read_csv('metadata/recordings.csv', index_col='pid')

def fetch_protocol_timings(one, eid, administration_time=True, save=True):
    """
    Get timings of protocol events throughout the recording sesison.
    """
    timingss = []

    if administration_time:
        df_meta = load_metadata()
    else:
        df_meta = None

    details = one.get_details(eid)
    
    # Assert that there were two tasks run throughout the session
    protocols = details['task_protocol'].split('/')
    if len(protocols) < 2:   # TODO: check for a more direct method
        warnings.warn(f"Fewer than 2 protocols found for {eid}")
    # Assert that both protocols are identical
    assert len(np.unique(protocols)) == 1

    timings = {'eid': eid}
    for i, protocol in enumerate(protocols):
         

        # try:
            # collection = f'alf/task_0{i}'
            # df = one.load_dataset(eid, '_ibl_passivePeriods.intervalsTable', collection).set_index('Unnamed: 0').rename_axis('')
            # spontaneous_start = df.loc['start', 'spontaneousActivity']
            # spontaneous_stop = df.loc['stop', 'spontaneousActivity']
            # rfm_start = df.loc['start', 'RFM']
            # rfm_stop = df.loc['stop', 'RFM']
            # replay_start = df.loc['stop', 'taskReplay']
            # # Note: in intervals table, the protocol doesn't end until a new
            # # protocol is started or the recording is stopped, so we 
            # # define task replay stop as the last garbor/valve event
            # gabors = one.load_dataset(eid, '_ibl_passiveGabor.table', collection)
            # stims = one.load_dataset(eid, '_ibl_passiveStims.table', collection)
            # replay_stop = np.max([gabors['stop'].max(), stims.max().max()])
        # except ALFObjectNotFound:
        warnings.warn(
            f"No ALF data found for {eid}, task {i:02d} "
            "Reverting to raw task data"
        )
        collection = f'raw_task_data_0{i}'
        task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json', collection)
        spontaneous_start = datetime.fromisoformat(task_settings['SESSION_DATETIME'])
        df_gabor = one.load_dataset(eid, '_iblrig_stimPositionScreen.raw.csv', collection)
        # First stimulus becomes the header, so we need to pull it out
        df_gabor = pd.concat([pd.DataFrame([df_gabor.columns], columns=df_gabor.columns), df_gabor], ignore_index=True)
        datetime_str = df_gabor.iloc[0, 2]
        main, decimals = datetime_str.split('.')
        decimals = decimals[:6]  # Keep only up to 6 digits
        datetime_str = f"{main}.{decimals}"
        replay_start = datetime.fromisoformat(datetime_str)
        datetime_str = df_gabor.iloc[-1, 2]
        main, decimals = datetime_str.split('.')
        decimals = decimals[:6]  # Keep only up to 6 digits
        datetime_str = f"{main}.{decimals}"
        replay_stop = datetime.fromisoformat(datetime_str)
        
        # Convert datetimes to seconds since session start
        session_details = one.get_details(eid)
        session_start = datetime.fromisoformat(session_details['start_time'])
        spontaneous_start = (spontaneous_start - session_start).seconds            
        replay_start = (replay_start - session_start).seconds
        replay_stop = (replay_stop - session_start).seconds
        # Fill missing values with NaN
        spontaneous_stop = np.nan
        rfm_start = rfm_stop = np.nan

        # Insert everything into timing dict
        timings[f'spontaneous_start_{i:02d}'] = spontaneous_start
        timings[f'spontaneous_stop_{i:02d}'] = spontaneous_stop
        timings[f'rfm_start_{i:02d}'] = rfm_start
        timings[f'rfm_stop_{i:02d}'] = rfm_stop
        timings[f'replay_start_{i:02d}'] = replay_start
        timings[f'replay_stop_{i:02d}'] = replay_stop
        
    if df_meta is not None:
        meta = df_meta[(df_meta['animal_ID'] == details['subject']) & (df_meta['date'] == details['date'])]
        if len(meta) < 1:
            warnings.warn(f"No entries in 'recordings.csv' for {eid}")
            return pd.DataFrame([timings]).set_index('eid')
        elif len(meta) > 1:
            warnings.warn(f"More than one entry in 'recordings.csv' for {eid}")
            return pd.DataFrame([timings]).set_index('eid')
        timings['admin_time'] = meta['administration_time'].values[0]
        
    return pd.DataFrame([timings]).set_index('eid')

def load_timings():
    return pd.read_csv('metadata/timings.csv', index_col='eid')

def fetch_spikes(one, pid):
    # Load in spike times and cluster info
    loader = SpikeSortingLoader(pid=pid, one=one, atlas=AllenAtlas(res_um=50))
    spikes, clusters, channels = loader.load_spike_sorting()
    # Merge QC metrics into clusters dict
    clusters = loader.merge_clusters(spikes, clusters, channels)
    if clusters is None:
        return 
    clusters['uuids'] = clusters['uuids'].values  # take values out of dataframe
    # Take only "good clusters"
    good_cluster_mask = clusters['label'] == 1  # TODO: find out what other label values mean
    good_clusters = {key:val[good_cluster_mask] for key, val in clusters.items()}
    if not good_cluster_mask.any():
        return 
    # Unpack dict of arrays into list of dicts
    cluster_infos = [{key:val[i] for key, val in good_clusters.items()} for i, cid in enumerate(good_clusters['cluster_id'])]
    print(f"Good clusters for PID {pid[:8]}...: {good_cluster_mask.sum()} / {len(good_cluster_mask)}")
    # Build dataframe from list for this probe
    df_probe = pd.DataFrame(cluster_infos)
    df_probe['pid'] = pid
    df_probe['eid'], df_probe['probe'] = one.pid2eid(pid)
    df_probe['spike_times'] = df_probe['cluster_id'].apply(lambda x: spikes.times[spikes.clusters == x])
    return df_probe.set_index('uuids')

def load_spikes():
   return pd.read_pickle('data/spike_times.pkl')