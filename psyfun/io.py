import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import warnings

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

    timings = {}
    timings['eid'] = eid
    dataset = '_ibl_passivePeriods.intervalsTable'
    for i, protocol in enumerate(protocols):
        collection = f'alf/task_0{i}' 

        try:
            df = one.load_dataset(eid, dataset, collection).set_index('Unnamed: 0').rename_axis('')
            # return pd.DataFrame([timings]).set_index('eid')
            # Assert all protocols are in expected order
            assert (np.diff(df.loc['start']) > 0).all()
            timings[f'spontaneous_start_{i:02d}'] = df.loc['start', 'spontaneousActivity']
            timings[f'spontaneous_stop_{i:02d}'] = df.loc['stop', 'spontaneousActivity']
            timings[f'rfm_start_{i:02d}'] = df.loc['start', 'RFM']
            timings[f'rfm_stop_{i:02d}'] = df.loc['stop', 'RFM']
            timings[f'replay_start_{i:02d}'] = df.loc['stop', 'taskReplay']
        except:
            warnings.warn(f"No intervals table found for {eid}, task {i:02d}")
        # Define task stop as the last garbor/valve event
        try:
            gabors = one.load_dataset(eid, '_ibl_passiveGabor.table', collection)
            stims = one.load_dataset(eid, '_ibl_passiveStims.table', collection)
            timings[f'replay_stop_{i:02d}'] = np.max([gabors['stop'].max(), stims.max().max()])
        except:
            warnings.warn(f"No stimulus timing found for {eid}, task {i:02d}")
            # timings[f'replay_stop_{i:02d}'] = np.nan
            # return pd.DataFrame([timings]).set_index('eid')
        
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