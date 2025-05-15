import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()
import warnings
import h5py

from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
atlas = AllenAtlas()

from .config import *

def fetch_sessions(one, save=True, qc=False):
    """
    Query Alyx for sessions tagged in the psychedelics project and add session
    info to a dataframe. Sessions are restricted to those with the 
    passiveChoiceWorld task protocol, quality control metadata is unpacked, and
    a list of key datasets is checked. Sessions are sorted and labelled
    (session_n) by their order.

    Parameters
    ----------
    one : one.api.OneAlyx
        Alyx database connection instance
    save : bool
        If true, the dataframe will be saved in ./metadata/sessions.csv

    Returns
    -------
    df_sessions : pandas.core.frame.DataFrame
        Dataframe containing detailed info for each session returned by the
        query
    """
    # Query for all sessions in the project with the specified task
    sessions = one.alyx.rest('sessions', 'list', project=ibl_project['name'], task_protocol=ibl_project['protocol'])
    df_sessions = pd.DataFrame(sessions).rename(columns={'id': 'eid'})
    df_sessions.drop(columns='projects')
    if qc:
        # Unpack the extended qc from the session dict into dataframe columns
        df_sessions = df_sessions.progress_apply(_unpack_session_dict, one=one, axis='columns')
    # Check if important datasets are present for the session
    df_sessions['n_probes'] = df_sessions.apply(lambda x: len(one.eid2pid(x['eid'])[0]), axis='columns')
    df_sessions['n_tasks'] = df_sessions.apply(lambda x: len(x['task_protocol'].split('/')), axis='columns')
    df_sessions['tasks'] = df_sessions.apply(lambda x: x['task_protocol'].split('/'), axis='columns')
    df_sessions = df_sessions.progress_apply(_check_datasets, one=one, axis='columns')
    # Add label for control sessions
    df_sessions['control_recording'] = df_sessions.apply(_label_controls, axis='columns')
    df_sessions['new_recording'] = df_sessions['start_time'].apply(lambda x: datetime.fromisoformat(x) > datetime(2025, 1, 1))
    # Fetch task protocol timings and add to dataframe
    df_sessions = df_sessions.progress_apply(_fetch_protocol_timings, one=one, axis='columns')
    # Add LSD administration time
    df_meta = load_metadata()
    df_sessions = df_sessions.progress_apply(_fetch_LSD_admin_time, df_metadata=df_meta, axis='columns')
    df_sessions = _label_LSD_or_control(df_sessions, one = one)

    # Label and sort by session number for each subject
    df_sessions['session_n'] = df_sessions.groupby('subject')['start_time'].rank(method='dense').astype(int)
    df_sessions = df_sessions.sort_values(by=['subject', 'start_time']).reset_index(drop=True)
    # Save as csv
    if save:
        df_sessions.to_parquet(paths['sessions'], index=False)
    return df_sessions

def _label_controls (session, controls=df_controls):
    eid = session['eid']
    control_session = controls.query('eid == @eid')
    if len(control_session) == 1:
        return True
    elif len(control_session) == 0:
        return False
    elif len(control_sesison) > 1:
        raise ValueError("More than one entry in df_controls!")

def _unpack_session_dict(series, one=None):
    """
    Unpack the extended QC from the session dict for a given eid.
    """
    assert one is not None
    # Fetch full session dict
    session_dict = one.alyx.rest('sessions', 'read', id=series['eid'])
    series['session_qc'] = session_dict['qc']  # aggregate session QC value
    # Skip if there is no extended QC present
    if session_dict['extended_qc'] is None:
        return series
    # Add QC vals to series
    for key, val in session_dict['extended_qc'].items():
        key = key.lstrip('_')
        # Add _qc flag to any keys that don't have it 
        if not key.endswith('_qc'): key += '_qc'
        if type(val) == str:
           series[key] = val 
        elif type(val) == list:  # lists have QC outcome as first entry
            series[key] = val[0]
            # Add video framerate
            if 'framerate' in key:
                series[key.rstrip('_qc')] = val[1]
            # Add number of dropped frames
            if 'dropped_frames' in key:
                series[key.rstrip('_qc')] = val[1]
    return series


def _check_datasets(series, one=None):
    """
    Create a boolean entry for each important dataset for the given eid.
    """
    assert one is not None
    # Fetch list of datasets listed under the given eid
    datasets = one.list_datasets(series['eid'])
    # Check each task in the recording
    for n, protocol in enumerate(series['tasks']):
        # TODO: handle spontaneous protocol in 2025 recordings!
        if 'spontaneous' in protocol: 
            continue
        task_datasets = [
            f'raw_task_data_{n:02d}/_iblrig_taskSettings.raw.json',
            f'alf/task_{n:02d}/_ibl_passivePeriods.intervalsTable.csv',
            ]
        for dataset in task_datasets:
            series[dataset] = dataset in datasets
    for n in range(series['n_probes']):
        ephys_datasets = [
            f'raw_ephys_data/probe{n:02d}/_spikeglx_ephysData_g0_t0.imec{n:02d}.sync.npy',
            f'raw_ephys_data/probe{n:02d}/_spikeglx_ephysData_g0_t0.imec{n:02d}.ap.cbin',
            f'alf/probe{n:02d}/pykilosort/spikes.times.npy'
            ]
        for dataset in ephys_datasets:
            series[dataset] = dataset in datasets  
    # Check if each important dataset is present
    for dataset in qc_datasets['video']:
        series[dataset] = dataset in datasets
    return series


def fetch_insertions(one, save=True):
    """
    Query Alyx for probe insertions tagged in the psychedelics project and
    collect insertion info in a dataframe.

    Parameters
    ----------
    one : one.api.OneAlyx
        Alyx database connection instance
    save : bool
        If true, the dataframe will be saved in ./metadata/sessions.csv

    Returns
    -------
    df_insertions : pandas.core.frame.DataFrame
        Dataframe containing detailed info for each probe insertion returned by
        the query
    """
    # Query for all probe insertions in the project
    insertions = one.alyx.rest('insertions', 'list', project=ibl_project['name'])
    df_insertions = pd.DataFrame(insertions).rename(columns={'id': 'pid', 'session': 'eid', 'name':'probe'})
    # Pull out some basic fields from the session info dict
    df_insertions = df_insertions.progress_apply(_unpack_session_info, axis='columns')
    # Unpack detailed QC info from the json
    df_insertions = df_insertions.progress_apply(_unpack_json, axis='columns')
    # Add any histology QC info present
    df_insertions = df_insertions.progress_apply(_check_histology, one=one, axis='columns')
    # Label and sort by session number for each subject
    df_insertions['session_n'] = df_insertions.groupby('subject')['start_time'].rank(method='dense').astype(int)
    df_insertions = df_insertions.sort_values(by=['subject', 'start_time']).reset_index(drop=True)
    # Save as csv
    if save:
        df_insertions.to_parquet(paths['insertions'], index=False)
    return df_insertions


def _unpack_session_info(series):
    series['subject'] = series['session_info']['subject']
    series['start_time'] = series['session_info']['start_time']
    return series


def _unpack_json(series):
    series['ephys_qc'] = series['json']['qc']
    jsonkeys = ['n_units', 'n_units_qc_pass', 'firing_rate_median', 'firing_rate_max']
    for key in jsonkeys:
        try:
            series[key] = series['json'][key]
        except KeyError:
            series[key] = np.nan
    if 'tracing_exists' not in series['json']['extended_qc']:
        series['tracing_qc'] = 'NOT SET'
        series['alignment_qc'] = 'NOT SET'
    elif series['json']['extended_qc']['tracing_exists']:
        if 'tracing' in series['json']['extended_qc']:
            series['tracing_qc'] = series['json']['extended_qc']['tracing']
        else:
            series['tracing_qc'] = 'NOT SET'
        try:
            alignment_resolved_by = series['json']['extended_qc']['alignment_resolved_by']
            series['alignment_qc'] = series['json']['extended_qc'][alignment_resolved_by]
        except KeyError:
            series['alignment_qc'] = 'NOT SET'
    elif not series['json']['extended_qc']['tracing_exists']:
        series['tracing_qc'] = series['json']['extended_qc']['tracing']
        series['alignment_qc'] = 'NOT SET'
    return series


def _check_histology(series, one=None):
    assert one is not None
    infos = np.array(one.alyx.rest('sessions', 'list', subject=series['subject'], histology=True))
    histology_in_protocols = ['histology' in info['task_protocol'].lower() for info in infos]
    if any(histology_in_protocols):
        ## NOT SET for all...
        histology_eid = infos[histology_in_protocols][0]['id']
        hist_dict = one.alyx.rest('sessions', 'read', id=histology_eid)
        series['histology_qc'] = hist_dict['qc']
        # series['histology_qc'] = 'NOT SET'
    else:
        series['histology_qc'] = np.nan
    return series


def load_metadata():
    """
    Loads recording metadata .csv as a pandas DataFrame. Converts date column 
    to a datetime object and administration time to seconds.
    """
    df_meta = pd.read_csv(paths['metadata'])
    df_meta['date'] = df_meta['date'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y').date())
    hms2sec = lambda hms: np.sum(np.array([int(val) for val in hms.split(':')]) * np.array([3600, 60, 1]))
    df_meta['administration_time'] = df_meta['administration_time'].apply(hms2sec)
    return df_meta


def _fetch_protocol_timings(series, one=None):
    """
    Get timings of protocol events throughout the recording sesison.
    """
    assert one is not None
    session_details = one.get_details(series['eid'])
    session_start = datetime.fromisoformat(session_details['start_time'])
    task_count = 0
    for n, protocol in enumerate(series['tasks']):
        collection = f'raw_task_data_{n:02d}'
        try:
            # Get start time of spontaneous epoch
            task_settings = one.load_dataset(series['eid'], '_iblrig_taskSettings.raw.json', collection)
        except:
            print(f"WARNING: no taskSettings for {series['eid']} {collection}")
            continue
        spontaneous_start_str = task_settings.get('SESSION_DATETIME')  # try old entry name
        if spontaneous_start_str is None:
            spontaneous_start_str = task_settings.get('SESSION_START_TIME')  # try new entry name
        if spontaneous_start_str is None:
            raise KeyError("Neither 'SESSION_DATETIME' nor 'SESSION_START_TIME' found")
        spontaneous_start = datetime.fromisoformat(spontaneous_start_str)  # convert to datetime object
        # FIXME: handle spontaneous protocol in 2025 recordings more gracefully!
        if 'spontaneous' in protocol:
            # Check for one session where LSD admin was delayed by ~40min
            if series['eid'] != '4b874c49-3c0c-4f30-9b1f-74c9dbfb57c8':  
                # Assume LSD was given at start of spontaneous period
                series['LSD_admin'] = (spontaneous_start - session_start).seconds
            continue
        # Get gabor patch presentation timings for task replay epoch
        df_gabor = one.load_dataset(series['eid'], '_iblrig_stimPositionScreen.raw.csv', collection)
        # first stimulus becomes the header, so we need to pull it out
        df_gabor = pd.concat([pd.DataFrame([df_gabor.columns], columns=df_gabor.columns), df_gabor], ignore_index=True)
        # Get start time of first gabor
        datetime_str = df_gabor.iloc[0, 2]  # start time is in second column
        replay_start = _datetime_clip_decimals_to_iso(datetime_str)
        # Get start time of last gabor
        datetime_str = df_gabor.iloc[-1, 2]
        replay_stop = _datetime_clip_decimals_to_iso(datetime_str)
        # Convert datetimes to seconds since session start
        spontaneous_start = (spontaneous_start - session_start).seconds            
        replay_start = (replay_start - session_start).seconds
        replay_stop = (replay_stop - session_start).seconds
        # Fill missing values with estimates based on protocol
        spontaneous_stop = rfm_start = spontaneous_start + 5 * 60
        rfm_stop = replay_start
        # Insert everything into series object
        series[f'task{task_count:02d}_spontaneous_start'] = spontaneous_start
        series[f'task{task_count:02d}_spontaneous_stop'] = spontaneous_stop
        series[f'task{task_count:02d}_rfm_start'] = rfm_start
        series[f'task{task_count:02d}_rfm_stop'] = rfm_stop
        series[f'task{task_count:02d}_replay_start'] = replay_start
        series[f'task{task_count:02d}_replay_stop'] = replay_stop
        task_count += 1
    return series


def _datetime_clip_decimals_to_iso(datetime_str):
    main, decimals = datetime_str.split('.')
    decimals = decimals[:6]  # keep only 6 digits
    datetime_str = f"{main}.{decimals}"
    return datetime.fromisoformat(datetime_str)


def _fetch_LSD_admin_time(series, df_metadata=None):
    assert df_metadata is not None
    # Find entry in metadata file by subject and date
    session_meta = df_metadata[
        (df_metadata['animal_ID'] == series['subject']) & 
        (df_metadata['date'] == datetime.fromisoformat(series['start_time']).date())
    ]
    # Ensure only one entry is present
    if len(session_meta) < 1:
        warnings.warn(f"No entries in 'recordings.csv' for {series['eid']}")
        return series
    elif len(session_meta) > 1:
        warnings.warn(f"More than one entry in 'recordings.csv' for {series['eid']}")
        return series
    series['LSD_admin'] = session_meta['administration_time'].values[0]
    return series


def add_postLSD_epochs(df_sessions, epochs=postLSD_epochs, length=epoch_length):
    for epoch in epochs:
        df_sessions[f'LSD{epoch}_start'] = df_sessions.apply(lambda x: x['LSD_admin'] + epoch, axis='columns')
        df_sessions[f'LSD{epoch}_stop'] = df_sessions[f'LSD{epoch}_start'] + length
    return df_sessions

class PsySpikeSortingLoader(SpikeSortingLoader):

    def merge_clusters(self, clusters, channels, compute_metrics=False, spikes=None):
        """
        A simplified method for merging metrics and channel info into the
        clusters dict. Does not require spikes to save memory, can be
        optionally given together with compute_metrics=True to re-compute
        quality control metrics on-the-fly.
        """
        nc = clusters['channels'].size
        metrics = None
        if 'metrics' in clusters:
            metrics = clusters.pop('metrics')
            if metrics.shape[0] != nc:
                metrics = None
        if metrics is None or compute_metrics is True:
            assert spikes is not None
            metrics = SpikeSortingLoader.compute_metrics(spikes, clusters)
        for k in metrics.keys():
            clusters[k] = metrics[k].to_numpy()
        for k in channels.keys():
            if k in ['localCoordinates', 'rawInd']: continue
            clusters[k] = channels[k][clusters['channels']]
        return clusters


def fetch_unit_info(one, df_insertions, uinfo_file='', spike_file='', atlas=atlas):
    probe_dfs = []
    for idx, probe in tqdm(df_insertions.iterrows(), total=len(df_insertions)):
        # Load in spike times and cluster info
        pid = probe['pid']
        loader = PsySpikeSortingLoader(pid=pid, one=one, atlas=atlas)
        try:
            clusters = loader.load_spike_sorting_object('clusters')
            channels = loader.load_channels()
        except KeyError:
            continue
        if clusters is None:
            print(f"WARNING: no clusters for {pid}")
            continue
        clusters['uuids'] = clusters['uuids'].to_numpy()  # take values out of dataframe
        clusters = loader.merge_clusters(clusters, channels)
        # Build dataframe from list for this probe
        df_probe = pd.DataFrame(clusters).rename(columns={'uuids':'uuid', 'depths':'depth', 'channels':'channel'})
        # Add additional metadata to cluster info df
        for field in ['subject', 'session_n', 'eid', 'probe', 'pid']:
            df_probe[field] = probe[field]
        df_probe['histology'] = loader.histology
        # Save spike times if a filename is given
        if spike_file:
            # Check hdf5 filename
            if not spike_file.endswith('.h5'):
                spike_file = spike_file.split('.')[0] + '.h5'
            # Load spike time for each probe collection separately to conserve memory
            for collection in loader.collections:
                spikes = one.load_object(probe['eid'], collection=collection, obj='spikes', attribute=['times', 'clusters'])
                with h5py.File(spike_file, 'a') as h5file:
                    # Store spike times for each cluster as a separate dataset in the hdf file
                    for uuid, cid in zip(df_probe['uuid'], df_probe['cluster_id']):
                        # Separate spike times for each cluster
                        spike_times = spikes['times'][spikes['clusters'] == cid]
                        # Delete existing dataset if present
                        if uuid in h5file: del h5file[uuid]
                        # Create new dataset for this unit
                        h5file.create_dataset(uuid, data=spike_times) #One single file with all the spikes, which can be accessed selectively by cluster without loading everything
                del spikes
        del clusters, loader
        # Append to list
        probe_dfs.append(df_probe)
    # Concatenate cluster info for all probes
    df_uinfo = pd.concat(probe_dfs)
    # Clean up some columns
    df_uinfo['histology'] = df_uinfo['histology'].fillna('')
    df_uinfo = df_uinfo.rename(columns={'acronym': 'region'})
    if uinfo_file:
        if not uinfo_file.endswith('.pqt'):
            uinfo_file = uinfo_file.split('.')[0] + '.pqt'
        df_uinfo.to_parquet(uinfo_file, index=False)
    return df_uinfo


def load_units(spike_file, uuids):
    #Takes in a list of unit IDs and the spike file name
    if not spike_file.endswith('.h5'):
        spike_file = spike_file.split('.')[0] + '.h5'
    units = []
    with h5py.File(spike_file, 'r') as h5file:
        for uuid in tqdm(uuids):
            unit = {
                'uuid': uuid,
                'spike_times': h5file[uuid][:]
            }
            units.append(unit)
    return pd.DataFrame(units).set_index('uuid')

def fetch_BWM_task_starts(one, save=True):
    # All BWM ephys sessions
    eids = np.array(one.search(project='brainwide', task_protocol='ephys'))
    # Collect timing of trial starts
    task_starts = np.full(len(eids), np.nan)
    for i, eid in tqdm(enumerate(eids), total=len(eids)):
        try:
            df_trials = one.load_dataset(eid, dataset='_ibl_trials.table')
            task_starts[i] = df_trials.loc[0, 'intervals_0']  # start of first trial
        except:  # if session is missing trials.table
            continue
    df = pd.DataFrame({'eid': eids, 'task_start': task_starts})
    if save:
        df.to_csv('metadata/BWM_task_starts.csv', index=False)
    return df


def fetch_BWM_insertions(one, df_controls, save=True):
    pids = np.concatenate([one.eid2pid(eid)[0] for eid in df_controls['eid']])
    insertions = [one.alyx.rest('insertions', 'list', id=pid)[0] for pid in pids]
    df_insertions = pd.DataFrame(insertions).rename(columns={'id': 'pid', 'session': 'eid', 'name':'probe'})
    # Pull out some basic fields from the session info dict
    df_insertions = df_insertions.progress_apply(_unpack_session_info, axis='columns')
    # Unpack detailed QC info from the json
    df_insertions = df_insertions.progress_apply(_unpack_json, axis='columns')
    # Add any histology QC info present
    df_insertions = df_insertions.progress_apply(_check_histology, one=one, axis='columns')
    # Label and sort by session number for each subject
    df_insertions['session_n'] = df_insertions.groupby('subject')['start_time'].rank(method='dense').astype(int)
    df_insertions = df_insertions.sort_values(by=['subject', 'start_time']).reset_index(drop=True)
    # Save as csv
    if save:
        df_insertions.to_parquet(paths['BWM_insertions'], index=False)
    return df_insertions