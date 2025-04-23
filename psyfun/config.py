import numpy as np
import matplotlib as mpl

ibl_project = {
    'name': 'psychedelics',
    'protocol': 'passiveChoiceWorld'
}

paths = {
    'metadata': 'metadata/metadata.csv',
    'sessions': 'metadata/sessions.pqt',
    'insertions': 'metadata/insertions.pqt',
    'units': 'data/units.pqt',
    'spikes': 'data/spikes.h5',
    'BWM_insertions': 'metadata/BWM_insertions.pqt',
    'BWM_units': 'data/BWM_units.pqt',
    'BWM_spikes': 'data/BWM_spikes.h5',
}

qc_datasets = {
    'task00': [
        'raw_task_data_00/_iblrig_taskSettings.raw.json',
        'alf/task_00/_ibl_passivePeriods.intervalsTable.csv',
        ],
    'task01': [
        'raw_task_data_01/_iblrig_taskSettings.raw.json',
        'alf/task_01/_ibl_passivePeriods.intervalsTable.csv',
        ],
    'probe00': [
        'raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.sync.npy',
        'raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin',
        'alf/probe00/pykilosort/spikes.times.npy'
        ],
    'probe01': [
        'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.sync.npy',
        'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin',
        'alf/probe01/pykilosort/spikes.times.npy'
        ],
    'video': [
        'raw_video_data/_iblrig_bodyCamera.raw.mp4',
        'raw_video_data/_iblrig_bodyCamera.frameData.bin',
        'raw_video_data/_iblrig_leftCamera.raw.mp4',
        'raw_video_data/_iblrig_leftCamera.frameData.bin',
        'raw_video_data/_iblrig_rightCamera.raw.mp4',
        'raw_video_data/_iblrig_rightCamera.frameData.bin',
        ]
}

epoch_length = 5 * 60  # seconds
postLSD_epochs = np.arange(0, 30, 10) * 60  # seconds

cmaps = {
    'n_units': mpl.colormaps['RdPu'],
    'LSD': mpl.colors.LinearSegmentedColormap.from_list("LSDcmap", ['gray', 'darkorchid'])
}

ap_coords = [2, 1, 0, -1, -2, -3, -4]  # mm
