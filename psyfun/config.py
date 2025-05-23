import numpy as np
import pandas as pd
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

control_recordings = [
    ['ZFM-08631', '2025-03-21', 'cdc1965e-4324-422a-a9d4-86b4e5d0de92'],
    ['ZFM-08584', '2025-03-19','5c28961e-1183-43f9-86a8-9c1c9d8ed743'],
    ['ZFM-08457', '2025-03-18', '878b86b7-9245-40a3-843e-9ebf0a8251db'],
    ['ZFM-08631', '2025-03-18', '58c61f29-d4e6-4ca7-b6de-bd860e83fe4e'],
    ['ZFM-08458', '2025-03-14', '55f3644c-3f86-423e-9beb-6604b5aa4e2c'],
    ['ZFM-08631', '2025-03-12', 'bb0e0ad2-4c98-4c98-b92f-559683e8a6f3'],
    ['ZFM-08584', '2025-03-12', 'e8128c55-b322-438a-9498-edd47ae4b794'],
    ['ZFM-08457', '2025-03-11', 'c7cf8e25-1e2c-4b03-a5f5-5a049f1cd228'],
    ['ZFM-08458', '2025-03-11', '3e9d9490-7fb2-4aa3-b8cd-83f216ad8cde']
]
df_controls = pd.DataFrame(data=control_recordings, columns=['subject', 'date', 'eid'])

bwm_pretask_length = 5 * 60  # seconds 

epoch_length = 5 * 60  # seconds
postLSD_epochs = np.arange(0, 30, 10) * 60  # seconds

cmaps = {
    'n_units': mpl.colormaps['RdPu'],
    'LSD': mpl.colors.LinearSegmentedColormap.from_list("LSDcmap", ['gray', 'darkorchid'])
}

ap_coords = [2, 1, 0, -1, -2, -3, -4]  # mm
