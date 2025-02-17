from one.api import ONE
import pandas as pd
from pathlib import Path
one = ONE(base_url='https://alyx.internationalbrainlab.org', cache_dir='/mnt/s0/psychedelics/one_cache')
sessions = one.alyx.rest('sessions', 'list', projects='psychedelics')

path_psychedelics = Path('/mnt/s0/psychedelics')
# we have 64 sessions
len(sessions)


def _rename_protocol(task_protocol):
    if 'habituation' in task_protocol:
        return 'habituation'
    elif 'passive' in task_protocol:
        return 'passive'
    elif 'Histology' in task_protocol:
        return 'histology'


df_sessions = pd.DataFrame(sessions)
df_sessions['task_protocol'] = df_sessions['task_protocol'].apply(_rename_protocol)
len(df_sessions['subject'].unique())
df_recap = df_sessions.pivot_table(index='subject', columns='task_protocol', values='id', aggfunc='count')

dsets = one.alyx.rest('datasets', 'list', django='session__projects__name,psychedelics')
siz = [dset['file_size'] for dset in dsets]

"""
so we have 12 subjects, all with histology
we have 74 sessions
- 12 histology sessions
- 33 habituation (2/3 per subject)
- 29 passive independent (2/3 per subject)
- 2.5 Tb total data size (4k+ dataset files)
"""

# %% Create the 'insertions.pqt' file
from brainbox.io.one import SpikeSortingLoader
import pandas as pd

df_sessions = pd.DataFrame(one.alyx.rest('sessions', 'list', projects='psychedelics', task_protocol='passive'))
df_sessions = df_sessions.set_index('id')
df_insertions = []
for eid, session in df_sessions.iterrows():
    pids, pnames = one.eid2pid(eid)
    for pid, pname in zip(pids, pnames):
        df_insertions.append({'pid': str(pid), 'pname': pname, 'eid': eid, 'subject': session['subject'], 'start_time': session['start_time'], 'number': session['number']})
df_insertions = pd.DataFrame(df_insertions).set_index('pid')
df_insertions.to_parquet(path_psychedelics / 'insertions.pqt')



