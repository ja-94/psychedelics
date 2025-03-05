# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:50:04 2023

@author: Guido
"""

from ibllib.qc.alignment_qc import AlignmentQC
from one.api import ONE
one = ONE(mode='refresh')

pids = list(one.search_insertions(project='psychedelics', query_type='remote'))

for i, ins_id in enumerate(pids):
    print(f'Resolving insertion {ins_id} [{i+1} of {len(pids)}]')
    traj = one.alyx.rest('trajectories', 'list', probe_insertion=ins_id, provenance='Ephys aligned histology track')
    if len(traj) == 0:
        print('No alignment found')
        continue
    alignment_keys = traj[0]['json'].keys()
    if len(alignment_keys) == 0:
        print('No alignement found')
        continue
    elif len(alignment_keys) > 1:
        print('More than one alignment found!')
        continue
    align_qc = AlignmentQC(ins_id, one=one)
    try:
        align_qc.resolve_manual(list(alignment_keys)[0])
    except Exception as err:
        print(err)
        pass