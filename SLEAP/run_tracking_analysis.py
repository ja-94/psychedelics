# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:05:50 2022

@author: Guido
"""

import numpy as np
from os.path import join
from psychedelic_functions import paths, load_tracking
import matplotlib.pyplot as plt
from glob import glob

# Settings
DOSAGES = ['Low_Dose']
NODE = 'nose'

# Get paths
path_dict = paths()
data_dir = join(path_dict['data_path'], 'OpenField', 'Tracking')

# Loop over different dosages
for i, this_dose in enumerate(DOSAGES):
    # Get list of subjects
    sub_dirs = glob(join(data_dir, this_dose, '*'))
    # Loop over subjects
    for j, this_sub in enumerate(sub_dirs):
        # Get list of sessions
        ses_paths = glob(join(this_sub, '*.h5'))
        # Loop over sessions
        for k, this_ses in enumerate(ses_paths):
            
            # Load in tracking data of this session
            tracking = load_tracking(this_ses)
            
            # Get index to node to use
            node_ind = [i for i, node in enumerate(tracking['node_names']) if NODE in node][0]
            
            
            
            
        

