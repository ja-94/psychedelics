# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:50:22 2022

@author: Guido
"""

import json
import h5py
import numpy as np
import pandas as pd
from os.path import join, dirname, realpath, isfile


def paths():
    """
    Load in figure path from paths.json, if this file does not exist it will be generated from
    user input
    """
    if not isfile(join(dirname(realpath(__file__)), 'paths.json')):
        paths = dict()
        paths['fig_path'] = input('Path folder to save figures: ')
        paths['data_path'] = input('Path to data folder:')
        path_file = open(join(dirname(realpath(__file__)), 'paths.json'), 'w')
        json.dump(paths, path_file)
        path_file.close()
    with open(join(dirname(realpath(__file__)), 'paths.json')) as json_file:
        paths = json.load(json_file)
    paths['repo_path'] = dirname(realpath(__file__))
    return paths


def load_subjects():
    path_dict = paths()
    subjects = pd.read_csv(join(path_dict['repo_path'], 'subjects.csv'), delimiter=';')
    return subjects


def load_tracking(file_path):
    
    # Load in SLEAP tracking
    with h5py.File(file_path, 'r') as f:
        node_names = f['node_names'][:]
        tracks_array = f['tracks'][:]
        
    # Reformat weird data to python formats
    tracks_array = np.transpose(np.squeeze(tracks_array))
    node_names = [str(i)[2:-1] for i in node_names]
    
    # Create dictonary with data
    tracking = dict()
    tracking['tracks'] = tracks_array
    tracking['node_names'] = node_names
    
    return tracking
