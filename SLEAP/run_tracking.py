#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:17:41 2022
By: Guido Meijer
"""

import os
import numpy as np
import pandas as pd
from os.path import join, isdir, split, isfile
from glob import glob
from shutil import copyfile

# Settings
EXT_DRIVE_PATH = '/media/guido/Seagate Backup Plus Drive/Experiments_Joana_Guido/OpenField_tests'
MODEL_PATH = '/home/guido/Data/SLEAP/OpenFieldCatheter/models/221205_170800.single_instance.n=135'
DATA_PATH = '/media/guido/Data2/Psychedelics/OpenField/Videos'
RESULT_PATH = '/home/guido/Dropbox/Work/Data/Psychedelics/OpenField/Tracking'

# Copy new videos from external drive
top_level_folders = glob(join(EXT_DRIVE_PATH, '*'))
for i, top_folder in enumerate(top_level_folders):
    if not isdir(join(DATA_PATH, split(top_folder)[-1])):
        os.mkdir(join(DATA_PATH, split(top_folder)[-1]))
    sub_folders = glob(join(EXT_DRIVE_PATH, top_folder, '*'))
    for j, sub_folder in enumerate(sub_folders):
        if not isdir(join(DATA_PATH, split(top_folder)[-1], split(sub_folder)[-1])):
            os.mkdir(join(DATA_PATH, split(top_folder)[-1], split(sub_folder)[-1]))

        # Copy video files
        video_files = glob(join(EXT_DRIVE_PATH, sub_folder, '*.avi'))
        for k, video_file in enumerate(video_files):
            output_path = join(DATA_PATH, split(top_folder)[-1], split(sub_folder)[-1],
                               split(video_file)[-1])
            if not isfile(output_path):
                print(f'Copying {split(video_file)[-1]} from external drive..')
                copyfile(video_file, output_path)

        # copy csv files
        csv_files = glob(join(EXT_DRIVE_PATH, sub_folder, '*.csv'))
        for k, csv_file in enumerate(csv_files):
            output_path = join(DATA_PATH, split(top_folder)[-1], split(sub_folder)[-1],
                               split(csv_file)[-1])
            if not isfile(output_path):
                print(f'Copying {split(csv_file)[-1]} from external drive..')
                copyfile(csv_file, output_path)


# Find videos to process
top_level_folders = glob(join(DATA_PATH, '*'))
for i, top_folder in enumerate(top_level_folders):
    if not isdir(join(RESULT_PATH, split(top_folder)[-1])):
        os.mkdir(join(RESULT_PATH, split(top_folder)[-1]))
    sub_folders = glob(join(DATA_PATH, top_folder, '*'))
    for j, sub_folder in enumerate(sub_folders):
        if not isdir(join(RESULT_PATH, split(top_folder)[-1], split(sub_folder)[-1])):
            os.mkdir(join(RESULT_PATH, split(top_folder)[-1], split(sub_folder)[-1]))
        video_files = glob(join(DATA_PATH, sub_folder, '*.avi'))
        for k, video_file in enumerate(video_files):
            output_path = join(RESULT_PATH, split(top_folder)[-1], split(sub_folder)[-1],
                               split(video_file)[-1][:-4])

            # Run SLEAP tracking
            if not isfile(output_path + '.slp'):
                print('Starting SLEAP tracking..')
                os.system((f'sleap-track --model {MODEL_PATH} '
                           f'--output {output_path} '
                           '--tracking.tracker flow '
                           f'{video_file}'))

            # Convert to H5 file format
            if not isfile(output_path + '.h5'):
                print('Converting output to .h5 file format..')
                os.system((f'sleap-convert -o {output_path + ".h5"} '
                           f'--format analysis '
                           f'{output_path + ".slp"}'))

            # Copy and save timestamps
            if not isfile(output_path + '_timestamps.npy'):
                meta_data = pd.read_csv(video_file[:-4] + '.csv', header=None)
                np.save(output_path + '_timestamps.npy', meta_data[1].values)

            """
            # Compress video
            if not isfile(f'{video_file[:-4]}.mp4'):
                print('Compressing video..')
                os.system(f'ffmpeg -i {video_file} -c:v libx264 -crf 21 {video_file[:-4]}.mp4')
            """

            # Delete original avi file
            # TO DO


