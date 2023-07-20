# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:53:26 2023

@author: Guido
"""

import os
from os.path import join
from glob import glob

PATH = 'path/to/videos'

video_paths = glob(join(PATH, '.avi'))
for i, video_file in enumerate(video_paths):
    print(f'Compressing video {i} of {len(video_paths)}')
    os.system(f'ffmpeg -i {video_file} -c:v libx264 -crf 21 {video_file[:-4]}.mp4')