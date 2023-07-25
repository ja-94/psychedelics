# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:53:26 2023

@author: Guido
"""

import os

PATH = 'path/to/videos'

for root, directory, files in os.walk(PATH):
    for file in files:
        if file[-4:] == '.avi':
            print(f'Compressing video {file}')
            os.system(f'ffmpeg -i {os.path.join(root, file)} -c:v libx264 -crf 21 {file[:-4]}.mp4')