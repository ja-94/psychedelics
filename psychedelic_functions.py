# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:50:22 2022

@author: Guido
"""

import json
import h5py
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import tkinter as tk
from os.path import join, dirname, realpath, isfile
from scipy.interpolate import interp1d
from one.api import ONE
from iblutil.numerical import ismember
from iblatlas.atlas import BrainRegions


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
    paths['save_path'] = join(dirname(realpath(__file__)), 'Data')
    return paths


def load_subjects():
    path_dict = paths()
    subjects = pd.read_csv(join(path_dict['repo_path'], 'subjects.csv'), delimiter=';')
    return subjects


def query_recordings(aligned=True, one=None):
    if one is None:
        one = ONE()
    elif one == 'local':
        rec = pd.read_csv(join(paths()['repo_path'], 'rec.csv'))
        return rec

    # Construct django query string
    DJANGO_STR = ('session__project__name__icontains,psychedelics,'
                 'session__qc__lt,50')
    if aligned:
        # Query all ephys-histology aligned sessions
        DJANGO_STR += ',json__extended_qc__alignment_count__gt,0'

    # Query sessions
    ins = one.alyx.rest('insertions', 'list', django=DJANGO_STR)
   
    # Get list of eids and probes
    rec = pd.DataFrame()
    rec['pid'] = np.array([i['id'] for i in ins])
    rec['eid'] = np.array([i['session'] for i in ins])
    rec['probe'] = np.array([i['name'] for i in ins])
    rec['subject'] = np.array([i['session_info']['subject'] for i in ins])
    rec['date'] = np.array([i['session_info']['start_time'][:10] for i in ins])
    rec = rec.drop_duplicates('pid', ignore_index=True)
    
    # Save to file
    rec.to_csv(join(paths()['repo_path'], 'rec.csv'))
    return rec


def remap(acronyms, source='Allen', dest='Beryl', combine=False, split_thalamus=False,
          abbreviate=True, brainregions=None):
    br = brainregions or BrainRegions()
    _, inds = ismember(br.acronym2id(acronyms), br.id[br.mappings[source]])
    remapped_acronyms = br.get(br.id[br.mappings[dest][inds]])['acronym']
    return remapped_acronyms


def combine_regions(allen_acronyms, split_thalamus=False, abbreviate=False):
    acronyms = remap(allen_acronyms)
    regions = np.array(['root'] * len(acronyms), dtype=object)
    if abbreviate:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'mPFC'
        regions[np.in1d(acronyms, ['MOs'])] = 'M2'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'OFC'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'PO'
            regions[np.in1d(acronyms, ['LP'])] = 'LP'
            regions[np.in1d(acronyms, ['LD'])] = 'LD'
            regions[np.in1d(acronyms, ['RT'])] = 'RT'
            regions[np.in1d(acronyms, ['VAL'])] = 'VAL'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thal'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'SC'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'RSP'
        regions[np.in1d(acronyms, ['MRN'])] = 'MRN'
        regions[np.in1d(acronyms, ['ZI'])] = 'ZI'
        regions[np.in1d(acronyms, ['PAG'])] = 'PAG'
        regions[np.in1d(acronyms, ['SSp-bfd'])] = 'BC'
        #regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'LG'
        regions[np.in1d(acronyms, ['PIR'])] = 'Pir'
        #regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'SN'
        regions[np.in1d(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'VIS'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amyg'
        regions[np.in1d(acronyms, ['AON', 'TTd', 'DP'])] = 'OLF'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Str'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hipp'
    else:
        regions[np.in1d(acronyms, ['ILA', 'PL', 'ACAd', 'ACAv'])] = 'Medial prefrontal cortex'
        regions[np.in1d(acronyms, ['MOs'])] = 'Secondary motor cortex'
        regions[np.in1d(acronyms, ['ORBl', 'ORBm'])] = 'Orbitofrontal cortex'
        if split_thalamus:
            regions[np.in1d(acronyms, ['PO'])] = 'Thalamus (PO)'
            regions[np.in1d(acronyms, ['LP'])] = 'Thalamus (LP)'
            regions[np.in1d(acronyms, ['LD'])] = 'Thalamus (LD)'
            regions[np.in1d(acronyms, ['RT'])] = 'Thalamus (RT)'
            regions[np.in1d(acronyms, ['VAL'])] = 'Thalamus (VAL)'
        else:
            regions[np.in1d(acronyms, ['PO', 'LP', 'LD', 'RT', 'VAL'])] = 'Thalamus'
        regions[np.in1d(acronyms, ['SCm', 'SCs', 'SCig', 'SCsg', 'SCdg'])] = 'Superior colliculus'
        regions[np.in1d(acronyms, ['RSPv', 'RSPd'])] = 'Retrosplenial cortex'
        regions[np.in1d(acronyms, ['MRN'])] = 'Midbrain reticular nucleus'
        regions[np.in1d(acronyms, ['AON', 'TTd', 'DP'])] = 'Olfactory areas'
        regions[np.in1d(acronyms, ['ZI'])] = 'Zona incerta'
        regions[np.in1d(acronyms, ['PAG'])] = 'Periaqueductal gray'
        regions[np.in1d(acronyms, ['SSp-bfd'])] = 'Barrel cortex'
        #regions[np.in1d(acronyms, ['LGv', 'LGd'])] = 'Lateral geniculate'
        regions[np.in1d(acronyms, ['PIR'])] = 'Piriform'
        #regions[np.in1d(acronyms, ['SNr', 'SNc', 'SNl'])] = 'Substantia nigra'
        regions[np.in1d(acronyms, ['VISa', 'VISam', 'VISp', 'VISpm'])] = 'Visual cortex'
        regions[np.in1d(acronyms, ['MEA', 'CEA', 'BLA', 'COAa'])] = 'Amygdala'
        regions[np.in1d(acronyms, ['CP', 'STR', 'STRd', 'STRv'])] = 'Tail of the striatum'
        regions[np.in1d(acronyms, ['CA1', 'CA3', 'DG'])] = 'Hippocampus'
    return regions


def high_level_regions(acronyms, merge_cortex=False):
    """
    Input Allen atlas acronyms
    """
    first_level_regions = combine_regions(remap(acronyms), abbreviate=True)
    cosmos_regions = remap(acronyms, dest='Cosmos')
    regions = np.array(['root'] * len(first_level_regions), dtype=object)
    if merge_cortex:
        regions[cosmos_regions == 'Isocortex'] = 'Cortex'
        regions[first_level_regions == 'Pir'] = 'Cortex'
    else:
        regions[np.in1d(first_level_regions, ['mPFC', 'OFC', 'M2'])] = 'Frontal'
        regions[np.in1d(first_level_regions, ['Pir', 'BC', 'VISa/am'])] = 'Sensory'
    regions[cosmos_regions == 'MB'] = 'Midbrain'
    regions[cosmos_regions == 'HPF'] = 'Hippocampus'
    regions[cosmos_regions == 'TH'] = 'Thalamus'
    regions[np.in1d(first_level_regions, ['Amyg'])] = 'Amygdala'
    regions[np.in1d(acronyms, ['CP', 'ACB', 'FS'])] = 'Striatum'
    return regions


def figure_style():
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="Arial",
            rc={"font.size": 7,
                "figure.titlesize": 7,
                "axes.titlesize": 7,
                "axes.labelsize": 7,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 3,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5,
                'legend.fontsize': 7,
                'legend.title_fontsize': 7,
                'legend.frameon': False,
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    colors = {}

    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 10
    return colors, dpi


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


def smooth_interpolate_signal_sg(signal, window=31, order=3, interp_kind='cubic'):
    """Run savitzy-golay filter on signal, interpolate through nan points.

    Parameters
    ----------
    signal : np.ndarray
        original noisy signal of shape (t,), may contain nans
    window : int
        window of polynomial fit for savitzy-golay filter
    order : int
        order of polynomial for savitzy-golay filter
    interp_kind : str
        type of interpolation for nans, e.g. 'linear', 'quadratic', 'cubic'
    Returns
    -------
    np.array
        smoothed, interpolated signal for each time point, shape (t,)

    """

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')

    signal = interpolater(timestamps)

    return signal


def non_uniform_savgol(x, y, window, polynom):
    """Applies a Savitzky-Golay filter to y with non-uniform spacing as defined in x.
    This is based on
    https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do
    https://dsp.stackexchange.com/a/64313
    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size
    Returns
    -------
    np.array
        The smoothed y values
    """

    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed

 
 
def get_dlc_XYs(one, eid, view='left', likelihood_thresh=0.9): 
    if f'alf/_ibl_{view}Camera.times.npy' in one.list_datasets(eid): 
        times = one.load_dataset(eid, f'_ibl_{view}Camera.times.npy') 
    elif isfile(join(one.eid2path(eid), f'{view}Camera.times.npy')): 
        times = np.load(join(one.eid2path(eid), f'{view}Camera.times.npy')) 
    else: 
        print('could not load camera timestamps') 
        return None, None 
    try: 
        cam = one.load_dataset(eid, '_ibl_%sCamera.dlc.pqt' % view) 
    except KeyError: 
        print('not all dlc data available') 
        return None, None 
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()]) 
    # Set values to nan if likelyhood is too low # for pqt: .to_numpy() 
    XYs = {} 
    for point in points: 
        x = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_x']) 
        x = x.filled(np.nan) 
        y = np.ma.masked_where(cam[point + '_likelihood'] < likelihood_thresh, cam[point + '_y']) 
        y = y.filled(np.nan) 
        XYs[point] = np.array([x, y]).T 
    return times, XYs 
 
 
def get_pupil_diameter(XYs): 
    """Estimate pupil diameter by taking median of different computations. 
 
    In the two most obvious ways: 
    d1 = top - bottom, d2 = left - right 
 
    In addition, assume the pupil is a circle and estimate diameter from other pairs of 
    points 
 
    Author: Michael Schartner 
 
    Parameters 
    ---------- 
    XYs : dict 
        keys should include `pupil_top_r`, `pupil_bottom_r`, 
        `pupil_left_r`, `pupil_right_r` 
    Returns 
    ------- 
    np.array 
        pupil diameter estimate for each time point, shape (n_frames,) 
 
    """ 
 
    # direct diameters 
    t = XYs['pupil_top_r'][:, :2] 
    b = XYs['pupil_bottom_r'][:, :2] 
    l = XYs['pupil_left_r'][:, :2] 
    r = XYs['pupil_right_r'][:, :2] 
 
    def distance(p1, p2): 
        return ((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2) ** 0.5 
 
    # get diameter via top-bottom and left-right 
    ds = [] 
    ds.append(distance(t, b)) 
    ds.append(distance(l, r)) 
 
    def dia_via_circle(p1, p2): 
        # only valid for non-crossing edges 
        u = distance(p1, p2) 
        return u * (2 ** 0.5) 
 
    # estimate diameter via circle assumption 
    for side in [[t, l], [t, r], [b, l], [b, r]]: 
        ds.append(dia_via_circle(side[0], side[1])) 
    diam = np.nanmedian(ds, axis=0) 
    return diam 
 
 
def get_raw_smooth_pupil_diameter(XYs): 
 
    # threshold (in standard deviations) beyond which a point is labeled as an outlier 
    std_thresh = 5 
 
    # threshold (in seconds) above which we will not interpolate nans, but keep them 
    # (for long stretches interpolation may not be appropriate) 
    nan_thresh = 1 
 
    # compute framerate of camera 
    fr = 60  # set by hardware 
    window = 61  # works well empirically 
 
    # compute diameter using raw values of 4 markers (will be noisy and have missing data) 
    diam0 = get_pupil_diameter(XYs) 
 
    # run savitzy-golay filter on non-nan timepoints to denoise 
    diam_sm0 = smooth_interpolate_signal_sg( 
        diam0, window=window, order=3, interp_kind='linear') 
 
    # find outliers, set to nan 
    errors = diam0 - diam_sm0 
    std = np.nanstd(errors) 
    diam1 = np.copy(diam0) 
    diam1[(errors < (-std_thresh * std)) | (errors > (std_thresh * std))] = np.nan 
    # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise 
    diam_sm1 = smooth_interpolate_signal_sg( 
        diam1, window=window, order=3, interp_kind='linear') 
 
    # don't interpolate long strings of nans 
    t = np.diff(1 * np.isnan(diam1)) 
    begs = np.where(t == 1)[0] 
    ends = np.where(t == -1)[0] 
    if begs.shape[0] > ends.shape[0]: 
        begs = begs[:ends.shape[0]] 
    for b, e in zip(begs, ends): 
        if (e - b) > (fr * nan_thresh): 
            diam_sm1[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff 
 
    # diam_sm1 is the final smoothed pupil diameter estimate 
    return diam0, diam_sm1