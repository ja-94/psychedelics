"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""

"""
Optimization possibilities:
    - Merge subsampled_means() with svd_computations. There is a double computation of binned frame arrays.
"""
import cv2
import h5py
import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import os
import time
from io import StringIO
import numpy as np
from numba import vectorize
from scipy import io
from tqdm import tqdm


# visual progress things
def update_mainwindow_progressbar(MainWindow, GUIobject, s, prompt):
    if MainWindow is not None and GUIobject is not None:
        message = s.getvalue().split("\x1b[A\n\r")[0].split("\r")[-1]
        MainWindow.update_status_bar(
            prompt + message, update_progress=True, hide_progress=False
        )
        GUIobject.QApplication.processEvents()


def update_mainwindow_message(MainWindow, GUIobject, prompt, hide_progress=True):
    if MainWindow is not None and GUIobject is not None:
        MainWindow.update_status_bar(
            prompt, update_progress=False, hide_progress=hide_progress
        )
        GUIobject.QApplication.processEvents()

#-------------------------------------------------------------------------------------
# resolution reduction function (averages pixel value per bin)
def bin1d(X, bin_size, axis=0):
    """mean bin over axis of data with bin bin_size"""
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        Xb = (
            Xb[: size[axis] // bin_size * bin_size]
            .reshape((size[axis] // bin_size, bin_size, -1))
            .mean(axis=1)
        )
        Xb = Xb.swapaxes(axis, 0)
        size[axis] = Xb.shape[axis]
        Xb = Xb.reshape(size)
        return Xb
    else:
        return X

#----------------- Get_frames functions--------------------------------------------
"""
My case is simpler in workflow. I will only be working with one grayscale videos.

Edits: 1. I've implemented changes to grayscale color schemes.
"""
# individual frame
def get_frame(cframe, nframes, cumframes, containers):
    """
    Get required 'cframe' from a video (individual frame). 
    """

    # checking target frame is within boundaries

    cframe = np.maximum(0, np.minimum(nframes - 1, cframe))
    cframe = int(cframe)

    # finding the frame in set of videos (only useful if many concatenated videos)
    # not my case
    try:
        ivid = (cumframes < cframe).nonzero()[0][-1]
    except:
        ivid = 0

    # list of numpy frames
    img = []

    # setting up the video.capt
    for vs in containers[ivid]:
        frame_ind = cframe - cumframes[ivid]
        capture = vs

        #setting the correct frame
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

        # important function, .read() reads the frame
        ret, frame = capture.read()

        # changing color schemes (kinda useless as i work in grayscale
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[np.newaxis, ...]
            img.append(frame)
        else:
            print("Error reading frame")
    return img


# [NOT USED]
def get_batch_frames(
    frame_indices, total_frames, cumframes, containers, video_idx, grayscale=True
):
    # edit: changes defaults from facemap to grayscaled true
    # frame_indices = np.maximum(0, np.minimum(total_frames - 1, frame_indices))
    # frame_indices as given by epoch segment
    """
    try:
        video_idx = (cumframes < frame_indices).nonzero()[0][-1]
    except:
        video_idx = 0
    """
    imgs = []
    # for vs in containers[video_idx]:
    capture = containers[0][video_idx]
    for idx in frame_indices:
        frame_ind = idx - cumframes[video_idx]
        # capture = vs
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
        ret, frame = capture.read()
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[np.newaxis, ...]
        if ret:
            imgs.append(frame)
        else:
            print("Error reading frame")
    return np.array(imgs)

#[NOT USED IN MY IMPLEMENTATION]
def load_images_from_video(video_path, selected_frame_ind):
    """
    Load images from a video file. Helpful to debug.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for frame_ind in selected_frame_ind:
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print("Error reading frame")
    frames = np.array(frames)
    return frames

#[NOT USED IN MY IMPLEMENTATION]
def resample_timestamps(init_timestamps, target_timestamps):
    """
    Resample timestamps to a new time base.
    Parameters
    ----------
    init_timestamps : 1D-array
        Timestamps of the data.
    target_timestamps : 1D-array
        Target timestamps for resampling the data.
    Returns
    -------
    resampled_timestamps : 1D-array
        Resampled timestamps.
    """
    # Estimate the interpolation function for the data
    f = interp1d(
        init_timestamps.squeeze(),
        np.arange(init_timestamps.size),
        kind="linear",
        axis=-1,
        fill_value="extrapolate",
    )
    # Resample the data
    resampled_timestamps = f(target_timestamps)
    # Set bounds of the resampled timestamps
    resampled_timestamps[resampled_timestamps < 0] = 0
    resampled_timestamps[resampled_timestamps > init_timestamps.size - 1] = (
        init_timestamps.size - 1
    )
    return resampled_timestamps.squeeze().astype(int)


def get_frames(imall, containers, cframes, cumframes):
    """
    Gets target 'cframes' from selected video (opened with cv2).
    Modifies imall array by inserting captured frames (nframes, height, width, channels).

    Parameters
    ----------
    
    imall: np.array
        Initialized container array.
    
    containers: cv2.VideoCapture
        Opened VideoCapture instances.
    
    cframes: list
        List of target frames. For video subclipping, provide sequences of consecutive frames.    
    """
    nframes = cumframes[-1]  # total number of frames
    cframes = np.maximum(0, np.minimum(nframes - 1, cframes))
    cframes = np.arange(cframes[0], cframes[-1] + 1).astype(int)
    # Check frames exist in which video (for multiple videos, one view)
    ivids = (cframes[np.newaxis, :] >= cumframes[1:, np.newaxis]).sum(axis=0)

    for ii in range(len(containers[0])):  # for each view in the list
        nk = 0
        for n in np.unique(ivids):
            cfr = cframes[ivids == n]
            start = cfr[0] - cumframes[n]
            end = cfr[-1] - cumframes[n] + 1
            nt0 = end - start
            capture = containers[n][ii]
            # all setting up for correct videos
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != start:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            fc = 0
            ret = True

            # while loop for getting frames (also allows batch loading)
            while fc < nt0 and ret:
                ret, frame = capture.read()
                # if lecture was correct, add the frame to image_all[0][idx]
                if ret:
                    imall[ii][nk + fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    print("img load failed, replacing with prev..")
                    imall[ii][nk + fc] = imall[ii][nk + fc - 1]
                fc += 1
            nk += nt0

    # trimming for memory storage
    if nk < imall[0].shape[0]:
        for ii, im in enumerate(imall):
            imall[ii] = im[:nk].copy()


def close_videos(containers):
    """Method is called to close all videos/containers open for reading
    using openCV.
    Parameters:-(Input) containers: a 2D list of pointers to videos captured by openCV
                (Output) N/A"""
    for i in range(len(containers)):  # for each video in the list
        for j in range(len((containers[0]))):  # for each cam/view
            cap = containers[i][j]
            cap.release()


def get_frame_details(filenames):
    """
    Uses cv2 to open video files and obtain their details
    Parameters:-(Input) filenames: a 2D list of video files
                (Output) cumframes: list of total frame size for each cam/view
                (Output) Ly: list of dimension x for each cam/view
                (Output) Lx: list of dimension y for each cam/view
                (Output) containers: a 2D list of pointers to videos that are open
    """
    cumframes = [0]
    containers = []
    
    # this is why filenames always have to go into a list
    
    for fs in filenames:  # for each video in the list
        Ly = []
        Lx = []
        cs = []
        for n, f in enumerate(fs):  # for each cam/view
            cap = cv2.VideoCapture(f)
            cs.append(cap)
            framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            Lx.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            Ly.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        containers.append(cs)
        cumframes.append(cumframes[-1] + framecount)
    cumframes = np.array(cumframes).astype(int)
    return cumframes, Ly, Lx, containers

#[NOT USED IN MY IMPLEMENTATION]
def get_cap_features(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return fps, nframes


def get_skipping_frames(imall, filenames, cframes, cumframes):
    nframes = cumframes[-1]  # total number of frames
    cframes = np.maximum(0, np.minimum(nframes - 1, cframes))
    cframes = np.arange(cframes[0], cframes[-1] + 1).astype(int)
    ivids = (cframes[np.newaxis, :] >= cumframes[1:, np.newaxis]).sum(axis=0)
    i = 0
    for ii in range(len(filenames[0])):
        for n in np.unique(ivids):
            cfr = cframes[ivids == n]
            ifr = cfr - cumframes[n]
            capture = cv2.VideoCapture(filenames[n][ii])
            for iframe in ifr:
                capture.set(cv2.CAP_PROP_POS_FRAMES, iframe)
                ret, frame = capture.read()
                if ret:
                    imall[ii][i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    i += 1
                else:
                    break
            capture.release()


def multivideo_reshape(X, LY, LX, sy, sx, Ly, Lx, iinds):
    """reshape X matrix pixels x n matrix into LY x LX - embed each video at sy, sx"""
    """ iinds are indices of each video in concatenated array"""
    X_reshape = np.zeros((LY, LX, X.shape[-1]), np.float32)
    for i in range(len(Ly)):
        X_reshape[sy[i] : sy[i] + Ly[i], sx[i] : sx[i] + Lx[i]] = np.reshape(
            X[iinds[i]], (Ly[i], Lx[i], X.shape[-1])
        )
    return X_reshape


def roi_to_dict(ROIs, rROI=None):
    rois = []
    for i, r in enumerate(ROIs):
        rois.append(
            {
                "rind": r.rind,
                "rtype": r.rtype,
                "iROI": r.iROI,
                "ivid": r.ivid,
                "color": r.color,
                "yrange": r.yrange,
                "xrange": r.xrange,
                "saturation": r.saturation,
            }
        )
        if hasattr(r, "pupil_sigma"):
            rois[i]["pupil_sigma"] = r.pupil_sigma
        if hasattr(r, "ellipse"):
            rois[i]["ellipse"] = r.ellipse
        if rROI is not None:
            if len(rROI[i]) > 0:
                rois[i]["reflector"] = []
                for rr in rROI[i]:
                    rdict = {
                        "yrange": rr.yrange,
                        "xrange": rr.xrange,
                        "ellipse": rr.ellipse,
                    }
                    rois[i]["reflector"].append(rdict)

    return rois


def get_reflector(yrange, xrange, rROI=None, rdict=None):
    reflectors = np.zeros((yrange.size, xrange.size), bool)
    if rROI is not None and len(rROI) > 0:
        for r in rROI:
            ellipse, ryrange, rxrange = (
                r.ellipse.copy(),
                r.yrange.copy(),
                r.xrange.copy(),
            )
            ix = np.logical_and(rxrange >= 0, rxrange < xrange.size)
            ellipse = ellipse[:, ix]
            rxrange = rxrange[ix]
            iy = np.logical_and(ryrange >= 0, ryrange < yrange.size)
            ellipse = ellipse[iy, :]
            ryrange = ryrange[iy]
            reflectors[np.ix_(ryrange, rxrange)] = np.logical_or(
                reflectors[np.ix_(ryrange, rxrange)], ellipse
            )
    elif rdict is not None and len(rdict) > 0:
        for r in rdict:
            ellipse, ryrange, rxrange = (
                r["ellipse"].copy(),
                r["yrange"].copy(),
                r["xrange"].copy(),
            )
            ix = np.logical_and(rxrange >= 0, rxrange < xrange.size)
            ellipse = ellipse[:, ix]
            rxrange = rxrange[ix]
            iy = np.logical_and(ryrange >= 0, ryrange < yrange.size)
            ellipse = ellipse[iy, :]
            ryrange = ryrange[iy]
            reflectors[np.ix_(ryrange, rxrange)] = np.logical_or(
                reflectors[np.ix_(ryrange, rxrange)], ellipse
            )
    return reflectors.nonzero()

def video_placement(Ly, Lx):
    """
    Arrange a single video (or a list of one video) at (0, 0).
    Ly and Lx should be 1-element arrays or scalars.
    Returns:
        LY: total height
        LX: total width
        sy: y-offsets (array of zeros)
        sx: x-offsets (array of zeros)
    """

    # Accept both scalar and array input
    Ly = np.atleast_1d(Ly)
    Lx = np.atleast_1d(Lx)
    n = Ly.size

    LY = int(Ly[0]) # [0] just one video
    LX = int(Lx[0]) # [0] just one video
    sy = np.zeros(n, dtype=int) # no clue what is this, just in case
    sx = np.zeros(n, dtype=int)
    return LY, LX, sy, sx

# original implementation, causes problems
"""
def video_placement(Ly, Lx):
    Ly and Lx are lists of video sizes

    npix = Ly * Lx # total number of pixels
    picked = np.zeros((Ly.size,), bool) # tracks video 
    ly = 0
    lx = 0
    sy = np.zeros(Ly.shape, int)
    sx = np.zeros(Lx.shape, int)
    if Ly.size == 2:
        gridy = 1
        gridx = 2
    elif Ly.size == 3:
        gridy = 1
        gridx = 2
    else:
        gridy = int(np.round(Ly.size**0.5 * 0.75))
        gridx = int(np.ceil(Ly.size / gridy))
    LY = 0
    LX = 0
    iy = 0
    ix = 0
    while (~picked).sum() > 0:
        # place biggest movie first
        npix0 = npix.copy()
        npix0[picked] = 0
        imax = np.argmax(npix0)
        picked[imax] = 1
        if iy == 0:
            ly = 0
            rowmax = 0
        if ix == 0:
            lx = 0
        sy[imax] = ly
        sx[imax] = lx

        ly += Ly[imax]
        rowmax = max(rowmax, Lx[imax])
        if iy == gridy - 1 or (~picked).sum() == 0:
            lx += rowmax
        LY = max(LY, ly)
        iy += 1
        if iy >= gridy:
            iy = 0
            ix += 1
    LX = lx
    return LY, LX, sy, sx
"""

# randomized PCA with 100 components (could possibly modify depending on how ti goes)
def svdecon(X, k=100):
    """
    Singular Value Decomposition. Decomposes matrix X using a randomized numerical algorithm, keeping
    the first 100 principal components.
    """
    np.random.seed(0)  # Fix seed to get same output for eigsh
    U, Sv, V = PCA(
        n_components=k, svd_solver="randomized", random_state=np.random.RandomState(0)
    )._fit(X)[:3]
    return U, Sv, V


def binned_inds(Ly, Lx, sbin):
    """
    Computes binned indices for frame dimensions. Most part of the implementation is vestigial
    to allow multivideo grid algorithm. Working with single videos simplifies the operation.
    """

    # storing arrays
    Lyb = np.zeros((len(Ly),), np.int32) 
    Lxb = np.zeros((len(Ly),), np.int32)

    ir = [] # list with all binned indices (flattened)
    ix = 0

    # for all videos, Ly.length is just one (could optimize all this, but meh)
    for n in range(len(Ly)):
        Lyb[n] = int(np.floor(Ly[n] / sbin)) # the n-the video height is downscaled by sbin
        Lxb[n] = int(np.floor(Lx[n] / sbin)) # idem with width

        # this works based on the grid format of video placement
        ir.append(np.arange(ix, ix + Lyb[n] * Lxb[n], 1, int)) # one single array from 0 to binned(height x width)
        ix += Lyb[n] * Lxb[n]
    return Lyb, Lxb, ir

# parallel function to floatize integers
@vectorize(["float32(uint8)"], nopython=True, target="parallel")
def ftype(x):
    return np.float32(x)


# IMPORTANT FUNCTION
def spatial_bin(im, sbin, Lyb, Lxb):
    """
    Resolution reduction via pixel pooling (downsamples by sbin factor). Returns flattened frames 
    with downsampled pixel bins.

    Parameters
    ----------
    im: np.array
        Frame array of shape (n_frames, height, width)
    sbin: int
        Downsampling factor. If sbin > 1, pixels are downscaled by 1/sbin.
    Lyb, Lxb: np.arrays
        Binned pixel arrays sourced from binned_inds().

    Returns
    --------
    imbin: np.array
        Pixel-pooled frame arrays of shape (n_frames, Lxb*Lyb [total pixel bins])
    """
    imbin = im.astype(np.float32) # floatize
    if sbin > 1:
        # cropping if sbin*binned_sizes doesn't match, crop
        
        # img_shape(frames, height, width)
        # after reshape, flatten on the frame (a multidimensional line of frames idk)
        # binning scheme:
        #   Lyb height groups of 'sbin' elements (e.g., in a 8x8, 4 groups of 2 bins)
        #   Lxb width groups of 'sbin' elements (e.g, idem)

        # first mean is horizontal (row averaging), second mean is vertical (column_average the row means)

        imbin = (
            (np.reshape(im[:, : Lyb * sbin, : Lxb * sbin], (-1, Lyb, sbin, Lxb, sbin)))
            .mean(axis=-1)
            .mean(axis=-2)
        )
    
    # reshape back into a working frame
    imbin = np.reshape(imbin, (-1, Lyb * Lxb))
    return imbin

# initializing array of frames
def imall_init(nfr, Ly, Lx):
    """
    Initializes array of shape (n_frames, Ly, Lx).
    """
    imall = []
    for n in range(len(Ly)):
        imall.append(np.zeros((nfr, Ly[n], Lx[n]), "uint8"))
    return imall


def subsampled_mean(
    containers, cumframes, Ly, Lx, sbin=4, GUIobject=None, MainWindow=None, start_sec = None, end_sec = None, fps = 60
):
    """
    Computes average motion energy and raw pixel expression for binned pixel groups.

    Parameters
    ----------
    containers: cv2.VideoCaptures
        Videos. 
    cumframes: int
        Cumulative frames (vestigial; for single-video, simply [0, total_frames])
    Ly, Lx: int
        Video dimensions (height, width resp.)
    sbin: int (default 4)
        Downsampling coefficient.
    start_sec, end_sec: int
        Interval timestamps (seconds). Useful for subclip analysis.
    fps: int (default 60 for left_camera videos)
        Frame rate.

    Returns
    ----------
    avgframe0: np.array
        Array of shape (Lyb * Lxb,) [single dimension, binned pixels].
        Average pixel expression.

    avgmotion0: np.array
        Array of shape (Lyb * Lxb,). Average absolute motion energy per binned pixel groups.

    """
    # grab up to 1000 frames in the form of non-sequentially extracted 100-frame chunks
    # to average over for mean (useful to standardize motion energy)

    # containers: list of videos loaded with opencv
    # cumframes: cumulative frames across videos (vestigial!)
    # Ly, Lx are the sizes of the videos (unique)
    # sbin is the size of spatial binning

    # added implementation, start_sec, end_sec, and fps allow for subclip processing

    nframes = cumframes[-1]

    if start_sec is not None and end_sec is not None and fps is not None:
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # range validation
        start_frame = max(0, start_frame)
        end_frame = min(nframes, end_frame)
    else:
        start_frame = 0
        end_frame = nframes
    # frame range
    print("start: ", start_frame, " end: ", end_frame)

# Get 1000 frames such that each 100 frame chunk 
# comes "equally" distant parts of the video

    # number of total frames
    nf = min(1000, end_frame - start_frame) # load up to 1000 frames or frame dif if smol

    # number of frames in each loaded chunk (nsegs chunks)
    nt0 = min(100, np.diff(cumframes).min(), nf) 

    #segments
    nsegs = int(np.floor(nf / nt0))
    
    #  Segment FRAMESTAMPS from start_frame to end_frame - 100
    tf = np.floor(np.linspace(start_frame, end_frame - nt0, nsegs)).astype(int)

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)

    # initializing subchunk container (100 frames, Ly, Lx)
    imall = imall_init(nt0, Ly, Lx)

    avgframe = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    avgmotion = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    ns = 0

    s = StringIO()

    for n in tqdm(range(nsegs), file=s):

        # initial timestamp (defining chunk window)
        t = tf[n]
        # imall: frame array; containers: video; np.arange() spans desired frames; cumframes
        get_frames(imall, containers, np.arange(t, t + nt0), cumframes)
        
        for n, im in enumerate(imall):

            #  imbin: (nframes, nbins (like flattened pixels))
            imbin = spatial_bin(im, sbin, Lyb[n], Lxb[n])
            
            # average pixel bin expression bin-to-bin
            avgframe[ir[n]] += imbin.mean(axis=0)

            # absolute motion energy (bin-to-bin)
            imbin = np.abs(np.diff(imbin, axis=0))
            avgmotion[ir[n]] += imbin.mean(axis=0)
        ns += 1
        update_mainwindow_progressbar(
            MainWindow, GUIobject, s, "Computing subsampled mean "
        )
    update_mainwindow_message(
        MainWindow, GUIobject, "Finished computing subsampled mean"
    )
    # averages
    avgframe /= float(ns)
    avgmotion /= float(ns)
    avgframe0 = []
    avgmotion0 = []

    # elements in avgframe0 and avgmotion0 are lists of bin-to-bin averages
    for n in range(len(Ly)):
        avgframe0.append(avgframe[ir[n]])
        avgmotion0.append(avgmotion[ir[n]])
    return avgframe0, avgmotion0


def compute_SVD(
    containers,
    cumframes,
    Ly,
    Lx,
    avgframe,
    avgmotion,
    motSVD=True,
    movSVD=False,
    ncomps=500,
    sbin=4,
    rois=None,
    fullSVD=True,
    GUIobject=None,
    MainWindow=None,
    start_sec = None,
    end_sec = None,
    fps = 60
):
    """
    Compute the SVD over frames in chunks, combine the chunks, and take a mega-SVD.

     Parameters
    ----------
    containers: cv2.VideoCaptures
        Videos. 
    cumframes: int
        Cumulative frames (vestigial; for single-video, simply [0, total_frames])
    Ly, Lx: int
        Video dimensions (height, width resp.)
    avgframe, avgmotion: np.array (Lyb * Lxb)
        Average bin pixel metrics. avgmotion useful for normalizing motion energy.
    
    motSVD: bool (True)
        Enables motion energy SVD analysis.
    movSVD: bool (False)
        Enables raw pixel SVD analysis.
    sbin: int (default 4)
        Downsampling coefficient.
    ncomps: int (300)
        Number of components to keep in the final SVD.
    start_sec, end_sec: int
        Interval timestamps (seconds). Useful for subclip analysis.
    fps: int (default 60 for left_camera videos)
        Frame rate.
    
    Returns
    -------
    U_mot: np.array (binned pixels, ncomps)
        Spatial PC (motion energy eigenmask). SVD motion energy loading per pixel bin.
    U_mov: np.array (binned pixels, ncomps)
        idem. Uses raw pixel bins instead of motion energy.
    S_mot: np.array (ncomps,)
        List of singular values (variance explained) for each motion energy PC.
    S_mov: np.array (ncomps,)
        idem. Uses raw pixel bins instead of motion energy.
    """
    sbin = max(1, sbin)
    nframes = cumframes[-1]

    # subclip implementation
    if start_sec is not None and end_sec is not None and fps is not None:
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        # range validation
        start_frame = max(0, start_frame)
        end_frame = min(nframes, end_frame)
    else:
        start_frame = 0
        end_frame = nframes

    total_frames = end_frame - start_frame

    # important
    if total_frames < 1000:
        raise ValueError("Subclip too short for SVD.")

    # load in chunks of up to 1000 frames
    nt0 = min(1000, total_frames)
    #nsegs = int(min(np.floor(15000 / nt0), np.floor(nframes / nt0))) 
    nsegs = int(np.ceil(total_frames / nt0))

    nc = int(250)  # <- how many PCs to keep in each chunk
    nc = min(nc, nt0 - 1)
    if nsegs == 1:
        nc = min(ncomps, nt0 - 1)

    # what times to sample
    tf = [start_frame + i * nt0 for i in range(nsegs)]

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    if fullSVD:
        U_mot = (
            [np.zeros(((Lyb * Lxb).sum(), nsegs * nc), np.float32)] if motSVD else []
        )
        U_mov = (
            [np.zeros(((Lyb * Lxb).sum(), nsegs * nc), np.float32)] if movSVD else []
        )
    else:
        U_mot = [np.zeros((0, 1), np.float32)] if motSVD else []
        U_mov = [np.zeros((0, 1), np.float32)] if movSVD else []
    nroi = 0
    motind = []
    ivid = []

    ni_mot = [0]
    ni_mov = [0]
    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r["ivid"])
            if r["rind"] == 1:
                nroi += 1
                motind.append(i)
                nyb = r["yrange_bin"].size
                nxb = r["xrange_bin"].size
                U_mot.append(
                    np.zeros((nyb * nxb, nsegs * min(nc, nyb * nxb)), np.float32)
                )
                U_mov.append(
                    np.zeros((nyb * nxb, nsegs * min(nc, nyb * nxb)), np.float32)
                )
                ni_mot.append(0)
                ni_mov.append(0)
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind)

    ns = 0
    w = StringIO()
    tic = time.time()
    for n in tqdm(range(nsegs), file=w):
        t = tf[n]
        # Last chunk may be shorter
        chunk_len = min(nt0, end_frame - t)
        if chunk_len < 2:
            continue
        img = imall_init(chunk_len, Ly, Lx)
        get_frames(img, containers, np.arange(t, t + chunk_len), cumframes)
        if fullSVD:
            imall_mot = np.zeros((img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32)
            imall_mov = np.zeros((img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32)
        for ii, im in enumerate(img):
            usevid = False
            if fullSVD:
                usevid = True
            if nroi > 0:
                wmot = (ivid[motind] == ii).nonzero()[0]
                if wmot.size > 0:
                    usevid = True

        # IMPORTANT SECTION
            if usevid:
                if motSVD:  # compute motion energy
                    imbin_mot = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    imbin_mot = np.abs(np.diff(imbin_mot, axis=0))
                    #standardize
                    imbin_mot -= avgmotion[ii]
                    if fullSVD:
                        imall_mot[:, ir[ii]] = imbin_mot
                if movSVD:  # for raw frame svd
                    imbin_mov = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    imbin_mov = imbin_mov[1:, :]
                    imbin_mov -= avgframe[ii]
                    if fullSVD:
                        imall_mov[:, ir[ii]] = imbin_mov
                if nroi > 0 and wmot.size > 0:
                    if motSVD:
                        imbin_mot = np.reshape(imbin_mot, (-1, Lyb[ii], Lxb[ii]))
                    if movSVD:
                        imbin_mov = np.reshape(imbin_mov, (-1, Lyb[ii], Lxb[ii]))
                    wmot = np.array(wmot).astype(int)
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        ymin = rois[wroi[i]]["yrange_bin"][0]
                        ymax = rois[wroi[i]]["yrange_bin"][-1] + 1
                        xmin = rois[wroi[i]]["xrange_bin"][0]
                        xmax = rois[wroi[i]]["xrange_bin"][-1] + 1
                        if motSVD:
                            lilbin = imbin_mot[:, ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))

                            # important to prevent small chunks nc outofbound
                            n_samples, n_features = imall_mot.T.shape
                            ncb = min(nc, n_samples, n_features)
                            usv = svdecon(imall_mot.T, k=ncb)

                            ncb = usv[0].shape[-1]
                            u0, uend = ni_mot[wmot[i] + 1], ni_mot[wmot[i] + 1] + ncb
                            U_mot[wmot[i] + 1][:, u0:uend] = usv[0] * usv[1]
                            ni_mot[wmot[i] + 1] += ncb
                        if movSVD:
                            lilbin = imbin_mov[:, ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))

                            n_samples, n_features = lilbin.T.shape
                            ncb = min(nc, n_samples, n_features)
                            usv = svdecon(lilbin.T, k=ncb)

                            ncb = usv[0].shape[-1]
                            u0, uend = ni_mov[wmot[i] + 1], ni_mov[wmot[i] + 1] + ncb
                            U_mov[wmot[i] + 1][:, u0:uend] = usv[0] * usv[1]
                            ni_mov[wmot[i] + 1] += ncb
            print(f"computed svd chunk {n} / {nsegs}, time {time.time()-tic: .2f}sec")
        update_mainwindow_progressbar(MainWindow, GUIobject, w, "Computing SVD ")

        if fullSVD:
            if motSVD:
                # preventing exceeding n of components
                n_samples, n_features = imall_mot.T.shape
                ncb = min(nc, n_samples, n_features)

                usv = svdecon(imall_mot.T, k=ncb) # svd with chunk
                ncb = usv[0].shape[-1]
                U_mot[0][:, ni_mot[0] : ni_mot[0] + ncb] = usv[0] * usv[1]
                ni_mot[0] += ncb
            if movSVD:
                ncb = min(nc, imall_mov.shape[-1])
                usv = svdecon(imall_mov.T, k=ncb)
                ncb = usv[0].shape[-1]
                U_mov[0][:, ni_mov[0] : ni_mov[0] + ncb] = usv[0] * usv[1]
                ni_mov[0] += ncb
        ns += 1
    #----------------------- end of segment processing -----------------------
    S_mot = np.zeros(500, "float32")
    S_mov = np.zeros(500, "float32")
    # take SVD of concatenated spatial PCs
    if ns > 1:
        for nr in range(len(U_mot)):
            if nr == 0 and fullSVD:
                if motSVD:
                    U_mot[nr] = U_mot[nr][:, : ni_mot[0]]
                    usv = svdecon(
                        U_mot[nr], k=min(ncomps, U_mot[nr].shape[0] - 1)
                    )
                    U_mot[nr] = usv[0]
                    S_mot = usv[1]
                if movSVD:
                    U_mov[nr] = U_mov[nr][:, : ni_mov[0]]
                    usv = svdecon(
                        U_mov[nr], k=min(ncomps, U_mov[nr].shape[0] - 1)
                    )
                    U_mov[nr] = usv[0]
                    S_mov = usv[1]
            elif nr > 0:
                if motSVD:
                    U_mot[nr] = U_mot[nr][:, : ni_mot[nr]]
                    usv = svdecon(
                        U_mot[nr], k=min(ncomps, U_mot[nr].shape[0] - 1)
                    )
                    U_mot[nr] = usv[0]
                    S_mot = usv[1]
                if movSVD:
                    U_mov[nr] = U_mov[nr][:, : ni_mov[nr]]
                    usv = svdecon(
                        U_mov[nr], k=min(ncomps, U_mov[nr].shape[0] - 1)
                    )
                    U_mov[nr] = usv[0]
                    S_mov = usv[1]

    update_mainwindow_message(MainWindow, GUIobject, "Finished computing svd")

    return U_mot, U_mov, S_mot, S_mov

"""
WARNING: GOTTA RECHECK THIS ONE.
Different things from FaceMap:
- No pupil, whisking, running specialize motifs. It could be useful to reimplement (look at FaceMap.process source code)
- No multivideo support. I've been trying to eliminate the grid implementation (harmless if single video,
 but makes code complicated)
 - No M array

 Improvements:
 - Doing some cleaning with V_mov and M
"""
def process_ROIs(
    containers,
    cumframes,
    Ly,
    Lx,
    avgframe,
    avgmotion,
    U_mot,
    U_mov,
    motSVD=True,
    movSVD=False,
    sbin=3,
    tic=None,
    rois=None,
    fullSVD=True,
    GUIobject=None,
    MainWindow=None,
    start_sec = None,
    end_sec = None,
    fps = 60,
):
    # project U onto each frame in the video and compute the motion energy for motSVD
    # also compute pupil on single frames on non binned data
    # the pixels are binned in spatial bins of size sbin
    # containers is a list of videos loaded with av
    # cumframes are the cumulative frames across videos
    if tic is None:
        tic = time.time()
    nframes = cumframes[-1]

        # --- Subclip support ---
    if start_sec is not None and end_sec is not None and fps is not None:
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        start_frame = max(0, start_frame)
        end_frame = min(nframes, end_frame)
    else:
        start_frame = 0
        end_frame = nframes
    total_frames = end_frame - start_frame
    # -----------------------
    print("start frame:", start_frame, "to ", "end frame: ", end_frame)
    motind = []
    ivid = []

    if fullSVD:
        if motSVD:
            ncomps_mot = U_mot[0].shape[-1]

        
        # n_samples x n_features
        V_mot = np.zeros((total_frames, ncomps_mot), np.float32) if motSVD else None

    else:
        V_mot = [np.zeros((0, 1), np.float32)] if motSVD else []
        V_mov = [np.zeros((0, 1), np.float32)] if movSVD else []
        M = [np.zeros((0,), np.float32)]

    if rois is not None:
        for i, r in enumerate(rois):
            ivid.append(r["ivid"])
            if r["rind"] == 1:
                motind.append(i)
                nroi += 1
                if motSVD:
                    V_mot.append(np.zeros((total_frames, U_mot[nroi].shape[1]), np.float32))
                if movSVD:
                    V_mov.append(np.zeros((total_frames, U_mov[nroi].shape[1]), np.float32))
                M.append(np.zeros((total_frames,), np.float32))

    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind).astype(np.int32)

    # compute in chunks of 1000
    nt0 = 500
    nsegs = int(np.ceil(total_frames / nt0))
    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    imend = []
    for ii in range(len(Ly)):
        imend.append([])

    s = StringIO()
    
    t0 = start_frame
    motion_frame_counter = 0

    for n in tqdm(range(nsegs), file=s):
        # setting the interval
        t1 = min(t0+nt0, end_frame) # interval end
        
        img = imall_init(t1-t0, Ly, Lx) # frame selection
        get_frames(img, containers, np.arange(t0, t1), cumframes)


        # bin and get motion
        if fullSVD:
            if n > 0:
                # n_frames x (total bin pixels)
                imall_mot = np.zeros((img[0].shape[0], (Lyb * Lxb).sum()), np.float32)
            else:
                imall_mot = np.zeros(
                    (img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32
                )
                imall_mov = np.zeros(
                    (img[0].shape[0] - 1, (Lyb * Lxb).sum()), np.float32
                )
        if fullSVD or nroi > 0:
            for ii, im in enumerate(img):
                usevid = False
                if fullSVD:
                    usevid = True
                if nroi > 0:
                    wmot = (ivid[motind] == ii).nonzero()[0]
                    if wmot.size > 0:
                        usevid = True
                if usevid:
                    imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    if n > 0:
                        imbin = np.concatenate(
                            (imend[ii][np.newaxis, :], imbin), axis=0
                        )
                    imend[ii] = imbin[-1].copy()
                    if motSVD:  # compute motion energy for motSVD
                        imbin_mot = np.abs(np.diff(imbin, axis=0))
                    if movSVD:  # use raw frames for movSVD
                        imbin_mov = imbin[1:, :]
                    if fullSVD:
                        if motSVD:
                            M[0][motion_frame_counter : motion_frame_counter + imbin_mot.shape[0]] += imbin_mot.sum(axis=-1)
                            imall_mot[:, ir[ii]] = imbin_mot - avgmotion[ii].flatten()
                        if movSVD:
                            imall_mov[:, ir[ii]] = imbin_mov - avgframe[ii].flatten()
                """ 
                ROI Implementation that has to be fixed
                if nroi > 0 and wmot.size > 0:
                    wmot = np.array(wmot).astype(int)
                    if motSVD:
                        imbin_mot = np.reshape(imbin_mot, (-1, Lyb[ii], Lxb[ii]))
                        avgmotion[ii] = np.reshape(avgmotion[ii], (Lyb[ii], Lxb[ii]))
                    if movSVD:
                        imbin_mov = np.reshape(imbin_mov, (-1, Lyb[ii], Lxb[ii]))
                        avgframe[ii] = np.reshape(avgframe[ii], (Lyb[ii], Lxb[ii]))
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        ymin = rois[wroi[i]]["yrange_bin"][0]
                        ymax = rois[wroi[i]]["yrange_bin"][-1] + 1
                        xmin = rois[wroi[i]]["xrange_bin"][0]
                        xmax = rois[wroi[i]]["xrange_bin"][-1] + 1
                        if motSVD:
                            lilbin = imbin_mot[:, ymin:ymax, xmin:xmax]
                            M[wmot[i] + 1][motion_frame_counter : motion_frame_counter + lilbin.shape[0]] = lilbin.sum(
                                axis=(-2, -1)
                            )
                            lilbin -= avgmotion[ii][ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            vproj = lilbin @ U_mot[wmot[i]]
                            V_mot[wmot[i] + 1][motion_frame_counter : motion_frame_counter + vproj.shape[0], :] = vproj

                        if movSVD:
                            lilbin = imbin_mov[:, ymin:ymax, xmin:xmax]
                            lilbin -= avgframe[ii][ymin:ymax, xmin:xmax]
                            lilbin = np.reshape(lilbin, (lilbin.shape[0], -1))
                            vproj = lilbin @ U_mov[wmot[i]]
                            V_mov[wmot[i] + 1][motion_frame_counter : motion_frame_counter + vproj.shape[0], :] = vproj
"""
            if fullSVD:
                if motSVD:
                    # imall_mot is n_frames x n_totalpixels_flat

                    # U_mot: n_totalpixels_Flat x n_comps

                    #all mean-centered motion frames (500)
                    print("imall_mot:", imall_mot.shape)

                    # projecting

                    vproj = imall_mot @ U_mot
                    print("vproj shape", vproj.shape)

                    # putting into global vmot
                    V_mot[motion_frame_counter : motion_frame_counter + vproj.shape[0], :] = vproj
                    
                    print(f"interval start {t0}, end {t1}, writing to rows {motion_frame_counter} to {motion_frame_counter + vproj.shape[0]}")

                    motion_frame_counter += vproj.shape[0]


            if n % 10 == 0:
                print(
                    f"computed video chunk {n} / {nsegs}, time {time.time()-tic: .2f}sec"
                )

            update_mainwindow_progressbar(
                MainWindow, GUIobject, s, "Computing ROIs and/or motSVD/movSVD "
            )
            # updating

            t0 += nt0

    update_mainwindow_message(
        MainWindow, GUIobject, "Finished computing ROIs and/or motSVD/movSVD "
    )

    return V_mot, V_mov, M


def save(proc, savepath=None):
    # save ROIs and traces
    basename, filename = os.path.split(proc["filenames"][0][0])
    filename, ext = os.path.splitext(filename)
    # Add epoch info to filename if available
    epoch_str = ""
    if proc.get("start_sec") is not None and proc.get("end_sec") is not None:
        epoch_str = f"_start{int(proc['start_sec'])}_end{int(proc['end_sec'])}"
    if savepath is not None:
        basename = savepath
    savename = os.path.join(basename, (f"{filename}{epoch_str}_proc.npy"))
    # TODO: use npz
    # np.savez(savename, **proc)
    np.save(savename, proc)
    if proc["save_mat"]:
        if "save_path" in proc and proc["save_path"] is None:
            proc["save_path"] = basename

        d2 = {}
        if proc["rois"] is None:
            proc["rois"] = 0
        for k in proc.keys():
            if (
                isinstance(proc[k], list)
                and len(proc[k]) > 0
                and isinstance(proc[k][0], np.ndarray)
            ):
                for i in range(len(proc[k])):
                    d2[k + "_%d" % i] = proc[k][i]
            else:
                d2[k] = proc[k]
        savenamemat = os.path.join(basename, (f"{filename}{epoch_str}_proc.mat"))
        io.savemat(savenamemat, d2)
        del d2
    return savename


def run(
    filenames,
    sbin=4,
    motSVD=True,
    movSVD=False,
    GUIobject=None,
    parent=None,
    proc=None,
    savepath=None,
    start_sec = None,
    end_sec = None,
    fps = 60
):
    """
    Process video files using SVD computation of motion and/or raw movie data.
    
    To set ROIs, use FaceMap GUI to generate a proc file with ROI parameters.
    Parameters
    ----------
    filenames: 2D-list
        List of video files to process. Each element of the list is a list of
        filenames for video(s) recorded simultaneously. For example, if two videos were recorded simultaneously, the list would be: [['video1.avi', 'video2.avi']], and if the videos were recorded sequentially, the list would be: [['video1.avi'], ['video2.avi']].
    sbin: int
        Spatial binning factor. If sbin > 1, the movie will be spatially binned by a factor of sbin.
    motSVD: bool
        If True, compute SVD of motion in the video i.e. the difference between consecutive frames.
    movSVD: bool
        If True, compute SVD of raw movie data.
    GUIobject: GUI object
        GUI object to update progress bar. If None, no progress bar will be updated.
    parent: GUI object
        Parent GUI object to update progress bar. If None, no progress bar will be updated.
    proc: dict
        Dictionary containing previously processed data. If provided, parameters from the saved data, such as sbin, rois, sy, sx, etc. will be used.
    savepath: str
        Path to save processed data. If None, the processed data will be saved in the same directory as the first video file.
    start_sec, end_sec: int
        Subclip timestamps.
    fps: int
        Frames per seconds.
    Returns
    -------
    savename: str
        Path to saved processed data.
    """
    start = time.time()
    # grab files
    rois = None
    sy, sx = 0, 0
    if parent is not None:
        filenames = parent.filenames
        _, _, _, containers = get_frame_details(filenames)
        cumframes = parent.cumframes
        sbin = parent.sbin
        rois = roi_to_dict(parent.ROIs, parent.rROI)
        Ly = parent.Ly
        Lx = parent.Lx
        fullSVD = parent.multivideo_svd_checkbox.isChecked()
        save_mat = parent.save_mat.isChecked()
        sy = parent.sy
        sx = parent.sx
        motSVD, movSVD = (
            parent.motSVD_checkbox.isChecked(),
            parent.movSVD_checkbox.isChecked(),
        )
    else:
        cumframes, Ly, Lx, containers = get_frame_details(filenames)

        if np.isnan(Ly) or np.isnan(Lx):
            raise ValueError("Video height (Ly) or width (Lx) is NaN. Check your video file and metadata.")
        Lybin, Lxbin, iinds = binned_inds(Ly, Lx, sbin)

        if proc is None:
            sbin = sbin
            fullSVD = True
            save_mat = False
            rois = None
        else:
            sbin = proc["sbin"]
            fullSVD = proc["fullSVD"]
            save_mat = proc["save_mat"]
            rois = proc["rois"]
            sy = proc["sy"]
            sx = proc["sx"]
            savepath = proc["savepath"] if savepath is None else savepath #proc["savepath"] if savepath is not None else savepath

    Lybin, Lxbin, iinds = binned_inds(Ly, Lx, sbin)
    LYbin, LXbin, sybin, sxbin = video_placement(Lybin, Lxbin)

    # number of mot/mov ROIs
    nroi = 0
    if rois is not None:
        for r in rois:
            if r["rind"] == 1:
                r["yrange_bin"] = np.arange(
                    np.floor(r["yrange"][0] / sbin), np.floor(r["yrange"][-1] / sbin)
                ).astype(int)
                r["xrange_bin"] = np.arange(
                    np.floor(r["xrange"][0] / sbin), np.floor(r["xrange"][-1]) / sbin
                ).astype(int)
                nroi += 1
    
    tic = time.time()
    # compute average frame and average motion across videos (binned by sbin) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tqdm.write("Computing subsampled mean...")
    avgframe, avgmotion = subsampled_mean(
        containers, cumframes, Ly, Lx, sbin, GUIobject, parent, start_sec, end_sec, fps
    )

    avgframe_reshape = multivideo_reshape(
        np.hstack(avgframe)[:, np.newaxis],
        LYbin,
        LXbin,
        sybin,
        sxbin,
        Lybin,
        Lxbin,
        iinds,
    )
    avgframe_reshape = np.squeeze(avgframe_reshape)
    avgmotion_reshape = multivideo_reshape(
        np.hstack(avgmotion)[:, np.newaxis],
        LYbin,
        LXbin,
        sybin,
        sxbin,
        Lybin,
        Lxbin,
        iinds,
    )
    avgmotion_reshape = np.squeeze(avgmotion_reshape)

    # Update user with progress
    tqdm.write("Computed subsampled mean at %0.2fs" % (time.time() - tic))
    if parent is not None:
        parent.update_status_bar("Computed subsampled mean")
    if GUIobject is not None:
        GUIobject.QApplication.processEvents()

    # Compute motSVD and/or movSVD from frames subsampled across videos
    #   and return spatial components                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ncomps = 500
    if fullSVD: #or nroi > 0:
        tqdm.write("Computing subsampled SVD...")
        U_mot, U_mov, S_mot, S_mov = compute_SVD(
            containers,
            cumframes,
            Ly,
            Lx,
            avgframe,
            avgmotion,
            motSVD,
            movSVD,
            ncomps=ncomps,
            sbin=sbin,
            rois=rois,
            fullSVD=fullSVD,
            GUIobject=GUIobject,
            MainWindow=parent,
            start_sec = start_sec,
            end_sec = end_sec,
            fps = fps
        )
        tqdm.write("Computed subsampled SVD at %0.2fs" % (time.time() - tic))

        if parent is not None:
            parent.update_status_bar("Computed subsampled SVD")
        if GUIobject is not None:
            GUIobject.QApplication.processEvents()

        U_mot_reshape = U_mot.copy()
        U_mov_reshape = U_mov.copy()
        if fullSVD:
            if motSVD:
                U_mot_reshape[0] = multivideo_reshape(
                    U_mot_reshape[0], LYbin, LXbin, sybin, sxbin, Lybin, Lxbin, iinds
                )
            if movSVD:
                U_mov_reshape[0] = multivideo_reshape(
                    U_mov_reshape[0], LYbin, LXbin, sybin, sxbin, Lybin, Lxbin, iinds
                )
            """
        if nroi > 0:
            k = 1
            for r in rois:
                if r["rind"] == 1:
                    ly = r["yrange_bin"].size
                    lx = r["xrange_bin"].size
                    if motSVD:
                        U_mot_reshape[k] = np.reshape(
                            U_mot[k].copy(), (ly, lx, U_mot[k].shape[-1])
                        )
                    if movSVD:
                        U_mov_reshape[k] = np.reshape(
                            U_mov[k].copy(), (ly, lx, U_mov[k].shape[-1])
                        )
                    k += 1
                """
    else:
        U_mot, U_mov, S_mot, S_mov = [], [], [], []
        U_mot_reshape, U_mov_reshape = [], []

    # Add V_mot and/or V_mov calculation: project U onto all movie frames ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # and compute pupil (if selected)
    tqdm.write("Computing ROIs and/or motSVD/movSVD")
    V_mot, V_mov, M = process_ROIs(
        containers,
        cumframes,
        Ly,
        Lx,
        avgframe,
        avgmotion,
        U_mot,
        U_mov,
        motSVD,
        movSVD,
        sbin=sbin,
        tic=tic,
        rois=rois,
        fullSVD=fullSVD,
        GUIobject=GUIobject,
        MainWindow=parent,
        start_sec=start_sec, 
        end_sec=end_sec, 
        fps=fps
    )
    tqdm.write("Computed ROIS and/or motSVD/movSVD at %0.2fs" % (time.time() - tic))

    # Save output  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    proc = {
        "filenames": filenames,
        "save_path": savepath,
        "Ly": Ly,
        "Lx": Lx,
        "sbin": sbin,
        "fullSVD": fullSVD,
        "save_mat": save_mat,
        "Lybin": Lybin,
        "Lxbin": Lxbin,
        "sybin": sybin,
        "sxbin": sxbin,
        "LYbin": LYbin,
        "LXbin": LXbin,
        "avgframe": avgframe,
        "avgmotion": avgmotion,
        "avgframe_reshape": avgframe_reshape,
        "avgmotion_reshape": avgmotion_reshape,
        "motion": M,
        "motSv": S_mot,
        "movSv": S_mov,
        "motMask": U_mot,
        "movMask": U_mov,
        "motMask_reshape": U_mot_reshape,
        "movMask_reshape": U_mov_reshape,
        "motSVD": V_mot,
        "movSVD": V_mov,
        "rois": rois,
        "sy": sy,
        "sx": sx,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "fps": fps,
    }
    # save processing
    savename = save(proc, savepath)
    close_videos(containers)

    if parent is not None:
        parent.update_status_bar("Output saved in " + savepath)
    if GUIobject is not None:
        GUIobject.QApplication.processEvents()
    tqdm.write("run time %0.2fs" % (time.time() - start))

    return savename



