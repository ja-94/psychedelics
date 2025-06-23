import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
from scipy.stats import entropy
import ibllib.io.video as vidio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


def get_frames(url, init, n_frames):
    """
    Gets n_frames frames from a video at url location.

    Parameters
    ----------
    url: string
        Local or digital location for the video. If locally saved, use one.eid2path(eid) to know location (eid i.e. experimental id)

    init, n_frames: integers
        Init refers to initial index. n_frames is the number of frames in the video segment.
        
    Returns
    -------
    numpy.ndarray
        An array of shape (n_frames, height, width) containing movie frames.
    
    """

    return vidio.get_video_frames_preload(url, range(init, init+n_frames), mask = np.s_[...,0])


def exponential (x, tau):
     """
    Exponential curve for fit.

    Parameters
    ----------
    x: float
        Function variable.

    tau: float
        Decay constant.

        
    Returns
    -------
    function: numpy.exp
        Exponential function with parameters as input.
    
    """
     return np.exp(-x/tau)


def motion_energy(frames):
    """
    Computes the motion energy matrix for each frame segment.
    
    Calculates pixel energy differences across frames.

    Parameters
    ----------
    frames: numpy.ndarray
        An array of shape (n_frames, height, width, channel) containing segmented frames from a given video.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_frames - 1, height, width) containing the motion energy differences for each pixel
        in different timestamps
    
    """
    return np.diff(frames[...], axis = 0)

def run_pca(energymatrix, components, plot = False):
    """
    Runs Principal Component Analysis (PCA) for each video segment.

    Reshapes the energy matrix so that each row represents a frame (height x width pixels) and min-max normalizes the matrix.
    Computes Principal Components via Singular Value Decomposition based on number or explained variance threshold.
    Fits an exponential curve to the explained variance to inspect individual components' explained variance contribution. 
    
    Parameters
    ----------
    energymatrix : numpy.ndarray
        Array of shape (n_frames, height, width) containing motion energy differences in a video segment
    components:
        (int) Number of components to extract; (float) 0 < cmp < 1; explained variance threshold
    plot: boolean
        Enables plotting of explained variance curve.

    Returns
    -------
    pca: sklearn.PCA()
        a PCA object containing the results of the PCA analysis.
    
    dict
        A dictionary containing the first principle component's variance explained,
        the variance explained by all components, the tau (exponential decay rate of the 
        PCA spectrum), and the normalized spectral entropy.        
    """
    n_frames = energymatrix.shape[0]

    X = energymatrix.reshape(n_frames, -1)

    # setting up the pca: normalization and fit
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    pca = PCA(components)
    pca.fit(X_norm)

    # computing curves of variance decay
    [tau], _ = scipy.optimize.curve_fit(exponential, np.arange(pca.n_components_), pca.explained_variance_ratio_/pca.explained_variance_ratio_[0])

    dict_measures = {}
    dict_measures[f'pc1'] = pca.explained_variance_ratio_[0] # first component
    dict_measures[f'pcs'] = pca.explained_variance_ratio_ # all components
    dict_measures[f'tau'] = tau # decay constant
    dict_measures[f'ngsc'] = entropy(pca.explained_variance_ratio_) / np.log2(pca.n_components_) # Shannon's entropy; measure of information contained by the system
    # plotting for frame
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(pca.explained_variance_ratio_)
        ax.set_xlabel('Component number')
        ax.set_ylabel(f'Variance explained')
        ax.set_title(f'Variance explained per component for video segment')
    return pca, dict_measures

def run_ipca(energymatrix, components, plot = False):
    """
    Runs Incremental Principal Component Analysis (PCA) for each video segment.

    Reshapes the energy matrix so that each row represents a frame (height x width pixels) and min-max normalizes the matrix.
    Computes Principal Components via Singular Value Decomposition in batches to optimize comptutation.
    Fits an exponential curve to the explained variance to inspect individual components' explained variance contribution. 
    
    
    Parameters
    ----------
    energymatrix : numpy.ndarray
        Array of shape (n_frames, height, width) containing motion energy differences in a video segment
    
    components: integer
        Number of components to extract from the video. Default is 200 based on observation.
    plot: boolean
        Enables plotting of explained variance curve.

    Returns
    -------
    pca: sklearn.IPCA()
        a PCA object containing the results of the PCA analysis.
    
    dict
        A dictionary containing the first principle component's variance explained,
        the variance explained by all components, the tau (exponential decay rate of the 
        PCA spectrum), and the normalized spectral entropy.        
    """

    n_frames = energymatrix.shape[0]

    X = energymatrix.reshape(n_frames, -1)

    # setting up the pca: normalization and fit
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    pca = IncrementalPCA(components)
    pca.fit(X_norm)

    # computing curves of variance decay
    [tau], _ = scipy.optimize.curve_fit(exponential, np.arange(pca.n_components_), pca.explained_variance_ratio_/pca.explained_variance_ratio_[0])

    dict_measures = {}
    dict_measures[f'pc1'] = pca.explained_variance_ratio_[0] # first component
    dict_measures[f'pcs'] = pca.explained_variance_ratio_ # all components
    dict_measures[f'tau'] = tau # decay constant
    dict_measures[f'ngsc'] = entropy(pca.explained_variance_ratio_) / np.log2(pca.n_components_) # Shannon's entropy; measure of information contained by the system
    # plotting for frame
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(pca.explained_variance_ratio_)
        ax.set_xlabel('Component number')
        ax.set_ylabel(f'Variance explained')
        ax.set_title(f'Variance explained per component for video segment')
    return pca, dict_measures

 # test functions
def get_80comp(pca: PCA):
    
    variance = np.cumsum(pca.explained_variance_)
    comp80 = np.searchsorted(variance, 0.8) + 1

    return pca.components_[:comp80]