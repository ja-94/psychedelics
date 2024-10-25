## Da Davide from labbox






def mtcsd(x, fs=1, nperseg=None, nfft=None, noverlap=None, nw=3, ntapers=None,
          detrend_method='constant'):
    """
    Pair-wise cross-spectral density using Slepian tapers. Adapted from the
    mtcsd function in the labbox Matlab toolbox (authors: Partha Mitra,
    Ken Harris).

    Parameters
    ----------
    x : ndarray
        2D array of signals across which to compute CSD, columns treated as
        channels
    fs : float (default = 1)
        sampling frequency
    nperseg : int, None (default = None)
        number of data points per segment, if None nperseg is set to 256
    nfft : int, None (default = None)
        number of points to include in scipy.fft.fft, if None nfft is set to
        2 * nperseg, if nfft > nperseg data will be zero-padded
    noverlap : int, None (default = None)
        amout of overlap between consecutive segments, if None noverlap is set
        to nperseg / 2
    nw : int (default = 3)
        time-frequency bandwidth for Slepian tapers, passed on to
        scipy.signal.windows.dpss
    ntapers : int, None (default = None)
        number of tapers, passed on to scipy.signal.windows.dpss, if None
        ntapers is set to nw * 2 - 1 (as suggested by original authors)
    detrend_method : {'constant', 'linear'} (default = 'constant')
        method used by scipy.signal.detrend to detrend each segment

    Returns
    -------
    f : ndarray
        frequency bins
    csd : ndarray
        full cross-spectral density matrix
    """
    # allow single channel input
    if x.ndim == 1:
        x = x[:, np.newaxis]

    # ensure no more than 2D input
    assert x.ndim == 2

    # set some default for parameters values
    if nperseg is None:
        nperseg = 256

    if nfft is None:
        nfft = nperseg * 2

    if noverlap is None:
        noverlap = nperseg / 2

    if ntapers is None:
        ntapers = 2 * nw - 1

    # get step size and total number of segments
    stepsize = nperseg - noverlap
    nsegs = int(np.floor(len(x) / stepsize))

    # initialize csd matrix
    csd = np.zeros((x.shape[1], x.shape[1], nfft), dtype='complex128')

    # get FFT frequency bins
    f = fft.fftfreq(nfft, 1/fs)

    # get tapers
    tapers = windows.dpss(nperseg, nw, Kmax=ntapers)

    # loop over segments
    for seg_ind in range(nsegs):

        # prepare segment
        i0 = int(seg_ind * stepsize)
        i1 = int(seg_ind * stepsize + nperseg)
        if i1 > len(x): # stop if segment is out of range of data
            nsegs -= (nsegs - seg_ind) # reduce segment count
            break
        seg = x[i0:i1, :]
        seg = detrend(seg, type=detrend_method, axis=0)

        # apply tapers
        tapered_seg = np.full((len(tapers), seg.shape[0], seg.shape[1]), np.nan)
        for taper_ind, taper in enumerate(tapers):
            tapered_seg[taper_ind] = (seg.T * taper).T

        # compute FFT for each channel-taper combination
        fftnorm = np.sqrt(2) # value taken from original matlab function
        pxx = fft.fft(tapered_seg, n=nfft, axis=1) / fftnorm

        # fill upper triangle of csd matrix
        for ch1 in range(x.shape[1]): # loop over unique channel combinations
            for ch2 in range(ch1, x.shape[1]):
                # compute csd bewteen channels, summing over tapers and segments
                csd[ch1, ch2, :] += (pxx[:, :, ch1] * np.conjugate(pxx[:, :, ch2])).sum(axis=0)

    # normalize csd by number of taper-segment combinations
    # (equivalent to averaging over segments and tapers)
    csdnorm = ntapers * nsegs
    csd /= csdnorm

    # fill lower triangle of csd matrix with complex conjugate of upper triangle
    for ch1 in range(x.shape[1]):
        for ch2 in range(ch1 + 1, x.shape[1]):
            csd[ch2, ch1, :] = np.conjugate(csd[ch1, ch2, :])

    return f, csd

def mtcoh(x, **kwargs):
    """
    Pair-wise multi-taper coherence for a set of signals.

    Parameters
    ----------
    See mtcsd documentation.

    Returns
    -------
    f : ndarray
        frequency bins
    coh : ndarray
        full spectral coherence matrix
    """
    # Compute cross-spectral density
    f, csd = mtcsd(x, **kwargs)
    # Compute power normalization matrix
    powernorm = np.zeros((x.shape[1], x.shape[1], len(f)))
    for ch1 in range(x.shape[1]):
        for ch2 in range(x.shape[1]):
            powernorm[ch1, ch2] = np.sqrt(np.abs(csd[ch1, ch1]) * np.abs(csd[ch2, ch2]))
    # Normalize CSD to get coherence
    coh = np.abs(csd) ** 2 / powernorm
    # Return frequency array, coherence, and phase differences
    return f, coh, np.angle(csd)