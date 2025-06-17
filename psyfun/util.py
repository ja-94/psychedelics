import numpy as np
from scipy import stats

def exponential_decay(x, alpha, beta, tau):
    return  alpha * np.exp(-x / tau) + beta

def bi_exponential_decay(x, tau1, tau2, p):
    return p * np.exp(-x / tau1) + (1 - p) * np.exp(-x / tau2)

def _get_exp_tau(eig):
    xx = np.arange(len(eig))
    yy = (eig - eig.min()) / (eig.max() - eig.min())
    try:
        (tau), pcov = curve_fit(lambda x, tau: exponential_decay(x, 1, 0, tau), xx, yy)
    except:
        tau = np.nan
    return tau

def _get_biexp_mean_lifetime(eig):
    xx = np.arange(len(eig))
    yy = (eig - eig.min()) / (eig.max() - eig.min())
    try:
        (tau1, tau2, p), pcov = curve_fit(bi_exponential_decay, xx, yy)
        mean_lifetime = p * tau1 + (1 - p) * tau2
    except:
        mean_lifetime = np.nan
    return mean_lifetime

def power_law_slope(eig, rank=100):
    if len(eig) < rank:
        rank = len(eig)
    xx = np.arange(rank) + 1
    res = stats.linregress(np.log(xx), np.log(eig[:rank]))
    return res.slope

def ngsc(a):
    return stats.entropy(a) / np.log2(len(a))

def angle(v1, v2):
    """
    Compute the angle between two vectors v1 and v2.
    
    Parameters
    ----------
    v1, v2 : array_like
        The two vectors (e.g. PC1 directions) between which to compute the angle.
    
    Returns
    -------
    angle_rad : float
        Angle in radians.
    angle_deg : float
        Angle in degrees.
    """
    # Compute dot product and norms.
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Avoid numerical issues: clip the cosine value to the valid range [-1, 1]
    cos_theta = np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    # angle_deg = np.degrees(angle_rad)
    return angle_rad