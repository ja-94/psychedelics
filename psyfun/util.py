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

