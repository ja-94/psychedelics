from functools import wraps
import numpy as np
from antropy import lziv_complexity


def spike_train_metric(metric_func):
    """
    Decorator that filters spike times to a time window before passing to metric.
    """
    @wraps(metric_func)
    def wrapper(ts, t0=None, t1=None, **kwargs):
        # Filter to time window if specified
        if t0 is not None and t1 is not None:
            ts = ts[(ts >= t0) & (ts <= t1)]
        return metric_func(ts, **kwargs)
    return wrapper


@spike_train_metric
def coefficient_of_variation(ts, **kwargs):
    if len(ts) < 2:
        return np.nan
    isis = np.diff(ts)
    if isis.mean() == 0:
        return np.nan
    return isis.std() / isis.mean()


def spike_counts(ts, t0, t1, dt=1):
    counts, tpts = np.histogram(ts, bins=np.arange(t0, t1, dt))
    return counts, tpts


def spike_count_metric(metric_func):
    """
    Decorator that allows functions to handle both spike times and spike counts.
    Use from_counts=True to bypass this conversion.
    """
    @wraps(metric_func)
    def wrapper(input_data, from_counts=False, t0=None, t1=None, dt=1, **metric_kwargs):
        if from_counts:
            return metric_func(input_data, **metric_kwargs)
        else:
            counts, _ = spike_counts(input_data, t0, t1, dt=dt)
            return metric_func(counts, **metric_kwargs)
    return wrapper


@spike_count_metric
def mean_rate(counts):
    """Mean firing rate from spike counts"""
    return counts.mean()


@spike_count_metric
def fano_factor(counts):
    """Fano factor (variance/mean) of spike counts"""
    mean = counts.mean()
    variance = counts.var()
    if mean == 0:
        return np.nan
    return variance / mean


@spike_count_metric
def lempel_ziv_complexity(counts, bin_thr=0.5):
    """Lempel-Ziv complexity from spike counts"""
    # Convert counts to binary string (spike/no spike)
    if isinstance(bin_thr, float):
        bin_thr = np.quantile(counts, bin_thr)
    else:
        raise NotImplementedError
    binary_sequence = ''.join(['1' if c > bin_thr else '0' for c in counts])
    return lziv_complexity(binary_sequence, normalize=True)


def shuffle_spikes(ts, rng):
    isi = np.diff(ts)
    return np.insert(ts[0] + rng.permutation(isi).cumsum(), 0, ts[0])


def modulation_index(pre, post):
    if post + pre == 0:
        return np.nan
    return (post - pre) / (post + pre)


def _apply_spike_counts(series, epoch, dt=1):
    if np.isnan(series[f'{epoch}_start']) or np.isnan(series[f'{epoch}_stop']):
        return np.array(np.nan)
    ts = series['spike_times']
    t0 = series[f'{epoch}_start']
    t1 = series[f'{epoch}_stop']
    ts, _ = spike_counts(ts, t0, t1, dt=dt)
    return ts


def _apply_modulation_index(
    series, epochs, metrics,
    n_shf=1000, shf_fun=shuffle_spikes, return_permuted=False
    ):
    res = {}
    ts = series['spike_times']
    if len(ts) <= 2:
        return res
    if n_shf:  # pre-generate shuffled spike trains for permutation testing
        rng = np.random.default_rng(42)  # rng outside of function !!
        ts_shf = np.array([shf_fun(ts, rng) for i in range(n_shf)])
    t0s = [series[f'{epoch}_start'] for epoch in epochs]
    t1s = [series[f'{epoch}_stop'] for epoch in epochs]
    for metric, metric_kwargs in metrics.items():
        for epoch, t0, t1 in zip(epochs, t0s, t1s):
            # ~ts_epoch = ts[(ts >= t0) & (ts <= t1)]
            metric_kwargs['t0'] = t0
            metric_kwargs['t1'] = t1
            res[f'{metric.__name__}_{epoch.split("_")[0]}'] = metric(
                ts, **metric_kwargs
                )
        mi = modulation_index(
            res[f'{metric.__name__}_{epochs[0].split("_")[0]}'],
            res[f'{metric.__name__}_{epochs[1].split("_")[0]}'],
            )
        res[f'{metric.__name__}_MI'] = mi
        if np.isnan(mi):
            continue
        if n_shf:
            mi_shf = np.full(n_shf, np.nan)
            for i, ts in enumerate(ts_shf):
                pre_post_vals = []
                for t0, t1 in zip(t0s, t1s):
                    metric_kwargs['t0'] = t0
                    metric_kwargs['t1'] = t1
                    pre_post_vals.append(metric(ts, **metric_kwargs))
                mi_shf[i] = modulation_index(*pre_post_vals)
            if np.isnan(mi_shf).all():
                print(
                    "WARNING: all permuted MIs are NaN for neuron "
                    f"{series.name} {metric.__name__}"
                    )
                continue
            elif np.isnan(mi_shf).mean() > 0.1:
                print(
                    "WARNING: >10% of permuted MIs are NaN for neuron "
                    f"{series.name} {metric.__name__}"
                    )
            res[f'{metric.__name__}_MIp'] = np.nanmean(mi > mi_shf)
            if return_permuted:
                res[f'{metric.__name__}_MIshf'] = mi_shf
    return res


def rate_modulation_index(ts, t0s, t1s):
    assert len(t0s) == 2
    assert len(t1s) == 2
    rates = np.array([np.sum(((ts >= t0) & (ts <= t1)), axis=-1) for t0, t1 in zip(t0s, t1s)])
    return np.squeeze(np.diff(rates, axis=0) / np.sum(rates, axis=0))

def _apply_rate_modulation_index(series, epoch_pairs=None, n_shf=100):
    assert epoch_pairs is not None
    ts = series['spike_times']
    rng = np.random.default_rng(42)
    ts_shf = np.array([shuffle_spikes(ts, rng) for i in range(n_shf)])
    MIs = {}
    for label, epochs in epoch_pairs.items():
        t0s = [series[f'{epoch}_start'] for epoch in epochs]
        t1s = [series[f'{epoch}_stop'] for epoch in epochs]
        # Compute the modulation index
        mi = rate_modulation_index(ts, t0s, t1s)
        # Permutation test
        mi_shf = rate_modulation_index(ts_shf, t0s, t1s)
        p = (mi > mi_shf).mean()
        MIs[f'{label}_MI'] = float(mi)  # return single number when only applied to single spike train
        MIs[f'{label}_p'] = p
    return MIs
