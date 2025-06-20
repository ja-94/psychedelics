import numpy as np

def spike_counts(ts, t0, t1, dt=1):
    counts, tpts = np.histogram(ts, bins=np.arange(t0, t1 + dt, dt))
    return counts, tpts

def _get_spike_counts(series, epoch=None, **kwargs):
    assert epoch is not None
    if np.isnan(series[f'{epoch}_start']) or np.isnan(series[f'{epoch}_stop']):
        return np.array(np.nan)
    ts = series['spike_times']
    t0 = series[f'{epoch}_start']
    t1 = series[f'{epoch}_stop']
    return spike_counts(ts, t0, t1, **kwargs)[0]

def shuffle_spikes(ts, rng):
    isi = np.diff(ts)
    return np.insert(ts[0] + rng.permutation(isi).cumsum(), 0, ts[0])

def modulation_index(ts, t0s, t1s):
    assert len(t0s) == 2
    assert len(t1s) == 2
    rates = np.array([np.sum(((ts >= t0) & (ts <= t1)), axis=-1) for t0, t1 in zip(t0s, t1s)])
    return np.squeeze(np.diff(rates, axis=0) / np.sum(rates, axis=0))

def _get_modulation_index(series, epoch_pairs=None, n_shf=100):
    assert epoch_pairs is not None
    ts = series['spike_times']
    rng = np.random.default_rng(42)
    ts_shf = np.array([shuffle_spikes(ts, rng) for i in range(n_shf)])
    MIs = {}
    for label, epochs in epoch_pairs.items():
        t0s = [series[f'{epoch}_start'] for epoch in epochs]
        t1s = [series[f'{epoch}_stop'] for epoch in epochs]
        # Compute the modulation index
        mi = modulation_index(ts, t0s, t1s)
        # Permutation test
        mi_shf = modulation_index(ts_shf, t0s, t1s)
        p = (mi > mi_shf).mean()
        MIs[f'{label}_MI'] = mi
        MIs[f'{label}_p'] = p
    return MIs