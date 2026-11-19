import numpy as np
from scipy.signal import find_peaks

FS = 250.0
BAND_EDGES = [0.5, 3, 6, 9, 12, 16, 20, 30, 45, 60, 90, 125]

FEATURE_NAMES = (
    ["log_mad", "zcr", "line_length", "hjorth_mobility", "hjorth_complexity",
     "kurtosis", "skewness", "crest_factor"]
    + [f"bp_{BAND_EDGES[i]:g}_{BAND_EDGES[i+1]:g}" for i in range(len(BAND_EDGES) - 1)]
    + ["spec_centroid", "spec_entropy", "dom_freq", "spec_edge95",
       "ac_peak", "ac_lag_hz", "n_peaks", "rate_hz", "rr_mean", "rr_std", "rr_cv"]
)


def _robust_scale(x):
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    mad = np.maximum(mad, 1e-6)
    return (x - med) / mad, mad[:, 0]


def extract_features(x):
    """x: (N, 1250) float32 -> (N, n_features) float32."""
    x = np.asarray(x, np.float64)
    if x.ndim == 3:
        x = x[..., 0]
    n, T = x.shape
    xs, mad = _robust_scale(x)

    out = []
    out.append(np.log10(mad + 1e-9))                                   # log_mad (gain, kept for info)
    out.append(np.mean(np.diff(np.signbit(xs), axis=1), axis=1))       # zero-crossing rate
    d1 = np.diff(xs, axis=1)
    d2 = np.diff(d1, axis=1)
    out.append(np.mean(np.abs(d1), axis=1))                            # line length
    s0 = np.std(xs, axis=1) + 1e-9
    s1 = np.std(d1, axis=1) + 1e-9
    s2 = np.std(d2, axis=1) + 1e-9
    out.append(s1 / s0)                                                # Hjorth mobility
    out.append((s2 / s1) / (s1 / s0))                                  # Hjorth complexity
    z = (xs - xs.mean(1, keepdims=True)) / s0[:, None]
    out.append(np.mean(z ** 4, axis=1))                                # kurtosis
    out.append(np.mean(z ** 3, axis=1))                                # skewness
    out.append(np.max(np.abs(xs), axis=1) / s0)                        # crest factor

    # ---- spectrum
    win = np.hanning(T)[None, :]
    P = np.abs(np.fft.rfft(xs * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(T, 1.0 / FS)
    Ptot = P.sum(1, keepdims=True) + 1e-12
    Pn = P / Ptot
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        m = (freqs >= lo) & (freqs < hi)
        out.append(Pn[:, m].sum(1))                                    # relative band power
    out.append((Pn * freqs[None, :]).sum(1))                           # spectral centroid
    out.append(-(Pn * np.log(Pn + 1e-12)).sum(1))                      # spectral entropy
    band = (freqs >= 0.5) & (freqs <= 40)
    out.append(freqs[band][np.argmax(Pn[:, band], axis=1)])            # dominant frequency
    csum = np.cumsum(Pn, axis=1)
    out.append(freqs[np.argmax(csum >= 0.95, axis=1)])                 # 95% spectral edge

    # ---- autocorrelation periodicity, lags 25..250 samples = 1..10 Hz = 60..600 bpm
    xz = xs - xs.mean(1, keepdims=True)
    F = np.fft.rfft(xz, n=2 * T, axis=1)
    ac = np.fft.irfft(F * np.conj(F), axis=1)[:, :T]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = 25, 250
    seg = ac[:, lo:hi]
    out.append(seg.max(1))                                             # autocorr peak height
    out.append(FS / (lo + np.argmax(seg, axis=1)))                     # its frequency, Hz

    # ---- beat intervals (refractory-limited peak picking)
    npk = np.zeros(n); rate = np.zeros(n)
    rr_mean = np.zeros(n); rr_std = np.zeros(n); rr_cv = np.zeros(n)
    absx = np.abs(xs)
    thr = 0.4 * absx.max(1)
    for i in range(n):
        pk, _ = find_peaks(absx[i], height=thr[i], distance=int(0.15 * FS))
        npk[i] = len(pk)
        rate[i] = len(pk) / (T / FS)
        if len(pk) >= 3:
            rr = np.diff(pk) / FS
            rr_mean[i], rr_std[i] = rr.mean(), rr.std()
            rr_cv[i] = rr.std() / (rr.mean() + 1e-9)
        else:
            rr_mean[i] = rr_std[i] = rr_cv[i] = 0.0
    out += [npk, rate, rr_mean, rr_std, rr_cv]

    F = np.stack(out, axis=1).astype(np.float32)
    F[~np.isfinite(F)] = 0.0
    assert F.shape[1] == len(FEATURE_NAMES), (F.shape[1], len(FEATURE_NAMES))
    return F


# Features grouped by the physiological question each answers; used by the group ablation.
FEATURE_GROUPS = {
    "amplitude":        [0],                 # log_mad
    "time_domain":      list(range(1, 8)),   # zcr, line length, Hjorth x2, kurtosis, skew, crest
    "band_power":       list(range(8, 19)),  # 11 relative band powers
    "spectral_summary": list(range(19, 23)), # centroid, entropy, dominant freq, 95% edge
    "periodicity":      [23, 24],            # autocorrelation peak height and its frequency
    "beat_interval":    list(range(25, 30)), # peak count, rate, RR mean / std / CV
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == list(range(len(FEATURE_NAMES)))


def select_columns(drop=(), only=()):
    """Column indices to keep, given group names to drop or to keep exclusively."""
    if only:
        keep = sorted(sum((FEATURE_GROUPS[g] for g in only), []))
    else:
        dropped = set(sum((FEATURE_GROUPS[g] for g in drop), []))
        keep = [i for i in range(len(FEATURE_NAMES)) if i not in dropped]
    return keep


def cached_features(splits, cache_path="features_cache.npz"):
    """Extract once, reuse on every run (extraction is the slow part of the feature encoder)."""
    import os
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        if all(len(z[k]) == len(splits[k]["x"]) for k in ("train", "val", "test")):
            return {k: z[k] for k in ("train", "val", "test")}
    out = {k: extract_features(splits[k]["x"]) for k in ("train", "val", "test")}
    np.savez(cache_path, **out)
    return out


def fit_scaler(F):
    """Quantile scaler fitted on TRAIN only: maps each feature to roughly [0, 1]."""
    lo = np.percentile(F, 1, axis=0)
    hi = np.percentile(F, 99, axis=0)
    hi = np.where(hi - lo < 1e-9, lo + 1e-9, hi)
    return lo.astype(np.float32), hi.astype(np.float32)


def apply_scaler(F, scaler):
    lo, hi = scaler
    return np.clip((F - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    import data as D
    s = D.load_splits("cache_iegm.npz")
    F = extract_features(s["train"]["x"][:200])
    print("features:", F.shape, "| names:", len(FEATURE_NAMES))
    for name, col in zip(FEATURE_NAMES, F.T):
        print(f"  {name:22s} mean={col.mean():9.3f} std={col.std():8.3f}")
