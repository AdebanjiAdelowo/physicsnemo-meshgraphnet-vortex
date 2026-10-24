"""Long-horizon wake diagnostics: probes, spectra, phase, wake profiles and threshold times.

Signals are NumPy arrays over rollout steps (step 1 is index 0). The spectral estimator is
the one of ``audit/probe_spectrum.py``: linear trend removed, Hann window, periodogram,
band of 0.7 to 1.3 times the reference shedding frequency. It is used unchanged for every
run so that the nine runs are compared with the same probes, windows and normalisation.
"""

import numpy as np

# probe name -> (x, y) offset from the cylinder centre, in diameters
PROBES = {
    "c1D": (1.0, 0.0),
    "c2D": (2.0, 0.0),
    "c3D": (3.0, 0.0),  # the probe of the earlier audit
    "c5D": (5.0, 0.0),
    "c8D": (8.0, 0.0),
    "u3D": (3.0, 0.5),
    "l3D": (3.0, -0.5),
}
MAIN_PROBE = "c3D"
LINE_X_OVER_D = tuple(np.round(np.arange(0.75, 12.01, 0.25), 2))
# rollout step windows, inclusive; the training window ends at step 299
WINDOWS = {"early": (1, 99), "late_train": (100, 299), "extrapolation": (300, 599)}
SPECTRAL_WINDOWS = ("late_train", "extrapolation")  # steps 1-99 hold < 2 periods for the slow trajectories
BAND = (0.7, 1.3)
SLIDING = 60  # steps; at least one reference shedding period (25 to 60 steps)
PRESERVED_POWER = 0.25  # shedding counted as preserved: predicted peak inside BAND and band power ratio >= this


def wake_geometry(pos: np.ndarray, wall: np.ndarray, interior: np.ndarray, height: float = 0.41) -> dict:
    """Cylinder centre and diameter from the wall nodes off the channel walls, and probe nodes.

    Probes and centreline points are the interior nodes nearest to the target points; targets
    outside the channel are dropped from the centreline (index -1 and NaN position).
    """
    pos = pos.astype(np.float64)
    walls = pos[wall]
    cyl = walls[(walls[:, 1] > 1e-6) & (walls[:, 1] < height - 1e-6)]
    centre = cyl.mean(axis=0)
    diameter = 2.0 * np.linalg.norm(cyl - centre, axis=1).mean()
    nodes = np.where(interior)[0]
    x_max = pos[:, 0].max()

    def nearest(target):
        return int(nodes[np.argmin(np.linalg.norm(pos[nodes] - target, axis=1))])

    probe_nodes = np.array([nearest(centre + diameter * np.asarray(off)) for off in PROBES.values()])
    line_nodes = np.array([nearest(centre + [diameter * x, 0.0]) if centre[0] + diameter * x < x_max else -1 for x in LINE_X_OVER_D])
    line_pos = np.where(line_nodes[:, None] >= 0, pos[np.maximum(line_nodes, 0)], np.nan)
    return {"centre": centre, "diameter": diameter, "probe_nodes": probe_nodes, "probe_pos": pos[probe_nodes],
            "line_nodes": line_nodes, "line_pos": line_pos}


def window_unsteadiness(velocity: np.ndarray, weights: np.ndarray, region: np.ndarray) -> float:
    """Area-weighted RMS of the fluctuation about the window mean over the RMS of the velocity, in ``region``."""
    v = velocity[:, region].astype(np.float64)
    w = weights[region]
    fluct = v - v.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.einsum("tnc,n->", fluct**2, w) / np.einsum("tnc,n->", v**2, w)))


def detrend(sig: np.ndarray) -> np.ndarray:
    t = np.arange(len(sig))
    return sig - np.polyval(np.polyfit(t, sig, 1), t)


def spectrum(sig: np.ndarray) -> np.ndarray:
    """Periodogram of the detrended, Hann-windowed signal (bins k = 0 .. n/2, k cycles per window)."""
    return np.abs(np.fft.rfft(detrend(sig) * np.hanning(len(sig)))) ** 2


def coefficients(sig: np.ndarray) -> np.ndarray:
    return np.fft.rfft(detrend(sig) * np.hanning(len(sig)))


def peak_bin(sig: np.ndarray) -> int:
    """Index of the largest non-zero-frequency bin of the periodogram."""
    return int(np.argmax(spectrum(sig)[1:]) + 1)


def peak_frequency(sig: np.ndarray, pad: int = 16) -> float:
    """Dominant frequency in cycles per step, from the zero-padded periodogram with parabolic refinement."""
    n = len(sig)
    spec = np.abs(np.fft.rfft(detrend(sig) * np.hanning(n), n * pad)) ** 2
    k = int(np.argmax(spec[pad:]) + pad)  # skip the bins below one cycle per window
    if 0 < k < len(spec) - 1:
        a, b, c = np.log(spec[k - 1 : k + 2] + 1e-300)
        k = k + 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else k
    return float(k / (n * pad))


def band(k_ref: int) -> slice:
    return slice(max(1, int(BAND[0] * k_ref)), int(np.ceil(BAND[1] * k_ref)) + 1)


def band_power_ratio(pred: np.ndarray, true: np.ndarray, k_ref: int) -> float:
    """Predicted power in the reference shedding band over the reference power in that band."""
    b = band(k_ref)
    return float(spectrum(pred)[b].sum() / spectrum(true)[b].sum())


def band_fraction(sig: np.ndarray, k_ref: int) -> float:
    """Share of the signal's fluctuation power (k >= 1) that lies in the shedding band."""
    spec = spectrum(sig)
    return float(spec[band(k_ref)].sum() / spec[1:].sum())


def phase_at(pred: np.ndarray, true: np.ndarray, k_ref: int) -> float:
    """Phase of the prediction minus that of the reference at the reference peak bin, in degrees (-180, 180]."""
    cp, ct = coefficients(pred)[k_ref], coefficients(true)[k_ref]
    return float(np.degrees(np.angle(cp * np.conj(ct))))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = detrend(a), detrend(b)
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else float("nan")


def best_lag_correlation(pred: np.ndarray, true: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag s (|s| <= max_lag) maximising corr(pred(t), true(t + s)) and that correlation."""
    best = (0, -np.inf)
    n = len(pred)
    for s in range(-max_lag, max_lag + 1):
        a = pred[max(0, -s) : n - max(0, s)]
        b = true[max(0, s) : n - max(0, -s)]
        c = correlation(a, b)
        if c > best[1]:
            best = (s, c)
    return best


def sliding(fn, pred: np.ndarray, true: np.ndarray, width: int = SLIDING) -> np.ndarray:
    """``fn(pred_window, true_window)`` for windows ending at every step from ``width`` on (NaN before)."""
    out = np.full(len(pred), np.nan)
    for end in range(width, len(pred) + 1):
        out[end - 1] = fn(pred[end - width : end], true[end - width : end])
    return out


def sliding_amplitude_ratio(pred: np.ndarray, true: np.ndarray, width: int = SLIDING) -> np.ndarray:
    return sliding(lambda a, b: detrend(a).std() / detrend(b).std(), pred, true, width)


def sliding_correlation(pred: np.ndarray, true: np.ndarray, width: int = SLIDING) -> np.ndarray:
    return sliding(correlation, pred, true, width)


def shedding_preserved(pred: np.ndarray, true: np.ndarray) -> bool:
    """Predicted dominant frequency inside the reference band and at least PRESERVED_POWER of its band power."""
    ratio = peak_frequency(pred) / peak_frequency(true)
    return bool(BAND[0] <= ratio <= BAND[1] and band_power_ratio(pred, true, peak_bin(true)) >= PRESERVED_POWER)


def demodulate(sig: np.ndarray, f_ref: float, order: int = 4) -> np.ndarray:
    """Complex amplitude of the detrended signal at ``f_ref``: shift ``f_ref`` to zero and low-pass.

    The low-pass (zero-phase Butterworth, cut-off 0.3 f_ref) keeps the band of BAND around the
    reference frequency. For a pure sinusoid the amplitude is within 10 % and the phase within
    5 degrees of the truth outside the first and last period (``tests/test_diagnostics.py``).
    """
    from scipy.signal import butter, sosfiltfilt

    t = np.arange(len(sig))
    z = detrend(sig) * np.exp(-2j * np.pi * f_ref * t)
    sos = butter(order, (1.0 - BAND[0]) * f_ref / 0.5, output="sos")
    return sosfiltfilt(sos, z.real, padtype="constant") + 1j * sosfiltfilt(sos, z.imag, padtype="constant")


def phase_tracking(pred: np.ndarray, true: np.ndarray, f_ref: float) -> dict:
    """In-band amplitude ratio and phase difference of prediction and reference over time.

    ``envelope_ratio`` = |A_pred| / |A_true| with A the complex amplitude at the reference shedding
    frequency (``demodulate``); ``phase_deg`` = phase of the prediction minus that of the
    reference, wrapped to [-180, 180) degrees, NaN wherever the prediction holds less than half
    of the reference's in-band amplitude (no shedding to phase). Both are NaN over the first and
    last reference period, where the filter edges distort them.
    """
    zp, zt = demodulate(pred, f_ref), demodulate(true, f_ref)
    ratio = np.abs(zp) / np.abs(zt)
    dphi = (np.degrees(np.angle(zp * np.conj(zt))) + 180.0) % 360.0 - 180.0
    edge = int(round(1.0 / f_ref))
    ratio[:edge], ratio[len(ratio) - edge :] = np.nan, np.nan
    return {"envelope_ratio": ratio, "phase_deg": np.where(ratio >= 0.5, dphi, np.nan)}


def first_below(series: np.ndarray, threshold: float, start: int = 0) -> int:
    """First index from ``start`` with a finite value below the threshold; -1 if never."""
    s = series[start:]
    hit = np.isfinite(s) & (s < threshold)
    return int(np.argmax(hit) + start) if hit.any() else -1


def first_above(series: np.ndarray, threshold: np.ndarray | float, start: int = 0) -> int:
    """First index from ``start`` with a value above the threshold or not finite; -1 if never."""
    s = series[start:]
    th = threshold[start:] if np.ndim(threshold) else threshold
    hit = ~np.isfinite(s) | (s > th)
    return int(np.argmax(hit) + start) if hit.any() else -1


def log_growth_per_100(series: np.ndarray, lo: int, hi: int) -> float:
    """Least-squares slope of log(series) over steps lo..hi (inclusive, 1-based), per 100 steps.

    0.69 per 100 steps is a doubling every 100 steps. NaN if the window holds non-finite or
    non-positive values.
    """
    seg = series[lo - 1 : hi].astype(np.float64)
    if not np.all(np.isfinite(seg)) or np.any(seg <= 0):
        return float("nan")
    return float(100 * np.polyfit(np.arange(lo, hi + 1), np.log(seg), 1)[0])
