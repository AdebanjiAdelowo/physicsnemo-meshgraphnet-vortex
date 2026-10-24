"""Phase-sensitive diagnostics from the probe signals written by independent_rollout.py.

The probe is the interior node nearest to three diameters downstream of the cylinder centre;
the signal is the cross-stream velocity v. For each window the linear trend is removed, and the
dominant period (FFT peak of the Hann-windowed signal), the RMS amplitude, the spectral power in
the band of the reference shedding frequency (0.7 to 1.3 times it) relative to the reference's
power in that band, and the normalised cross-correlation at zero lag are reported. A predicted
signal without shedding has a band-power ratio near zero.

usage: probe_spectrum.py <indep.npz> [<indep.npz> ...]
"""
import sys
from collections import defaultdict

import numpy as np


def detrend(sig):
    t = np.arange(len(sig))
    return sig - np.polyval(np.polyfit(t, sig, 1), t)


def spectrum(sig):
    return np.abs(np.fft.rfft(detrend(sig) * np.hanning(len(sig)))) ** 2


def period_and_amp(sig):
    spec = spectrum(sig)
    k = np.argmax(spec[1:]) + 1
    return len(sig) / k, detrend(sig).std(), k


def band_power_ratio(pred, true, k_ref):
    band = slice(max(1, int(0.7 * k_ref)), int(np.ceil(1.3 * k_ref)) + 1)
    return spectrum(pred)[band].sum() / spectrum(true)[band].sum()


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


for path in sys.argv[1:]:
    data = np.load(path)
    groups = defaultdict(dict)
    for key in data.files:
        run, traj, name = key.split("/")
        groups[(run, int(traj))][name] = data[key]
    for (run, traj), d in sorted(groups.items()):
        line = f"{run} traj {traj}:"
        if "probe_pred" in d:
            for lo, hi in ((100, 299), (300, 599)):
                true, pred = d["probe_true"][lo - 1 : hi], d["probe_pred"][lo - 1 : hi]
                pt, at, kt = period_and_amp(true)
                pp, ap, _ = period_and_amp(pred)
                c = corr(detrend(pred), detrend(true))
                band = band_power_ratio(pred, true, kt)
                line += f" steps {lo}-{hi}: period ref {pt:.1f} pred {pp:.1f}, rms ratio {ap / at:.2f}, band power ratio {band:.3f}, corr {c:+.2f};"
        tf_err, tf_inc = d["tf"]
        line += f" teacher-forced increment error {np.sqrt(tf_err / tf_inc):.3f}; boundary max |err| {float(d['boundary_max']):.1e}"
        print(line)
