import numpy as np
from scipy.signal import welch, iirnotch, filtfilt, detrend, butter, sosfilt, hilbert
from scipy.stats import skew, kurtosis

from core import CHANNELS, FS

# Pre-compute filter coefficients
_b60, _a60 = iirnotch(60.0, 200.0, FS)

# V2 filter bank (Butterworth 4th order)
_sos_bandpass = butter(4, [0.5, 100], btype='band', fs=FS, output='sos')   # full band
_sos_eog = butter(4, [0.5, 10], btype='band', fs=FS, output='sos')        # EOG/slow potentials — blinks
_sos_mu = butter(4, [8, 30], btype='band', fs=FS, output='sos')           # alpha-beta — neural
_sos_emg = butter(4, [30, 100], btype='band', fs=FS, output='sos')        # EMG — muscle contraction
_b60_narrow, _a60_narrow = iirnotch(60.0, 30.0, FS)                       # narrow notch for v2


def preprocess_window(window):
    """Preprocess a (128, 4) raw µV window: notch 60Hz + detrend + z-score."""
    out = np.zeros_like(window)
    for i in range(4):
        sig = window[:, i].copy()
        sig = filtfilt(_b60, _a60, sig)
        sig = detrend(sig)
        mu, std = sig.mean(), sig.std()
        if std > 0:
            sig = (sig - mu) / std
        out[:, i] = sig
    return out


def preprocess_df(df):
    """Preprocess an entire DataFrame in-place: notch 60Hz + detrend + z-score."""
    for ch in CHANNELS:
        sig = df[ch].values.copy()
        sig = filtfilt(_b60, _a60, sig)
        sig = detrend(sig)
        mu, std = sig.mean(), sig.std()
        if std > 0:
            sig = (sig - mu) / std
        df[ch] = sig
    return df


def band_power(signal, fs, fmin, fmax):
    """Compute power in a frequency band using Welch's method."""
    nperseg = min(len(signal), 128)
    if nperseg < 4:
        return 0.0
    f, pxx = welch(signal, fs=fs, nperseg=nperseg)
    mask = (f >= fmin) & (f <= fmax)
    return np.trapezoid(pxx[mask], f[mask]) if mask.any() else 0.0


def zero_crossing_rate(signal):
    """Count zero crossings normalized by signal length."""
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


def max_derivative(signal):
    """Maximum absolute first derivative."""
    return np.max(np.abs(np.diff(signal))) if len(signal) > 1 else 0.0


# ─── Basic features (compatible with blink pipeline) ───

def extract_features_basic(window):
    """Extract 20 basic features from a (128, 4) window. Same as blink pipeline."""
    feats = []
    for i in range(4):
        sig = window[:, i]
        feats.extend([
            np.mean(np.abs(sig)),
            np.std(sig),
            np.ptp(sig),
            band_power(sig, FS, 0.5, 5.0),
            band_power(sig, FS, 8.0, 12.0),
        ])
    return feats


BASIC_FEATURE_NAMES = []
for _ch in CHANNELS:
    for _f in ["mean_abs", "std", "ptp", "power_0.5_5", "power_8_12"]:
        BASIC_FEATURE_NAMES.append(f"{_ch}_{_f}")


# ─── Extended features (for multi-gesture classification) ───

def extract_features_extended(window):
    """Extract ~44 features from a (128, 4) window for multi-gesture classification.

    Includes basic features plus:
    - EMG band power (20-100Hz) per channel — key gesture discriminator
    - Skewness and kurtosis per channel — waveform shape
    - Zero-crossing rate per channel — frequency characteristics
    - Max derivative per channel — transient sharpness
    - Frontal/temporal ratio — spatial gesture signature
    - AF7/AF8 asymmetry
    """
    feats = []

    # Per-channel features (11 per channel = 44)
    for i in range(4):
        sig = window[:, i]
        feats.extend([
            np.mean(np.abs(sig)),               # mean absolute
            np.std(sig),                         # standard deviation
            np.ptp(sig),                         # peak-to-peak
            band_power(sig, FS, 0.5, 5.0),      # delta power
            band_power(sig, FS, 8.0, 12.0),     # alpha power
            band_power(sig, FS, 20.0, 100.0),   # EMG band power
            float(skew(sig)),                    # skewness
            float(kurtosis(sig)),                # kurtosis
            zero_crossing_rate(sig),             # zero-crossing rate
            max_derivative(sig),                 # max |diff|
        ])

    # Cross-channel ratios (3 features)
    eps = 1e-8
    frontal_power = np.mean(np.abs(window[:, 1])) + np.mean(np.abs(window[:, 2]))  # AF7 + AF8
    temporal_power = np.mean(np.abs(window[:, 0])) + np.mean(np.abs(window[:, 3]))  # TP9 + TP10

    feats.append(frontal_power / (temporal_power + eps))   # frontal ratio
    feats.append(temporal_power / (frontal_power + eps))   # temporal ratio

    af7_power = np.mean(np.abs(window[:, 1]))
    af8_power = np.mean(np.abs(window[:, 2]))
    feats.append((af7_power - af8_power) / (af7_power + af8_power + eps))  # asymmetry

    return feats


EXTENDED_FEATURE_NAMES = []
for _ch in CHANNELS:
    for _f in ["mean_abs", "std", "ptp", "power_delta", "power_alpha",
               "power_emg", "skewness", "kurtosis", "zcr", "max_deriv"]:
        EXTENDED_FEATURE_NAMES.append(f"{_ch}_{_f}")
EXTENDED_FEATURE_NAMES.extend(["frontal_ratio", "temporal_ratio", "af7_af8_asymmetry"])


# ─── V2 Pipeline: Multi-band, amplitude-preserving ───
#
# Key design decisions vs V1:
#   1. NO z-score normalization — preserves absolute amplitude differences
#   2. Multi-band decomposition: EOG (0.5-10Hz), mu/beta (8-30Hz), EMG (30-100Hz)
#   3. Clipping-aware: interpolates ADC-railed samples, flags bad windows
#   4. Baseline-referenced features: measures change vs pre-event baseline
#   5. Temporal dynamics: energy in window quarters captures impulse vs sustained

ADC_LIMIT = 15500  # µV, flag clipping above this


def interpolate_clipped(sig, limit=ADC_LIMIT):
    """Replace clipped samples (at ADC rail) with linear interpolation."""
    clipped = np.abs(sig) >= limit
    if not clipped.any():
        return sig.copy()
    out = sig.copy()
    good = np.where(~clipped)[0]
    if len(good) < 2:
        return out  # can't interpolate
    bad = np.where(clipped)[0]
    out[bad] = np.interp(bad, good, out[good])
    return out


def preprocess_v2(window):
    """V2 preprocessing: notch 60Hz + bandpass 0.5-100Hz, NO z-score.

    Input:  (N, 4) raw µV window
    Output: (N, 4) filtered window preserving absolute amplitudes
    """
    out = np.zeros_like(window, dtype=np.float64)
    for i in range(4):
        sig = window[:, i].astype(np.float64)
        sig = interpolate_clipped(sig)
        sig = filtfilt(_b60_narrow, _a60_narrow, sig)  # notch 60Hz
        sig = sosfilt(_sos_bandpass, sig)               # bandpass 0.5-100Hz
        sig = detrend(sig)                              # remove linear drift
        out[:, i] = sig
    return out


def decompose_bands(sig):
    """Decompose a single-channel signal into 3 frequency bands.

    Returns: (eog, mu_beta, emg) — each same length as input.
    """
    eog = sosfilt(_sos_eog, sig)
    mu_beta = sosfilt(_sos_mu, sig)
    emg = sosfilt(_sos_emg, sig)
    return eog, mu_beta, emg


def extract_features_v2(window, baseline=None):
    """V2 feature extraction: multi-band, normalized within bands, ~30 features.

    Designed for small datasets (~25 events per class). Uses:
    - Multi-band decomposition with within-band normalization
    - Temporal energy distribution (quarter-based)
    - Cross-channel ratios (spatial signatures)
    - All features are ratios/normalized — no raw amplitudes

    Args:
        window:   (128, 4) preprocessed (v2) µV window centered on event
        baseline: unused (kept for API compat)

    Returns: list of float features
    """
    n_samples = window.shape[0]
    quarter = n_samples // 4
    feats = []
    eps = 1e-8

    # === Per-channel band features ===
    band_rms = {'eog': [], 'mu': [], 'emg': []}

    for ci in range(4):
        sig = window[:, ci]
        eog, mu, emg = decompose_bands(sig)

        for band_name, band_sig in [('eog', eog), ('mu', mu), ('emg', emg)]:
            rms = np.sqrt(np.mean(band_sig ** 2)) + eps
            band_rms[band_name].append(rms)

            # Temporal centroid: where energy concentrates (0=start, 1=end)
            energy = band_sig ** 2
            total_e = energy.sum() + eps
            tc = np.sum(np.arange(n_samples) * energy) / (n_samples * total_e)
            feats.append(tc)

        # Temporal energy distribution — normalized quarters (4 ratios that sum to 1)
        total_rms = np.sqrt(np.mean(sig ** 2)) + eps
        for q in range(4):
            q_slice = sig[q * quarter:(q + 1) * quarter]
            feats.append(np.sqrt(np.mean(q_slice ** 2)) / total_rms)

        # Transient sharpness: max |derivative| / std (scale-free)
        md = np.max(np.abs(np.diff(sig))) if n_samples > 1 else 0.0
        sd = np.std(sig) + eps
        feats.append(md / sd)

        # Band energy distribution: what fraction of total energy is in each band
        total_band = band_rms['eog'][-1] + band_rms['mu'][-1] + band_rms['emg'][-1]
        feats.append(band_rms['eog'][-1] / total_band)  # EOG fraction
        feats.append(band_rms['emg'][-1] / total_band)  # EMG fraction

    # === Cross-channel ratios (spatial signatures) ===
    for band_name in ['eog', 'mu', 'emg']:
        frontal = band_rms[band_name][1] + band_rms[band_name][2]
        temporal = band_rms[band_name][0] + band_rms[band_name][3]
        feats.append(frontal / (temporal + eps))

    # Left/right asymmetry
    for band_name in ['eog', 'mu', 'emg']:
        left = band_rms[band_name][1]
        right = band_rms[band_name][2]
        feats.append((left - right) / (left + right + eps))

    # Inter-channel correlations
    corr_af = np.corrcoef(window[:, 1], window[:, 2])[0, 1]
    corr_tp = np.corrcoef(window[:, 0], window[:, 3])[0, 1]
    feats.append(0.0 if np.isnan(corr_af) else corr_af)
    feats.append(0.0 if np.isnan(corr_tp) else corr_tp)

    return feats


# Build V2 feature names
V2_FEATURE_NAMES = []
for _ch in CHANNELS:
    for _band in ['eog', 'mu', 'emg']:
        V2_FEATURE_NAMES.append(f'{_ch}_{_band}_centroid')
    for _q in range(4):
        V2_FEATURE_NAMES.append(f'{_ch}_q{_q}_ratio')
    V2_FEATURE_NAMES.append(f'{_ch}_sharpness')
    V2_FEATURE_NAMES.append(f'{_ch}_eog_frac')
    V2_FEATURE_NAMES.append(f'{_ch}_emg_frac')

for _band in ['eog', 'mu', 'emg']:
    V2_FEATURE_NAMES.append(f'{_band}_frontal_ratio')
for _band in ['eog', 'mu', 'emg']:
    V2_FEATURE_NAMES.append(f'{_band}_lr_asymmetry')
V2_FEATURE_NAMES.extend(['corr_af7_af8', 'corr_tp9_tp10'])


# ─── BLE packet parsing ───

def raw_to_uv(val):
    """Convert raw 16-bit value to microvolts."""
    if val > 32767:
        return (val - 65536) * 0.48828125
    return val * 0.48828125


def parse_packet(data):
    """Parse a BLE EEG packet into a list of µV samples."""
    samples = []
    for i in range(2, len(data), 2):
        val = int.from_bytes(data[i:i+2], byteorder='big')
        samples.append(raw_to_uv(val))
    return samples
