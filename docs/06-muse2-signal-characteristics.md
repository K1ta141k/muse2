# Muse 2 Signal Characteristics

Empirical measurements from our data collection sessions.

## Hardware

- **Model:** Muse 2 (2016+)
- **Electrodes:** 4 dry EEG (TP9, AF7, AF8, TP10) + 1 reference (FPz)
- **Placement:** TP9/TP10 behind ears (temporal), AF7/AF8 on forehead (frontal)
- **ADC:** 16-bit, ±16,000 µV range
- **Sample rate:** 256 Hz (Nyquist: 128 Hz)

## Noise Floor

With dry electrodes, the noise floor is extremely high:

| Channel | Rest noise (std, µV) | Clipping (% samples > ±15,000) |
|---|---|---|
| TP9 | ~11,000 | ~20% |
| AF7 | ~10,400 | ~16% |
| AF8 | ~10,300 | ~13% |
| TP10 | ~10,500 | ~14% |

This means **12-20% of samples are at the ADC rail** even during rest. The signal regularly saturates.

## Gesture Signal Characteristics

### What the Muse 2 CAN detect

| Gesture | Signal type | Primary channels | Frequency band | Amplitude vs rest |
|---|---|---|---|---|
| Blink | EOG artifact (eyeball dipole) | AF7, AF8 | 0.5-10 Hz | 1.1-1.3x |
| Brow furrow | Corrugator EMG + slow shift | AF7, AF8 | 0.5-10 Hz | 1.1-1.2x |
| Brow raise | Frontalis EMG + slow shift | AF7, AF8 | 0.5-10 Hz | 1.0-1.2x |
| Jaw clench | Masseter EMG | TP9, TP10 | 8-30 Hz | 1.1x |

### What the Muse 2 CANNOT reliably detect

- **High-frequency EMG signatures** (30-100 Hz): Dry electrode noise dominates this band. EMG power ratios between gestures are 0.89-1.06x — essentially indistinguishable from noise.
- **Subtle muscle patterns**: The difference between furrow and raise is extremely small. Best t-statistic for this pair is ~3.0, which is marginal.
- **Any neural signal during gestures**: Alpha/beta rhythms are completely masked by the massive artifact from facial movements.

## Spectral Profile (per event window, after 60Hz notch + detrend)

### Rest (baseline)
```
         delta(0.5-4)  theta(4-8)  alpha(8-13)  beta(13-30)  gamma(30-50)  EMG(50-100)
TP9:     1,657K        2,925K      1,940K       6,877K       9,440K        67,651K
AF7:     1,963K        2,804K      2,583K       7,282K       8,604K        55,722K
AF8:     1,789K        3,560K      2,916K       8,019K       11,401K       49,901K
TP10:    2,439K        3,170K      2,887K       9,128K       10,327K       49,470K
```

The EMG band (50-100 Hz) has **5-30x more power** than the low-frequency bands even during rest. This is all electrode noise, not muscle activity.

## Implications for Pipeline Design

1. **Low-frequency features are best** — gesture information lives in 0.5-10 Hz
2. **Absolute amplitudes are noisy** — use ratios and normalized features
3. **Clipping must be handled** — interpolation or clipping-robust features (duration above threshold)
4. **Inter-channel ratios** capture spatial patterns that raw amplitudes miss
5. **Temporal dynamics** (energy distribution across the window) are more discriminative than spectral features
6. **Need many events** — with SNR this low, statistical power requires 25+ events per class minimum, 50+ preferred
