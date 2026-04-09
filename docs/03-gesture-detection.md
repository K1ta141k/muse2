# Experiment 3: Multi-Gesture Detection

**Date:** April 6-8, 2026
**Pipeline:** V1 (failed at 43% F1) → V2 (80% F1)
**Gestures:** rest, blink, brow furrow, brow raise, jaw clench

## Goal

Detect 4 distinct facial gestures from Muse 2 EEG in real-time, to use as input controls for external applications via WebSocket/OSC.

## Data Collection

**Protocol** (`collect_gestures.py`):
- 120s sessions, round-robin shuffled gesture order
- 5s rest → 1s countdown → gesture cue (distinct audio tone per gesture) → 0.5s gesture window → 1.5s recovery
- ~7-8 rounds per session = ~7-8 events per gesture per session

**Audio cues:** 880Hz (blink), 600Hz (furrow), 1200Hz (raise), 350Hz (clench)

**Dataset:** 6 sessions total
- 142,560 raw samples across ~739s
- 27 blink, 24 furrow, 26 raise, 25 clench events
- Relabeled with per-gesture peak detection (frontal channels for blink/furrow/raise, temporal for clench)

## The V1 Failure (43% macro F1)

Applied the same pipeline that got 97% on blink detection:
- Preprocessing: 60Hz notch + detrend + z-score
- Features: 43 extended features (basic 20 + EMG power, skewness, kurtosis, etc.)
- Models: RF, SVM, LR, CNN

**What went wrong in real-time testing:**
- "Raise" dominated predictions (44/83 detections)
- "Furrow" barely detected (3/83 detections)
- Model was essentially guessing

## Deep Signal Analysis

Ran comprehensive analysis on the raw data to understand why V1 failed.

### Finding 1: Catastrophic ADC Clipping

96-100% of gesture event windows had >5% of samples clipping at ±16,000 µV (the ADC limit). Average clipping: 13-19% of samples per window.

This means peak detection during relabeling was finding the ADC rail, not the actual signal peak.

### Finding 2: Tiny Gesture-vs-Rest Differences

| Band | Best gesture-vs-rest ratio |
|---|---|
| EOG (0.5-10 Hz) | 1.11-1.28x |
| Mu/Beta (8-30 Hz) | 0.91-1.12x |
| EMG (30-100 Hz) | 0.89-1.06x |

Gestures barely move the needle above the noise floor.

### Finding 3: Z-Score Was Destroying Signal

The V1 z-score normalization was removing the small absolute amplitude differences between gestures and rest. After z-scoring, a window with a blink looks the same as a window with noise — both normalized to unit variance.

### Finding 4: EMG Band Is NOT Discriminative

Counter to expectations, the 30-100 Hz EMG band showed almost identical power across all gesture types on the Muse 2. The dry electrodes and forehead placement produce so much broadband noise that any muscle-specific EMG is buried.

**The discriminative information lives in the low-frequency band (0.5-10 Hz)** — slow potential shifts from facial movements, not high-frequency EMG.

### Finding 5: Best Separability Features (t-test, gesture vs rest)

| Gesture | Best feature | t-stat | Direction |
|---|---|---|---|
| Blink | TP10 low-band RMS | 3.90 | higher |
| Furrow | TP10 low-band RMS | 2.86 | higher |
| Raise | AF7 EMG envelope std | 3.35 | higher |
| Clench | TP10 mid-band RMS | 2.14 | higher |

Hardest pair: furrow vs raise (best t=2.97 on AF7_emg_env_std)

## V2 Pipeline Design

### Preprocessing (`preprocess_v2`)

```
raw µV → interpolate clipped samples → 60Hz notch → bandpass 0.5-100Hz → detrend
```

Key change: **NO z-score normalization**. Amplitudes are preserved for band decomposition.

**Clipping interpolation:** Samples hitting ±15,500 µV are replaced with linear interpolation from neighboring non-clipped samples. This recovers some peak shape information.

### Multi-Band Decomposition (`decompose_bands`)

Each channel is decomposed into 3 frequency bands using Butterworth bandpass filters (4th order):

| Band | Range | What it captures |
|---|---|---|
| EOG | 0.5-10 Hz | Blink artifacts, slow facial movement potentials |
| Mu/Beta | 8-30 Hz | Neural oscillations, some movement-related activity |
| EMG | 30-100 Hz | Muscle contraction (noisy on Muse 2 but still informative as ratios) |

### Feature Extraction (`extract_features_v2` — 48 features)

**Design principle:** All features are **ratios or normalized values** — no raw amplitudes. This makes the model robust to inter-session amplitude variance while still capturing gesture-specific patterns.

**Per channel (4 channels x 10 features = 40):**
- 3x **temporal centroids** (one per band) — where energy concentrates temporally
- 4x **quarter energy ratios** — temporal dynamics (impulse vs sustained)
- 1x **transient sharpness** — max|derivative|/std (scale-free)
- 2x **band fractions** — EOG fraction and EMG fraction of total band energy

**Cross-channel (8 features):**
- 3x **frontal/temporal ratio** (one per band) — spatial signature
- 3x **L/R asymmetry** (one per band) — lateralization
- 2x **inter-channel correlation** (AF7-AF8, TP9-TP10) — spatial coherence

### Data Augmentation

Each gesture event generates 5 training windows by shifting the center by [-20, -10, 0, +10, +20] samples. This increases effective dataset from ~200 to ~1,000 windows.

**Why this works:** Gesture events have variable timing relative to peaks. Time-shifted copies teach the model to recognize gestures at slightly different alignments, which is exactly what happens in real-time detection.

## V2 Results (5-fold stratified CV)

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| Random Forest | 0.678 | 0.854 | 0.494 | 0.555 |
| XGBoost | 0.774 | 0.821 | 0.672 | 0.722 |
| SVM | 0.759 | 0.737 | 0.786 | 0.752 |
| **CNN** | **0.802** | **0.806** | **0.813** | **0.797** |

**Top 15 features by importance (RF fallback model):**

1. `AF7_q3_ratio` (0.045) — energy in last quarter of window on AF7
2. `AF8_q3_ratio` (0.041) — energy in last quarter on AF8
3. `corr_af7_af8` (0.032) — frontal channel correlation
4. `emg_frontal_ratio` (0.031) — frontal vs temporal EMG balance
5. `AF7_emg_centroid` (0.030) — where EMG energy peaks on AF7
6. `AF8_emg_centroid` (0.030) — where EMG energy peaks on AF8
7. `emg_lr_asymmetry` (0.026) — left vs right EMG balance
8. `AF8_eog_frac` (0.025) — fraction of AF8 energy in EOG band
9. `corr_tp9_tp10` (0.024) — temporal channel correlation
10. `AF7_q1_ratio` (0.024) — early energy on AF7

## What Makes Each Gesture Detectable

| Gesture | Primary signal | Key discriminating features |
|---|---|---|
| **Blink** | Large, fast EOG deflection on frontal channels | High q0 ratio (energy at start), low q3 ratio, high sharpness |
| **Furrow** | Sustained frontal activation | Different EMG centroid than raise, asymmetry pattern |
| **Raise** | Frontal activation, different spatial pattern | EMG envelope variability, different LR asymmetry |
| **Clench** | Temporal channel activation (masseter muscle) | High EMG frontal ratio shifts toward temporal, different correlation pattern |

## Lessons Learned

1. **Don't z-score for multi-class** — it destroys the amplitude differences that separate similar gestures
2. **Multi-band decomposition is essential** — different gestures have energy in different frequency ranges
3. **Ratio features beat raw features** for small, noisy datasets — they're naturally normalized
4. **Data augmentation via time-shifting** gives a massive boost (43% → 80% F1)
5. **The Muse 2's best signal for gestures is low-frequency (0.5-10 Hz)**, not EMG — dry electrode noise dominates the high frequencies
6. **With ~25 events per class**, keep feature count low (~50 max) to avoid overfitting
