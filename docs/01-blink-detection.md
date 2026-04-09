# Experiment 1: Blink Detection

**Date:** Early April 2026
**Pipeline:** V1 (notch + detrend + z-score)
**Result:** 97% F1 (binary: blink vs rest)

## Goal

Prove that the Muse 2 can reliably detect blinks from EEG, as a stepping stone to more complex gesture detection.

## Data Collection

- Cue-based protocol: audio tone signals "blink now", rest periods between
- 17 blink events collected across sessions
- Output: `data/blink_data/` CSVs with `[timestamp, TP9, AF7, AF8, TP10, label]`

## Key Discovery: Reaction Time

Raw cue-based labels are ~660ms late — the label says "blink here" but the actual blink artifact in the EEG is delayed by human reaction time.

**Fix:** Peak detection relabeling (`relabel_blinks.py`):
- Search for the maximum `|AF7| + |AF8|` amplitude in a window after each cue
- Re-center a 0.5s (128-sample) window around the detected peak
- This aligns the label with the actual physiological event

## Preprocessing (V1)

```
raw signal → 60Hz notch filter → linear detrend → z-score normalization
```

- Notch: IIR notch at 60Hz (Q=200) removes power line interference
- Detrend: removes slow DC drift
- Z-score: normalizes to zero mean, unit variance per channel

## Features (V1 — 20 per window)

Per channel (4 channels x 5 features):
- `mean_abs` — mean absolute amplitude
- `std` — standard deviation
- `ptp` — peak-to-peak range
- `power_0.5_5` — delta band power (Welch)
- `power_8_12` — alpha band power (Welch)

## Models

All trained with 5-fold stratified cross-validation:
- Random Forest (200 trees)
- SVM (RBF kernel)
- Logistic Regression
- 1D CNN (Conv1d → BatchNorm → ReLU → MaxPool, 2 blocks)

All achieved ~97% F1 on this binary task.

## Lessons Learned

1. **Reaction time correction is essential** — without it, you're training on noise
2. **Event-aligned windowing** (center on peak, not on cue) is the single biggest accuracy driver
3. **Even 17 events** can give high accuracy for a binary task with a strong signal (blink artifacts are large)
4. **V1 z-score works fine for binary** — but will cause problems for multi-class (see Experiment 3)
