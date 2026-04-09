# Experiment 2: Signal Quality — Dry vs Water vs Saline

**Date:** Early April 2026
**Result:** Saline electrodes give best SNR (3.50 vs 3.08 dry)

## Goal

Determine if wetting the Muse 2's dry electrodes improves signal quality enough to matter for gesture detection.

## Method

Recorded 30-60s segments under three conditions, same session:
1. **Dry** — electrodes as-is
2. **Water** — electrodes dampened with water
3. **Saline** — electrodes dampened with saline solution

Metrics computed per condition:
- Rest noise (std of signal during eyes-open rest)
- Blink amplitude (peak `|AF7| + |AF8|` during blinks)
- SNR (blink amplitude / rest noise)
- Alpha power (8-12 Hz, eyes-closed vs eyes-open)

## Results

| Metric | Dry | Water | Saline |
|---|---|---|---|
| Rest noise (std, µV) | 10,399 | 10,039 | **8,834** |
| SNR | 3.08 | 3.14 | **3.50** |

## Key Finding

Saline wins — but the improvement is modest. The SNR gain comes from **lower noise floor**, not bigger signals. The Muse 2's fundamental limitation is the dry electrode design and forehead-only placement, not easily fixed by wetting.

## Implications for Gesture Detection

- With dry electrodes, expect ~12-22% ADC clipping at ±16,000 µV
- Gesture signals are only 1.1-1.3x above the noise floor
- Saline helps but doesn't fundamentally change the difficulty of multi-gesture classification
- Any processing pipeline must be robust to high noise and frequent clipping
