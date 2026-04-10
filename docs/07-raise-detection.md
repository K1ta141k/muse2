# Brow Raise Detection

Binary raise detector built on the V2 signal pipeline. First gesture targeted
for reliable single-gesture detection as a usable feature.

## Journey & key learnings

### Starting point

The multi-gesture classifier (V2) achieved 80% macro F1 across 5 classes
(rest, blink, furrow, raise, clench). Rather than push all gestures
simultaneously, we focused on making one gesture production-ready: brow raise.

### Problem 1: Binary classifier had high CV, terrible real-time performance

First attempt: train rest-vs-raise on 26 raise events from gesture data.
Cross-validation showed 0.90 F1 (SVM). Simulated real-time: **241 false
positives in 9 minutes** (26 FP/min) — completely unusable.

**Root cause: distribution mismatch.** Training used carefully peak-centered
windows. Real-time inference slides a window every 50ms across raw signal,
seeing noise, drift, electrode artifacts, and other gestures the model had
never encountered as "not-raise".

### Problem 2: Other gestures triggered false positives

The binary model had never seen blinks, furrows, or clenches — they all
share frontal channel activity with raises. Adding other gesture events as
hard negatives (label=0) helped in CV but didn't fix real-time because the
core distribution mismatch remained.

### Solution: sliding-window training

**The fix that worked:** sample negative training windows the same way
inference sees them — random positions across entire recordings, not just
curated rest zones. This includes windows near noise, other gestures, signal
transitions, and electrode artifacts.

Results after this change:

| Metric | Before (curated negatives) | After (random negatives) |
|--------|---------------------------|--------------------------|
| Sim FP at thresh=0.5 | 241 (26/min) | 58 (6.2/min) |
| Sim FP at thresh=0.9 | 91 (9.8/min) | 6 (0.6/min) |
| Recall | 1.000 | 1.000 |

### Problem 3: Single-window noise spikes

Even with better training, individual windows occasionally spike to high
confidence during rest (transient noise artifacts). Single-threshold
detection triggers on these.

**Solution: moving average.** Average the last 5 predictions (~250ms). Real
raises sustain across multiple windows; noise spikes are transient.

| Threshold | Without avg (FP) | With avg N=5 (FP) |
|-----------|-----------------|-------------------|
| 0.5 | 58 | 12 |
| 0.7 | 29 | **0** |
| 0.9 | 6 | 0 |

At threshold=0.7 with N=5 averaging: **26/26 detected, 0 false positives,
90ms median latency.**

### Problem 4: Simulation vs real-life gap

Simulation at threshold=0.7 was perfect. Real-life needed threshold=0.3 to
catch most raises, and still missed some. Sources of the gap:

- **BLE streaming instability on macOS** — packets arrive in bursts, channels
  stall independently (100-500ms), full stream stalls (>2s) happen. When the
  model runs on stale/partial data, predictions are garbage.
- **Electrode contact drift** — impedance changes as the headband shifts
- **Session-to-session amplitude variance** — different headband placement
- **Signal non-stationarity** — baseline EEG drifts over minutes

### Problem 5: BLE stalls cause missed raises and phantom predictions

When a channel stalls, the buffer contains old samples mixed with the
assumption of fresh data. The model sees a frozen signal and predicts
nonsense (often 0.0 confidence, sometimes high confidence).

**Solution: per-channel freshness tracking.** Each BLE callback records its
timestamp. Before running the model, check all 4 channels are <500ms fresh.
If any channel is stale: skip prediction and clear the moving average. When
the stream resumes, the model restarts clean.

Trade-off: if a raise happens during a stall, it's missed. No way around
this — if the data never arrives, the model can't see it.

### Problem 6: Moving average adds latency, can swallow brief raises

The average needs several consecutive high-confidence windows to cross
threshold. A brief or weak raise might produce one or two high windows
surrounded by low ones — the average never crosses.

**Solution: peak detection mode.** Instead of triggering when average > threshold,
trigger on the **rising edge** — when the average crosses threshold from below.
This catches the spike even if it immediately drops back. Also prevents
re-triggering on sustained high noise (no new rising edge = no new trigger).

### Problem 7: Cue-based data collection has reaction-time offset

Audio cue at t=0, human reacts at t+300-800ms. Requires post-hoc relabeling
with peak detection to re-center windows. Inaccurate, loses some events.

**Solution: model-assisted collection.** Use the live detector as the labeler.
The model fires at the actual signal event, not after a cue + human delay.
After recording, review each detection (keep/delete), add missed timestamps.
Labels are placed at the right time automatically.

## Data

| Source | Raises | Rest | Notes |
|--------|--------|------|-------|
| Gesture sessions (6x 120s) | 26 (label=3) | ~133k samples | Cue-based, shared with multi-gesture |
| Raise-only sessions (3x 120s) | 42 (label=1) | ~70k samples | Cue-based, randomized rest 4-8s |
| Blink sessions | 0 | ~10k samples | Rest-only source |

Total: ~68 raise events before augmentation, 340 after 5x time-shift.

## Training pipeline

### Positive windows
- Raise events from all sources (raise-only CSVs + gesture data label=3)
- 5x time-shift augmentation (±10, ±20 samples)

### Negative windows
- **Random windows** from all recordings — sampled from any position except
  ±128 samples around raise events. Matches real-time inference distribution.
- **Hard negatives**: blink, furrow, clench events from gesture data (augmented)
- 4:1 negative-to-positive ratio

### Models evaluated (5-fold stratified CV)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Random Forest | 0.908 | 0.918 | 0.521 | 0.664 |
| XGBoost | 0.930 | 0.899 | 0.676 | 0.772 |
| **SVM** | **0.953** | **0.851** | **0.885** | **0.868** |
| CNN | 0.924 | 0.803 | 0.876 | 0.818 |

SVM (RBF kernel, balanced class weights) is the production model.

### Top discriminating features

1. `AF8_q3_ratio` (0.161) — late-window energy on right frontal
2. `emg_lr_asymmetry` (0.043) — left/right muscle activity difference
3. `TP10_emg_centroid` (0.038) — temporal EMG timing
4. `AF8_emg_centroid` (0.037) — frontal EMG timing

## Real-time detection

### Detection modes

**Peak mode** (default, recommended):
Triggers on the rising edge of the moving average crossing the threshold.
Catches brief spikes, prevents re-triggering on sustained noise.

**Average mode**:
Triggers whenever the moving average is above threshold (with cooldown).
Simpler but can miss brief raises and re-trigger on sustained noise.

### Simulated results (peak mode, avg N=5, gesture data 556.9s)

| Threshold | TP | FP | Precision | Recall | F1 | FP/min |
|-----------|----|----|-----------|--------|-----|--------|
| 0.3 | 26/26 | 32 | 0.448 | 1.000 | 0.619 | 3.4 |
| 0.5 | 26/26 | 11 | 0.703 | 1.000 | 0.825 | 1.2 |
| 0.6 | 26/26 | 3 | 0.897 | 1.000 | 0.945 | 0.3 |
| **0.7** | **26/26** | **0** | **1.000** | **1.000** | **1.000** | **0.0** |
| 0.8 | 25/26 | 0 | 1.000 | 0.962 | 0.980 | 0.0 |

### Simulation vs real-life thresholds

| Setting | Simulation | Real-life (best so far) |
|---------|-----------|------------------------|
| Threshold | 0.7 | **0.2** |
| Avg window | 5 | **3** |
| Cooldown | 0.8s | 0.8s |
| Mode | peak | peak |

The lower avg window (3 vs 5) reduces latency (~150ms vs ~250ms) and prevents
a single low-confidence window from dragging down the average too much.
Combined with the lower threshold, this catches more real raises while peak
mode prevents re-triggering on sustained noise.

### BLE stream resilience

| Issue | Detection | Recovery |
|-------|-----------|----------|
| Channel stall (>500ms) | Per-channel timestamp check | Skip prediction, clear avg |
| Full stall (>2s) | Last-packet timestamp | Warning message, wait for resume |
| Post-stall garbage | Cleared moving average | Fresh predictions from clean buffer |
| Raise during stall | Not possible to detect | Documented limitation |

## Model-assisted data collection

Instead of cue-based collection, `collect_with_model.py` uses the live
detector as the labeler:

1. Record raw EEG while running the model in real-time
2. Model auto-labels detections (with audio feedback)
3. After recording, review each detection: keep or delete (false positives)
4. Add timestamps for any missed raises
5. Save corrected CSV

Advantages:
- **No reaction-time correction** — labels at the signal, not the cue
- **Natural timing** — raises happen when you want
- **Real conditions** — data matches inference environment
- **Iterative** — each round corrects what the model gets wrong

## Files

```
experiments/raise_detection/
├── collect_raises.py        # Cue-based collection (original)
├── collect_with_model.py    # Model-assisted collection (recommended)
├── train_raise.py           # Training pipeline
├── simulate_realtime.py     # Offline simulation on recorded data
└── realtime_raise.py        # Live real-time detection

data/raise_data/
├── raise_*.csv              # Raw session data
├── raise_model.pkl          # Trained model (SVM) + scaler
├── raise_cnn.pt             # CNN weights (if CNN wins)
├── confusion_matrices.png
├── training_results.png
└── simulated_realtime.png
```

## Usage

```bash
# Collect with live model (recommended)
python -m experiments.raise_detection.collect_with_model --threshold 0.3

# Collect with cues (original method)
python -m experiments.raise_detection.collect_raises

# Train (auto-discovers all raise_*.csv in data/raise_data/)
python -m experiments.raise_detection.train_raise

# Simulate on recorded data
python -m experiments.raise_detection.simulate_realtime --threshold 0.7 --avg-window 5 --mode peak

# Live detection
python -m experiments.raise_detection.realtime_raise --threshold 0.3 --avg-window 5 --mode peak
```

## Next steps

- Collect more data with model-assisted labeling to close the sim/real gap
- Investigate BLE stability improvements (connection parameters, retry logic)
- Once raise is solid, add second gesture (blink — easiest, 97% F1 in V1)
- Build composable gesture vocabulary (raise=action1, blink=action2, etc.)
