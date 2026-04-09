# muse2

EEG experiments with the Muse 2 headband via BLE (bleak).

## Setup

```bash
conda create -n muse python=3.11
conda activate muse
pip install bleak numpy scipy matplotlib pandas scikit-learn seaborn sounddevice torch xgboost websockets
```

## Repo Structure

```
muse2/
├── core/                   # Shared Muse 2 BLE utilities
│   ├── __init__.py         # Constants: address, UUIDs, channels, sample rate, gesture labels
│   ├── audio.py            # Audio cue/feedback utilities (lazy sounddevice to avoid COM conflict)
│   ├── features.py         # V1 + V2 signal processing & feature extraction pipelines
│   ├── stream.py           # Basic BLE EEG streaming
│   ├── viz.py              # Real-time EEG visualization
│   └── scan.py             # BLE device scanner
│
├── experiments/
│   ├── blink_detection/    # Blink classification pipeline (single gesture, V1 pipeline)
│   │   ├── collect_blinks.py       # Cue-based blink data collection
│   │   ├── relabel_blinks.py       # Re-label using peak detection (reaction time fix)
│   │   ├── train_blink.py          # Train RF/SVM/LR/CNN classifiers
│   │   ├── realtime_blink.py       # Real-time blink detection with audio feedback
│   │   └── ...
│   │
│   ├── gesture_detection/  # Multi-gesture classification (4 gestures + rest, V2 pipeline)
│   │   ├── collect_gestures.py     # Cue-based multi-gesture data collection
│   │   ├── relabel_gestures.py     # Peak detection relabeling per gesture type
│   │   ├── train_gesture.py        # Train RF/XGBoost/SVM/CNN with V2 features
│   │   └── realtime_gesture.py     # Real-time multi-gesture detection
│   │
│   └── signal_quality/     # Electrode contact solution comparison
│       └── signal_quality.py
│
├── server/                 # WebSocket/OSC event server for external app integration
│   ├── config.py           # Thresholds, cooldowns, ports
│   ├── event_server.py     # EventBus + GestureDetector
│   └── run.py              # Main entry: BLE + detection + WebSocket + OSC
│
├── data/
│   ├── blink_data/         # Blink experiment recordings
│   ├── gesture_data/       # Multi-gesture recordings + trained models
│   └── signal_quality/     # Dry/water/saline comparison data
│
└── notebooks/
    └── blink_detector.ipynb
```

## Hardware

- **Headband**: Muse 2 (2016+ model)
- **Channels**: TP9, AF7, AF8, TP10 (4 dry EEG electrodes)
- **Sample rate**: 256 Hz
- **Connection**: Bluetooth Low Energy (BLE) via bleak
- **Address**: `00:55:DA:B8:35:23`

## Experiments

### 1. Blink Detection (V1 pipeline)

Binary blink vs rest classification using cue-based data collection.

**Key findings:**
- Cue-based labeling needs reaction time correction (~660ms median delay)
- Peak-detected re-labeling + event-aligned windows → 97% F1
- Preprocessing: 60Hz notch + detrend + z-score (V1)
- 17 blink events is enough for a proof of concept

### 2. Multi-Gesture Detection (V2 pipeline)

5-class classification: rest, blink, brow furrow, brow raise, jaw clench.

**Data:** 6 sessions, ~25 events per gesture, 142K total samples.

**V1 → V2 pipeline evolution:**

The V1 pipeline (notch + detrend + z-score + broadband features) only achieved 43% macro F1 on multi-gesture. Analysis revealed:
- Z-score normalization destroyed amplitude differences between gestures
- 96-100% of event windows had ADC clipping (±16,000 µV)
- Gesture-vs-rest signal ratios were only 1.1-1.3x (buried in noise)
- EMG band (50-100 Hz) was NOT discriminative — dominated by electrode noise

**V2 pipeline design (`core/features.py`):**

| Stage | V1 | V2 |
|---|---|---|
| Clipping | Ignored | Interpolated before filtering |
| Filtering | 60Hz notch only | Bandpass 0.5-100Hz + 60Hz notch |
| Normalization | Global z-score (destroys amplitude) | None — amplitudes preserved for band decomposition |
| Band decomposition | None | EOG (0.5-10Hz), mu/beta (8-30Hz), EMG (30-100Hz) |
| Features | 43 raw-amplitude features | 48 normalized/ratio features (no raw amplitudes) |
| Augmentation | None | 5x time-shifted windows (±20, ±10, 0 samples) |

**Key V2 features (what actually discriminates gestures on Muse 2):**
- **Temporal energy distribution** (quarter-based RMS ratios) — blinks are impulsive, clench is sustained
- **Band energy fractions** (EOG/EMG fraction per channel) — blinks are low-frequency, clench is EMG-heavy
- **Temporal centroids** (where energy concentrates per band) — captures onset dynamics
- **Cross-channel ratios** (frontal/temporal, L/R asymmetry) — clench lights up temporal channels
- **Inter-channel correlations** — gestures change spatial coherence

**Results (5-fold stratified CV):**

| Model | V1 F1 | V2 F1 |
|---|---|---|
| Random Forest | 0.20 | 0.56 |
| XGBoost | 0.36 | 0.72 |
| SVM | 0.30 | 0.75 |
| **CNN** | 0.21 | **0.80** |

### 3. Signal Quality (Dry vs Water vs Saline)

Compared electrode contact solutions. Saline wins on SNR (3.50 vs 3.08 dry).

## Event Server

WebSocket/OSC server for external app integration:

```bash
python -m server.run [--model data/gesture_data/gesture_model.pkl] [--osc]
```

- WebSocket on `ws://localhost:8765` — JSON gesture events
- Optional OSC on `udp://localhost:9000` — `/muse/gesture` messages
- Events: `{"type": "gesture", "gesture": "blink", "confidence": 0.87, ...}`

## Usage

```bash
# Collect gesture data (120s session, shuffled gesture cues)
python -m experiments.gesture_detection.collect_gestures --duration 120

# Relabel with peak detection
python -m experiments.gesture_detection.relabel_gestures

# Train classifiers
python -m experiments.gesture_detection.train_gesture

# Real-time detection
python -m experiments.gesture_detection.realtime_gesture

# Event server (for external apps)
python -m server.run
```

## Next Steps

- [ ] Collect more gesture data (target 50+ events per class)
- [ ] Test realtime V2 model with Muse
- [ ] Test WebSocket event server end-to-end
- [ ] Explore saline electrodes for better SNR
