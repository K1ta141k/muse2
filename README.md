# muse2

EEG experiments with the Muse 2 headband via BLE (bleak).

## Setup

```bash
python -m venv brainvenv
source brainvenv/bin/activate
pip install bleak numpy scipy matplotlib pandas scikit-learn seaborn sounddevice torch
```

## Repo Structure

```
muse2/
├── core/                   # Shared Muse 2 BLE utilities
│   ├── __init__.py         # Constants: address, UUIDs, channels, sample rate
│   ├── stream.py           # Basic BLE EEG streaming
│   ├── viz.py              # Real-time EEG visualization
│   └── scan.py             # BLE device scanner
│
├── experiments/
│   ├── blink_detection/    # Blink classification pipeline
│   │   ├── collect_blinks.py       # Cue-based blink data collection (audio cues)
│   │   ├── blink_test.py           # Quick signal validation (blink artifact check)
│   │   ├── compare.py              # On-head vs off-head comparison
│   │   ├── viz_blinks.py           # Visualize collected blink data
│   │   ├── relabel_blinks.py       # Re-label blinks using peak detection (reaction time fix)
│   │   ├── train_blink.py          # Train RF/SVM/LR/CNN classifiers
│   │   ├── realtime_blink.py       # Real-time blink detection with audio feedback
│   │   └── water_blink_detect.py   # Anomaly-based blink detection (no cues)
│   │
│   └── signal_quality/     # Electrode contact solution comparison
│       └── signal_quality.py       # Dry vs water vs saline benchmark
│
├── notebooks/
│   └── blink_detector.ipynb        # Full blink detection analysis notebook
│
├── models/                 # Trained model artifacts
│   ├── blink_model.pkl     # Random Forest blink classifier
│   └── blink_cnn.pt        # CNN blink classifier (PyTorch)
│
└── data/
    ├── blink_data/         # Blink experiment recordings
    └── signal_quality/     # Dry/water/saline comparison data
```

## Hardware

- **Headband**: Muse 2 (2016+ model)
- **Channels**: TP9, AF7, AF8, TP10 (4 dry EEG electrodes)
- **Sample rate**: 256 Hz
- **Connection**: Bluetooth Low Energy (BLE) via bleak
- **Address**: `00:55:DA:B8:35:23`

## Experiments

### 1. Blink Detection

Classified blink vs rest from EEG using cue-based data collection.

**Key findings:**
- Cue-based labeling needs reaction time correction (~660ms median delay)
- Peak-detected re-labeling + event-aligned windows → 97% accuracy
- Random Forest, SVM, Logistic Regression, and CNN all performed comparably
- Preprocessing (60Hz notch + detrend + z-score) is essential
- 17 blink events is enough for a proof of concept but not production

### 2. Signal Quality (Dry vs Water vs Saline)

Compared electrode contact solutions using SNR, rest noise, blink amplitude, and alpha power.

**Results:**
| Metric | Dry | Water | Saline | Best |
|---|---|---|---|---|
| Rest Noise (std) | 10,399 | 10,039 | **8,834** | Saline |
| SNR | 3.08 | 3.14 | **3.50** | Saline |

Saline wins on SNR — lower noise floor, not bigger blinks.

## Next Steps

- [ ] Brow movement detection (furrowing, raising) for app interaction
- [ ] Collect more blink data (100+ events) for robust models
- [ ] Real-time brow movement classifier
- [ ] App integration (brow gestures as input)
