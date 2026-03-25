import asyncio
import time
import threading
import pickle
import numpy as np
import sounddevice as sd
from collections import deque
from scipy.signal import iirnotch, filtfilt, detrend, welch
from bleak import BleakClient

MUSE_ADDRESS = "00:55:DA:B8:35:23"
CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
EEG_UUIDS = {
    "TP9":  "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7":  "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8":  "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FS = 256
WINDOW = 128  # 0.5s
COOLDOWN = 0.6  # seconds between detections

# ─── Sound ───
AUDIO_SR = 44100

def make_tone(freq, duration, volume=0.5):
    t = np.linspace(0, duration, int(AUDIO_SR * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * volume
    fade = int(AUDIO_SR * 0.01)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)

BLINK_SOUND = np.concatenate([make_tone(1047, 0.08, 0.5), make_tone(1319, 0.08, 0.5)])

def play(sound):
    threading.Thread(target=lambda: sd.play(sound, AUDIO_SR), daemon=True).start()

# ─── Preprocessing ───
b60, a60 = iirnotch(60.0, 200.0, FS)

def preprocess_window(window):
    """window: (128, 4) raw µV -> preprocessed z-scored"""
    out = np.zeros_like(window)
    for i in range(4):
        sig = window[:, i].copy()
        sig = filtfilt(b60, a60, sig)
        sig = detrend(sig)
        mu, std = sig.mean(), sig.std()
        if std > 0:
            sig = (sig - mu) / std
        out[:, i] = sig
    return out

# ─── Feature extraction (same as training) ───
def band_power(signal, fs, fmin, fmax):
    nperseg = min(len(signal), 128)
    if nperseg < 4:
        return 0.0
    f, pxx = welch(signal, fs=fs, nperseg=nperseg)
    mask = (f >= fmin) & (f <= fmax)
    return np.trapezoid(pxx[mask], f[mask]) if mask.any() else 0.0

def extract_features(window):
    """window: (128, 4) -> feature vector"""
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
    return np.array(feats).reshape(1, -1)

# ─── µV conversion ───
def raw_to_uv(val):
    if val > 32767:
        return (val - 65536) * 0.48828125
    return val * 0.48828125

def parse_packet(data):
    samples = []
    for i in range(2, len(data), 2):
        val = int.from_bytes(data[i:i+2], byteorder='big')
        samples.append(raw_to_uv(val))
    return samples

# ─── Main ───
async def main():
    # Load RF model
    print("Loading Random Forest model...")
    with open("blink_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    print(f"Loaded: {bundle['best_name']}")

    # Ring buffers — keep 2 windows worth so we can slide
    buffers = {ch: deque(maxlen=WINDOW * 2) for ch in CHANNELS}
    lock = threading.Lock()
    blink_count = [0]
    last_blink_time = [0.0]

    def make_handler(ch):
        def callback(sender, data):
            samples = parse_packet(data)
            with lock:
                buffers[ch].extend(samples)
        return callback

    print(f"\nConnecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        for ch, uuid in EEG_UUIDS.items():
            await client.start_notify(uuid, make_handler(ch))
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

        # Calibration: collect 3 seconds of rest to establish baseline
        print("\nCalibrating... keep still for 3 seconds...")
        await asyncio.sleep(3)

        with lock:
            if all(len(buffers[ch]) >= WINDOW for ch in CHANNELS):
                cal_window = np.column_stack([list(buffers[ch])[-WINDOW:] for ch in CHANNELS])
                cal_window = preprocess_window(cal_window)
                cal_feats = extract_features(cal_window)
                cal_feats_scaled = scaler.transform(cal_feats)
                cal_prob = model.predict_proba(cal_feats_scaled)[0, 1]
                print(f"Baseline blink probability: {cal_prob:.3f} (should be low)")

        print("\n" + "=" * 50)
        print("REAL-TIME BLINK DETECTION (Random Forest)")
        print("Blink and listen for the sound!")
        print("Press Ctrl+C to stop")
        print("=" * 50 + "\n")

        try:
            while True:
                await asyncio.sleep(0.05)

                with lock:
                    if any(len(buffers[ch]) < WINDOW for ch in CHANNELS):
                        continue
                    # Take the latest 128 samples
                    window = np.column_stack([list(buffers[ch])[-WINDOW:] for ch in CHANNELS])

                # Preprocess + extract features
                window = preprocess_window(window)
                feats = extract_features(window)

                # Use scaler if not RF
                if bundle["best_name"] != "Random Forest":
                    feats = scaler.transform(feats)

                prob = model.predict_proba(feats)[0, 1]

                # Live readout
                bar = "█" * int(prob * 30)
                print(f"  blink_prob={prob:.3f} {bar:30s}", end="\r", flush=True)

                now = time.monotonic()
                if prob > 0.6 and (now - last_blink_time[0]) > COOLDOWN:
                    blink_count[0] += 1
                    last_blink_time[0] = now
                    play(BLINK_SOUND)
                    print(f"  BLINK #{blink_count[0]:3d}  (prob={prob:.2f})                    ")

        except KeyboardInterrupt:
            pass
        finally:
            await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
            print(f"\nStopped. Total blinks detected: {blink_count[0]}")

asyncio.run(main())
