"""
Real-time multi-gesture detection using trained model.

Supports both single-stage (5-class) and two-stage (gate + classify) architectures.
Streams EEG from Muse 2, extracts features every 50ms, and classifies
gestures with per-gesture cooldowns and audio feedback.

Usage:
    python -m experiments.gesture_detection.realtime_gesture [--model ...]
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import ctypes
    ctypes.windll.ole32.CoInitializeEx(0, 0x0)  # COINIT_MULTITHREADED

import argparse
import time
import threading
import pickle
import numpy as np
from collections import deque
from bleak import BleakClient

from core import MUSE_ADDRESS, CONTROL_UUID, EEG_UUIDS, CHANNELS, FS, GESTURE_LABELS
from core.audio import play, GESTURE_DETECT_SOUNDS
from core.features import preprocess_v2, extract_features_v2, raw_to_uv, parse_packet

# ─── Config ───
WINDOW = 128  # 0.5s
GESTURE_THRESHOLDS = {1: 0.45, 2: 0.35, 3: 0.45, 4: 0.35}
GESTURE_COOLDOWNS = {1: 0.6, 2: 0.8, 3: 0.8, 4: 0.8}


class ModelInference:
    """Unified inference for single-stage and two-stage models."""

    def __init__(self, bundle):
        self.arch = bundle.get("architecture", "single_stage")
        self.name = bundle["best_name"]

        if self.arch == "two_stage":
            self.gate = bundle["gate_model"]
            self.classify = bundle["classify_model"]
            self.scaler1 = bundle["scaler1"]
            self.scaler2 = bundle["scaler2"]
            self.gate_name = bundle["gate_name"]
            self.classify_name = bundle["classify_name"]
            print(f"Loaded two-stage: gate={self.gate_name}, classify={self.classify_name}")
        else:
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            print(f"Loaded single-stage: {self.name}")

    def predict(self, feats):
        """Return (probabilities_array_5, predicted_class, confidence)."""
        if self.arch == "two_stage":
            return self._predict_two_stage(feats)
        else:
            return self._predict_single(feats)

    def _predict_single(self, feats):
        if self.name not in ("1S-RF",) and "RF" not in self.name:
            feats_scaled = self.scaler.transform(feats)
        else:
            feats_scaled = feats

        probs = self.model.predict_proba(feats_scaled)[0]
        pred = int(np.argmax(probs))
        return probs, pred, float(probs[pred])

    def _predict_two_stage(self, feats):
        # Stage 1: gate (rest vs gesture)
        if self.gate_name == "RF":
            feats_g = feats
        else:
            feats_g = self.scaler1.transform(feats)

        gate_prob = self.gate.predict_proba(feats_g)[0]
        p_rest = gate_prob[0]
        p_gesture = gate_prob[1]

        # If gate says rest with high confidence, skip stage 2
        probs = np.zeros(5)
        probs[0] = p_rest

        if p_gesture > 0.4:  # only run stage 2 if gate thinks it might be a gesture
            # Stage 2: which gesture?
            if self.classify_name == "RF":
                feats_c = feats
            else:
                feats_c = self.scaler2.transform(feats)

            stage2_prob = self.classify.predict_proba(feats_c)[0]
            classes = self.classify.classes_

            if self.classify_name == "XGB":
                # XGB classes are 0-3, map to 1-4
                for k in range(len(stage2_prob)):
                    probs[k + 1] = p_gesture * stage2_prob[k]
            else:
                for k_idx, k in enumerate(classes):
                    probs[int(k)] = p_gesture * stage2_prob[k_idx]

        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        return probs, pred, conf


async def main():
    parser = argparse.ArgumentParser(description="Real-time multi-gesture detection")
    parser.add_argument("--model", type=str, default="data/gesture_data/gesture_model.pkl")
    args = parser.parse_args()

    # Load model
    print("Loading gesture model...")
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)

    inference = ModelInference(bundle)

    # Ring buffers
    buffers = {ch: deque(maxlen=WINDOW * 2) for ch in CHANNELS}
    lock = threading.Lock()
    gesture_counts = {gid: 0 for gid in range(1, 5)}
    last_gesture_time = {gid: 0.0 for gid in range(1, 5)}
    packet_count = [0]

    def make_handler(ch):
        def callback(sender, data):
            packet_count[0] += 1
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

        # Check if packets are arriving
        print("Waiting for BLE packets...")
        for i in range(10):
            await asyncio.sleep(0.5)
            if packet_count[0] > 0:
                break
        print(f"Packets received: {packet_count[0]}")
        if packet_count[0] == 0:
            print("ERROR: No BLE packets! Try power-cycling the Muse.")
            return

        print("Calibrating... keep still for 3 seconds...")
        await asyncio.sleep(3)

        print("\n" + "=" * 55)
        print("REAL-TIME GESTURE DETECTION")
        arch_label = "two-stage" if inference.arch == "two_stage" else "single-stage"
        print(f"Model: {inference.name} ({arch_label})")
        print("Gestures: blink, furrow, raise, clench")
        print("Press Ctrl+C to stop")
        print("=" * 55 + "\n")

        try:
            while True:
                await asyncio.sleep(0.05)

                with lock:
                    sizes = {ch: len(buffers[ch]) for ch in CHANNELS}
                    if any(s < WINDOW for s in sizes.values()):
                        min_ch = min(sizes, key=sizes.get)
                        print(f"  Buffering... {sizes[min_ch]}/{WINDOW} ({min_ch})    ", end="\r", flush=True)
                        continue
                    window = np.column_stack([list(buffers[ch])[-WINDOW:] for ch in CHANNELS])

                # V2 preprocessing + feature extraction
                window = preprocess_v2(window)
                feats = np.array(extract_features_v2(window)).reshape(1, -1)
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

                # Inference
                probs, pred, conf = inference.predict(feats)

                # Live readout
                parts = []
                for gid in range(1, 5):
                    gname = GESTURE_LABELS[gid][:5]
                    p = probs[gid]
                    bar = "#" * int(p * 10)
                    parts.append(f"{gname}:{p:.2f}|{bar:<10s}|")
                line = " ".join(parts)
                print(f"\r  {line}", end="", flush=True)

                # Detect gesture (skip rest class 0)
                if pred > 0:
                    threshold = GESTURE_THRESHOLDS.get(pred, 0.5)
                    cooldown = GESTURE_COOLDOWNS.get(pred, 0.8)
                    now = time.monotonic()

                    if conf > threshold and (now - last_gesture_time[pred]) > cooldown:
                        gesture_counts[pred] += 1
                        last_gesture_time[pred] = now
                        gesture_name = GESTURE_LABELS[pred].upper()
                        total = sum(gesture_counts.values())

                        if pred in GESTURE_DETECT_SOUNDS:
                            play(GESTURE_DETECT_SOUNDS[pred])

                        print(f"\n  >>> {gesture_name} #{gesture_counts[pred]} "
                              f"(conf={conf:.2f}) [total={total}]")

        except KeyboardInterrupt:
            pass
        finally:
            try:
                await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
            except Exception:
                pass
            print(f"\n\nStopped. Gesture counts:")
            for gid in range(1, 5):
                print(f"  {GESTURE_LABELS[gid]:8s}: {gesture_counts[gid]}")
            print(f"  {'TOTAL':8s}: {sum(gesture_counts.values())}")


asyncio.run(main())
