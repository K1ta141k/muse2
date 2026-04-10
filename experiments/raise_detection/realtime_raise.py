"""
Real-time brow-raise detection using trained model.

Streams EEG from Muse 2, extracts features every 50ms, and detects
raise gestures with threshold + cooldown + audio feedback.

Usage:
    python -m experiments.raise_detection.realtime_raise [--model ...] [--threshold 0.9]
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import ctypes
    ctypes.windll.ole32.CoInitializeEx(0, 0x0)

import argparse
import time
import threading
import pickle
import warnings
import numpy as np
from collections import deque
from bleak import BleakClient

warnings.filterwarnings("ignore", category=RuntimeWarning)

from core import MUSE_ADDRESS, CONTROL_UUID, EEG_UUIDS, CHANNELS, FS
from core.audio import play, GESTURE_DETECT_SOUNDS, COUNTDOWN_BEEP, DONE_SOUND
from core.features import preprocess_v2, extract_features_v2, raw_to_uv, parse_packet

WINDOW = 128  # 0.5s


async def main():
    parser = argparse.ArgumentParser(description="Real-time brow-raise detection")
    parser.add_argument("--model", type=str, default="data/raise_data/raise_model.pkl")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--cooldown", type=float, default=0.8)
    parser.add_argument("--avg-window", type=int, default=3,
                        help="Number of consecutive predictions to average (~150ms at 50ms step)")
    parser.add_argument("--mode", choices=["avg", "peak"], default="peak",
                        help="Detection mode: 'avg' = moving average, "
                             "'peak' = trigger on rising edge crossing threshold")
    args = parser.parse_args()

    # Load model
    print("Loading raise model...")
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    model_name = bundle["best_name"]
    needs_scaling = model_name != "Random Forest"
    print(f"Model: {model_name}")
    print(f"Threshold: {args.threshold}, Cooldown: {args.cooldown}s, "
          f"Avg window: {args.avg_window} (~{args.avg_window * 50}ms), "
          f"Mode: {args.mode}")

    # Ring buffers — large enough to never drop data
    buffers = {ch: deque(maxlen=WINDOW * 4) for ch in CHANNELS}
    # Track per-channel freshness: monotonic time of last sample received
    ch_last_time = {ch: 0.0 for ch in CHANNELS}
    lock = threading.Lock()
    raise_count = 0
    last_raise_time = 0.0
    packet_count = [0]
    last_packet_time = [time.monotonic()]
    prob_history = deque(maxlen=args.avg_window)
    stall_count = [0]

    def make_handler(ch):
        def callback(sender, data):
            now = time.monotonic()
            packet_count[0] += 1
            last_packet_time[0] = now
            samples = parse_packet(data)
            with lock:
                buffers[ch].extend(samples)
                ch_last_time[ch] = now
        return callback

    print(f"\nConnecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        for ch, uuid in EEG_UUIDS.items():
            await client.start_notify(uuid, make_handler(ch))
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

        # Verify stream
        print("Waiting for BLE data...")
        for _ in range(20):
            await asyncio.sleep(0.5)
            if packet_count[0] > 100:
                break
        if packet_count[0] < 50:
            print(f"ERROR: Only {packet_count[0]} packets. Power-cycle the Muse.")
            return
        print(f"Stream OK ({packet_count[0]} packets)")

        print("Calibrating... keep still for 3 seconds...")
        await asyncio.sleep(3)

        print()
        print("=" * 50)
        print("REAL-TIME BROW RAISE DETECTION")
        print(f"Model: {model_name}")
        print(f"Threshold: {args.threshold} | Cooldown: {args.cooldown}s")
        print("Raise your eyebrows to trigger detection")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        print()

        try:
            while True:
                await asyncio.sleep(0.05)

                now = time.monotonic()

                # Check for full stream stall (no packets at all)
                stale = now - last_packet_time[0]
                if stale > 3.0:
                    stall_count[0] += 1
                    prob_history.clear()
                    print(f"\r  !! STALL #{stall_count[0]}: no data for {stale:.1f}s — "
                          f"waiting for stream...                  ", end="", flush=True)
                    continue

                with lock:
                    sizes = {ch: len(buffers[ch]) for ch in CHANNELS}
                    if any(s < WINDOW for s in sizes.values()):
                        continue

                    # Track channel freshness for display (don't block on it)
                    ch_ages = {ch: now - ch_last_time[ch] for ch in CHANNELS}
                    max_age = max(ch_ages.values())

                    window = np.column_stack(
                        [list(buffers[ch])[-WINDOW:] for ch in CHANNELS])

                # V2 preprocessing + feature extraction
                window = preprocess_v2(window)
                feats = np.array(extract_features_v2(window)).reshape(1, -1)
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

                if needs_scaling:
                    feats = scaler.transform(feats)

                probs = model.predict_proba(feats)[0]
                p_raise = float(probs[1])
                prev_avg = np.mean(prob_history) if len(prob_history) > 0 else 0.0
                prob_history.append(p_raise)
                p_avg = float(np.mean(prob_history))

                # Live bar — show average, raw, and stream health
                bar_avg = "#" * int(p_avg * 30)
                age_ms = int(max_age * 1000)
                stream_indicator = "OK" if age_ms < 100 else f"{age_ms}ms"
                print(f"\r  avg: {p_avg:.2f} |{bar_avg:<30s}| "
                      f"raw: {p_raise:.2f}  ble: {stream_indicator}    ",
                      end="", flush=True)

                # Detection
                past_cooldown = (now - last_raise_time) > args.cooldown
                if args.mode == "peak":
                    # Peak mode: trigger on rising edge — avg crosses threshold
                    # from below. Catches the spike even if it doesn't sustain.
                    triggered = (p_avg >= args.threshold and
                                 prev_avg < args.threshold and
                                 past_cooldown)
                else:
                    # Average mode: trigger when avg is above threshold
                    triggered = (p_avg >= args.threshold and past_cooldown)

                if triggered:
                    raise_count += 1
                    last_raise_time = now
                    play(GESTURE_DETECT_SOUNDS[3])
                    print(f"\n  >>> RAISE #{raise_count} (conf={p_raise:.2f}) <<<")

        except KeyboardInterrupt:
            pass
        finally:
            try:
                await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
            except Exception:
                pass
            print(f"\n\nStopped. Total raises detected: {raise_count}")


asyncio.run(main())
import os
os._exit(0)
