"""
Collect raise data using the live model as labeler.

Records raw EEG while running the raise detector. Detections are auto-labeled
as raise events. After collection, review and correct labels interactively.

Usage:
    python -m experiments.raise_detection.collect_with_model [--duration 120] [--threshold 0.3]
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import ctypes
    ctypes.windll.ole32.CoInitializeEx(0, 0x0)

import argparse
import time
import csv
import threading
import pickle
import numpy as np
from collections import deque
from datetime import datetime
from bleak import BleakClient

from core import MUSE_ADDRESS, CONTROL_UUID, EEG_UUIDS, CHANNELS, FS
from core.audio import play, GESTURE_DETECT_SOUNDS, COUNTDOWN_BEEP, DONE_SOUND, make_tone, AUDIO_SR
from core.features import preprocess_v2, extract_features_v2, raw_to_uv, parse_packet

WINDOW = 128

# Sounds
MISS_BEEP = make_tone(200, 0.15, 0.4)      # low tone — mark a missed raise
UNDO_BEEP = make_tone(400, 0.1, 0.3)       # mid tone — undo last detection


async def main():
    parser = argparse.ArgumentParser(description="Collect raise data with live model")
    parser.add_argument("--model", type=str, default="data/raise_data/raise_model.pkl")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--cooldown", type=float, default=0.8)
    parser.add_argument("--avg-window", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raise_data/raise_{ts}.csv"

    # Load model
    print("Loading model...")
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    model_name = bundle["best_name"]
    needs_scaling = model_name != "Random Forest"

    # Shared state
    raw_data = []  # (monotonic_time, channel, uv)
    buffers = {ch: deque(maxlen=WINDOW * 4) for ch in CHANNELS}
    ch_last_time = {ch: 0.0 for ch in CHANNELS}
    lock = threading.Lock()
    packet_count = [0]
    last_packet_time = [time.monotonic()]
    prob_history = deque(maxlen=args.avg_window)

    # Detection log — mutable, user can undo/add
    detection_times = []   # monotonic times of confirmed detections
    removed_times = []     # monotonic times of undone detections
    manual_times = []      # monotonic times of manually marked raises

    def make_handler(ch):
        def callback(sender, data):
            now = time.monotonic()
            packet_count[0] += 1
            last_packet_time[0] = now
            samples = parse_packet(data)
            with lock:
                buffers[ch].extend(samples)
                ch_last_time[ch] = now
            for j, uv in enumerate(samples):
                sample_t = now - (len(samples) - 1 - j) / FS
                raw_data.append((sample_t, ch, uv))
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

        await asyncio.sleep(2)

        print()
        print("=" * 60)
        print(f"COLLECTING WITH LIVE MODEL ({args.duration}s)")
        print(f"Model: {model_name} | Thresh: {args.threshold} | Avg: {args.avg_window}")
        print()
        print("  Do raises naturally. Model auto-labels detections.")
        print("  After recording, you'll review and fix labels.")
        print("  Press Ctrl+C to stop early.")
        print("=" * 60)
        print()

        t0 = time.monotonic()
        raise_count = 0
        last_raise_time = 0.0

        try:
            while (time.monotonic() - t0) < args.duration:
                await asyncio.sleep(0.05)

                now = time.monotonic()
                stale = now - last_packet_time[0]
                if stale > 2.0:
                    prob_history.clear()
                    print(f"\r  !! STALL: no data for {stale:.1f}s          ",
                          end="", flush=True)
                    continue

                with lock:
                    sizes = {ch: len(buffers[ch]) for ch in CHANNELS}
                    if any(s < WINDOW for s in sizes.values()):
                        continue

                    ch_ages = {ch: now - ch_last_time[ch] for ch in CHANNELS}
                    max_age = max(ch_ages.values())
                    if max_age > 0.5:
                        prob_history.clear()
                        continue

                    window = np.column_stack(
                        [list(buffers[ch])[-WINDOW:] for ch in CHANNELS])

                window = preprocess_v2(window)
                feats = np.array(extract_features_v2(window)).reshape(1, -1)
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

                if needs_scaling:
                    feats = scaler.transform(feats)

                probs = model.predict_proba(feats)[0]
                p_raise = float(probs[1])
                prob_history.append(p_raise)
                p_avg = float(np.mean(prob_history))

                elapsed = time.monotonic() - t0
                bar = "#" * int(p_avg * 30)
                remaining = args.duration - elapsed
                print(f"\r  [{elapsed:5.1f}s] avg: {p_avg:.2f} |{bar:<30s}| "
                      f"raises: {raise_count}  ({remaining:.0f}s left)  ",
                      end="", flush=True)

                now = time.monotonic()
                if p_avg >= args.threshold and (now - last_raise_time) > args.cooldown:
                    raise_count += 1
                    last_raise_time = now
                    detection_times.append(now)
                    play(GESTURE_DETECT_SOUNDS[3])
                    print(f"\n  >>> RAISE #{raise_count} (avg={p_avg:.2f}) "
                          f"[{elapsed:.1f}s] <<<")

        except KeyboardInterrupt:
            pass
        finally:
            try:
                await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
            except Exception:
                pass

    play(DONE_SOUND)
    duration = time.monotonic() - t0 if 't0' in dir() else 0
    print(f"\n\nRecording complete. {len(raw_data)} raw samples, "
          f"{raise_count} detections.")

    # ─── Review phase ───
    print(f"\n{'=' * 60}")
    print("REVIEW DETECTIONS")
    print("=" * 60)
    print("For each detection, press:")
    print("  Enter  = keep (correct detection)")
    print("  d      = delete (false positive)")
    print("  q      = done reviewing")
    print()

    confirmed = []
    for i, det_t in enumerate(detection_times):
        t_rel = det_t - t0
        resp = input(f"  Detection #{i+1} at t={t_rel:.1f}s — keep? [Enter/d/q]: ").strip().lower()
        if resp == 'q':
            # Keep remaining as-is
            confirmed.extend(detection_times[i:])
            break
        elif resp == 'd':
            removed_times.append(det_t)
            print(f"    Removed.")
        else:
            confirmed.append(det_t)

    print(f"\nAdd any missed raises? Enter timestamps (seconds), empty to skip.")
    print(f"  (Recording was {duration:.0f}s long)")
    while True:
        resp = input("  Missed raise at t=? (or Enter to finish): ").strip()
        if not resp:
            break
        try:
            t_sec = float(resp)
            manual_times.append(t0 + t_sec)
            print(f"    Added raise at t={t_sec:.1f}s")
        except ValueError:
            print(f"    Invalid, try again")

    all_raise_times = confirmed + manual_times
    print(f"\nFinal: {len(all_raise_times)} raises "
          f"({len(confirmed)} auto + {len(manual_times)} manual, "
          f"{len(removed_times)} removed)")

    # ─── Assemble CSV ───
    print(f"\nProcessing {len(raw_data)} raw samples...")
    raw_data.sort(key=lambda x: x[0])

    ch_buffers = {ch: [] for ch in CHANNELS}
    for ts, ch, uv in raw_data:
        ch_buffers[ch].append((ts, uv))

    ref_ch = max(CHANNELS, key=lambda c: len(ch_buffers[c]))
    ref_times = [t for t, _ in ch_buffers[ref_ch]]

    ch_arrays = {}
    for ch in CHANNELS:
        if len(ch_buffers[ch]) < 2:
            ch_arrays[ch] = np.zeros(len(ref_times))
            continue
        times = np.array([t for t, _ in ch_buffers[ch]])
        vals = np.array([v for _, v in ch_buffers[ch]])
        ch_arrays[ch] = np.interp(ref_times, times, vals)

    ref_times = np.array(ref_times)
    labels = np.zeros(len(ref_times), dtype=int)
    GESTURE_WINDOW = 0.5
    for raise_t in all_raise_times:
        mask = (ref_times >= raise_t - 0.15) & (ref_times < raise_t + 0.35)
        labels[mask] = 1

    rel_times = ref_times - ref_times[0]

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "TP9", "AF7", "AF8", "TP10", "label"])
        for i in range(len(ref_times)):
            writer.writerow([
                f"{rel_times[i]:.6f}",
                f"{ch_arrays['TP9'][i]:.4f}",
                f"{ch_arrays['AF7'][i]:.4f}",
                f"{ch_arrays['AF8'][i]:.4f}",
                f"{ch_arrays['TP10'][i]:.4f}",
                labels[i],
            ])

    n_raise = int(np.sum(labels == 1))
    n_rest = int(np.sum(labels == 0))
    print(f"\nSaved to {args.output}")
    print(f"Total samples:   {len(ref_times)}")
    print(f"Duration:        {rel_times[-1]:.1f}s")
    print(f"Raise events:    {len(all_raise_times)}")
    print(f"Raise samples:   {n_raise} (label=1)")
    print(f"Rest samples:    {n_rest} (label=0)")


asyncio.run(main())
import os
os._exit(0)
