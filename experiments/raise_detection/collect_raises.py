"""
Brow-raise data collection with audio cues.

Longer rest periods than gesture collection to capture natural idle EEG.
Randomized rest durations (4-8s) to prevent anticipation artifacts.

Usage:
    python -m experiments.raise_detection.collect_raises [--duration 120] [--output ...]
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
import random
import numpy as np
from bleak import BleakClient

from core import MUSE_ADDRESS, CONTROL_UUID, EEG_UUIDS, CHANNELS, FS
from core.audio import play, GESTURE_CUES, REST_TONE, COUNTDOWN_BEEP, DONE_SOUND
from core.features import raw_to_uv, parse_packet

# ─── Config ───
RAISE_CUE = GESTURE_CUES[3]     # 1200Hz tone
RAISE_LABEL = 1                  # binary: 0=rest, 1=raise
GESTURE_WINDOW = 0.5             # seconds to label as raise after cue
POST_RAISE_PAUSE = 1.5           # recovery time after raise
REST_RANGE = (4.0, 8.0)          # randomized rest duration


async def main():
    parser = argparse.ArgumentParser(description="Brow-raise EEG data collection")
    parser.add_argument("--duration", type=int, default=120,
                        help="Total recording duration in seconds")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (auto-generated if omitted)")
    args = parser.parse_args()

    total_seconds = args.duration
    if args.output:
        csv_path = args.output
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/raise_data/raise_{ts}.csv"

    raw_data = []
    t0 = None

    def make_handler(ch):
        def callback(sender, data):
            now = time.monotonic()
            samples = parse_packet(data)
            for j, uv in enumerate(samples):
                sample_t = now - (len(samples) - 1 - j) / FS
                raw_data.append((sample_t, ch, uv))
        return callback

    # Preview cue tone
    print("Raise cue tone (preview):")
    play(RAISE_CUE)
    print("  You'll hear this — raise your eyebrows briefly")
    await asyncio.sleep(0.8)

    input("\nPut Muse on your head, press Enter to start... ")

    print(f"\nConnecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        # Start streaming
        for ch, uuid in EEG_UUIDS.items():
            await client.start_notify(uuid, make_handler(ch))
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

        # Verify data is flowing before proceeding
        print("Waiting for BLE data stream...")
        for attempt in range(20):
            await asyncio.sleep(0.5)
            if len(raw_data) > 100:
                break
        n = len(raw_data)
        if n < 50:
            print(f"ERROR: Only {n} samples after 10s — stream not working.")
            print("Try power-cycling the Muse (hold button 5s) and retry.")
            return
        print(f"Stream OK — {n} samples received")

        print()
        print("=" * 55)
        print(f"Recording {total_seconds}s — REST then RAISE BROWS on cue")
        print("Rest periods are randomized (4-8s). Stay relaxed.")
        print("=" * 55)

        input("\nReady? Press Enter to begin... ")

        # Countdown
        for i in [3, 2, 1]:
            play(COUNTDOWN_BEEP)
            print(f"  Starting in {i}...")
            await asyncio.sleep(1)
        print()

        t0 = time.monotonic()
        cue_times = []
        elapsed = 0.0
        raise_count = 0

        while elapsed < total_seconds:
            # REST phase — random duration
            rest_dur = random.uniform(*REST_RANGE)
            rest_end = elapsed + rest_dur
            if rest_end + GESTURE_WINDOW > total_seconds:
                # Not enough time for another raise, just rest until end
                remaining = total_seconds - elapsed
                print(f"[{elapsed:5.1f}s] REST — final rest ({remaining:.0f}s)...")
                await asyncio.sleep(remaining)
                break

            play(REST_TONE)
            print(f"[{elapsed:5.1f}s] REST — relax... ({rest_dur:.1f}s)")
            await asyncio.sleep(rest_dur - 1.0)
            elapsed = time.monotonic() - t0

            if elapsed >= total_seconds:
                break

            # 1-second countdown
            play(COUNTDOWN_BEEP)
            print(f"[{elapsed:5.1f}s] Get ready...")
            await asyncio.sleep(1.0)
            elapsed = time.monotonic() - t0

            # RAISE cue
            cue_time = time.monotonic()
            cue_times.append(cue_time)
            play(RAISE_CUE)
            raise_count += 1
            print(f"[{elapsed:5.1f}s] >>> RAISE BROWS! <<< (#{raise_count})")
            await asyncio.sleep(GESTURE_WINDOW)

            # Recovery pause
            await asyncio.sleep(POST_RAISE_PAUSE)
            elapsed = time.monotonic() - t0

        # Stop streaming
        try:
            await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
            for uuid in EEG_UUIDS.values():
                await client.stop_notify(uuid)
        except Exception as e:
            print(f"\n(Muse disconnected during cleanup: {e})")

    play(DONE_SOUND)
    print(f"\nRecording complete. Processing {len(raw_data)} raw samples...")

    # ─── Assemble into aligned rows ───
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

    # Label: 0=rest, 1=raise
    ref_times = np.array(ref_times)
    labels = np.zeros(len(ref_times), dtype=int)
    for cue_t in cue_times:
        mask = (ref_times >= cue_t) & (ref_times < cue_t + GESTURE_WINDOW)
        labels[mask] = RAISE_LABEL

    rel_times = ref_times - ref_times[0]

    # ─── Write CSV ───
    with open(csv_path, "w", newline="") as f:
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

    # ─── Summary ───
    n_raise = int(np.sum(labels == RAISE_LABEL))
    n_rest = int(np.sum(labels == 0))

    print(f"\nSaved to {csv_path}")
    print(f"Total samples:   {len(ref_times)}")
    print(f"Duration:        {rel_times[-1]:.1f}s")
    print(f"Raise cues:      {raise_count}")
    print(f"Raise samples:   {n_raise} (label=1)")
    print(f"Rest samples:    {n_rest} (label=0)")
    print(f"Rest ratio:      {n_rest / max(n_raise, 1):.1f}:1")


asyncio.run(main())
import os
os._exit(0)
