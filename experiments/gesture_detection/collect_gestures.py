"""
Multi-gesture data collection with audio cues.

Cycles through 4 gesture types (blink, furrow, raise, clench) with rest periods.
Each gesture gets a distinct audio cue tone. Order is shuffled each round.

Usage:
    python -m experiments.gesture_detection.collect_gestures [--duration 120] [--output data/gesture_data/gesture_data.csv]
"""

import asyncio
import sys

# Must set event loop policy AND force COM MTA before any imports that touch
# COM/WinRT (sounddevice initializes COM as STA via PortAudio, which blocks bleak)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import ctypes
    ctypes.windll.ole32.CoInitializeEx(0, 0x0)  # COINIT_MULTITHREADED

import argparse
import time
import csv
import random
import numpy as np
from bleak import BleakClient

from core import MUSE_ADDRESS, CONTROL_UUID, EEG_UUIDS, CHANNELS, FS, GESTURE_LABELS
from core.audio import play, GESTURE_CUES, REST_TONE, COUNTDOWN_BEEP, DONE_SOUND
from core.features import raw_to_uv, parse_packet

# ─── Config ───
REST_DURATION = 5.0      # seconds of rest between gestures (includes 1s countdown)
GESTURE_WINDOW = 0.5     # seconds to label as gesture after cue
POST_GESTURE_PAUSE = 1.5 # extra pause after gesture before next rest starts
GESTURES = [1, 2, 3, 4]  # blink, furrow, raise, clench

GESTURE_INSTRUCTIONS = {
    1: "BLINK NOW!",
    2: "FURROW BROWS!",
    3: "RAISE BROWS!",
    4: "CLENCH JAW!",
}


async def main():
    parser = argparse.ArgumentParser(description="Multi-gesture EEG data collection")
    parser.add_argument("--duration", type=int, default=120, help="Total recording duration (seconds)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (auto-generated if omitted)")
    args = parser.parse_args()

    total_seconds = args.duration
    if args.output:
        csv_path = args.output
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"data/gesture_data/gesture_{ts}.csv"

    # Shared state
    raw_data = []  # list of (timestamp, channel, uv_value)
    t0 = None

    def make_handler(ch):
        def callback(sender, data):
            now = time.monotonic()
            samples = parse_packet(data)
            for j, uv in enumerate(samples):
                sample_t = now - (len(samples) - 1 - j) / FS
                raw_data.append((sample_t, ch, uv))
        return callback

    # Preview cue tones before connecting (no BLE timeout risk)
    print("Gesture cue tones (preview):")
    for gid, name in sorted(GESTURE_LABELS.items()):
        if gid == 0:
            continue
        play(GESTURE_CUES[gid])
        print(f"  {gid}: {name}")
        await asyncio.sleep(0.5)

    input("\nPut Muse on your head, press Enter to start... ")

    print(f"\nConnecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        # Start streaming immediately to keep connection alive
        for ch, uuid in EEG_UUIDS.items():
            await client.start_notify(uuid, make_handler(ch))
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

        print()
        print("=" * 55)
        print(f"Recording {total_seconds}s — REST then GESTURE on cue")
        print("Gestures: blink, furrow brows, raise brows, clench jaw")
        print("=" * 55)

        input("\nReady? Press Enter to begin... ")

        # Countdown
        for i in [3, 2, 1]:
            play(COUNTDOWN_BEEP)
            print(f"  Starting in {i}...")
            await asyncio.sleep(1)
        print()

        t0 = time.monotonic()
        cue_times = []  # list of (monotonic_time, gesture_label)
        elapsed = 0.0
        round_num = 0

        done = False
        while not done and elapsed < total_seconds:
            round_num += 1
            order = GESTURES.copy()
            random.shuffle(order)
            print(f"\n--- Round {round_num} ---")

            for gesture_id in order:
                gesture_name = GESTURE_LABELS[gesture_id].upper()

                # Check if enough time for rest + gesture
                if elapsed + REST_DURATION + GESTURE_WINDOW > total_seconds:
                    done = True
                    break

                # REST phase — tell user what's coming next
                play(REST_TONE)
                print(f"[{elapsed:5.1f}s] REST — relax... (next: {gesture_name})")
                await asyncio.sleep(REST_DURATION - 1.0)  # rest minus 1s countdown
                elapsed = time.monotonic() - t0

                if elapsed >= total_seconds:
                    done = True
                    break

                # 1-second countdown before gesture
                play(COUNTDOWN_BEEP)
                print(f"[{elapsed:5.1f}s] Get ready... {gesture_name}!")
                await asyncio.sleep(1.0)
                elapsed = time.monotonic() - t0

                # GESTURE cue
                cue_time = time.monotonic()
                cue_times.append((cue_time, gesture_id))
                play(GESTURE_CUES[gesture_id])
                instruction = GESTURE_INSTRUCTIONS[gesture_id]
                print(f"[{elapsed:5.1f}s] >>> {instruction} <<<")
                await asyncio.sleep(GESTURE_WINDOW)

                # Pause after gesture — let user recover
                await asyncio.sleep(POST_GESTURE_PAUSE)
                elapsed = time.monotonic() - t0

        # Stop streaming (gracefully handle disconnect)
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

    # Use channel with most samples as time reference
    ref_ch = max(CHANNELS, key=lambda c: len(ch_buffers[c]))
    ref_times = [t for t, _ in ch_buffers[ref_ch]]

    # Interpolate all channels to reference timeline
    ch_arrays = {}
    for ch in CHANNELS:
        if len(ch_buffers[ch]) < 2:
            ch_arrays[ch] = np.zeros(len(ref_times))
            continue
        times = np.array([t for t, _ in ch_buffers[ch]])
        vals = np.array([v for _, v in ch_buffers[ch]])
        ch_arrays[ch] = np.interp(ref_times, times, vals)

    # Label each sample (0=rest, 1=blink, 2=furrow, 3=raise, 4=clench)
    ref_times = np.array(ref_times)
    labels = np.zeros(len(ref_times), dtype=int)
    for cue_t, gesture_id in cue_times:
        mask = (ref_times >= cue_t) & (ref_times < cue_t + GESTURE_WINDOW)
        labels[mask] = gesture_id

    # Timestamps relative to start
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
    print(f"\nSaved to {csv_path}")
    print(f"Total samples:   {len(ref_times)}")
    print(f"Duration:        {rel_times[-1]:.1f}s")
    print(f"Rounds:          {round_num}")
    print()
    for gid, name in sorted(GESTURE_LABELS.items()):
        count = int(np.sum(labels == gid))
        n_cues = sum(1 for _, g in cue_times if g == gid) if gid > 0 else "-"
        print(f"  {name:8s} (label={gid}): {count:5d} samples, {n_cues} cues")


asyncio.run(main())
