"""
Collect raise data using the live model as labeler with real-time corrections.

Records raw EEG while running the raise detector. Corrections during recording:
  Right arrow = false positive (undo last detection)
  Left arrow  = false negative (mark a missed raise — finds peak in last ~2s)

Usage:
    python -m experiments.raise_detection.collect_with_model [--duration 120] [--threshold 0.2]
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
import warnings
import numpy as np
from collections import deque
from datetime import datetime
from bleak import BleakClient

warnings.filterwarnings("ignore", category=RuntimeWarning)

from core import MUSE_ADDRESS, CONTROL_UUID, EEG_UUIDS, CHANNELS, FS
from core.audio import play, GESTURE_DETECT_SOUNDS, COUNTDOWN_BEEP, DONE_SOUND, make_tone
from core.features import preprocess_v2, extract_features_v2, raw_to_uv, parse_packet

WINDOW = 128
LOOKBACK_SEC = 2.0  # how far back to search for missed raise peaks

# Feedback sounds
FP_BEEP = make_tone(300, 0.12, 0.4)   # low — false positive removed
FN_BEEP = make_tone(900, 0.12, 0.4)   # high — false negative added


class KeyListener:
    """Non-blocking keyboard listener using pynput or fallback."""

    def __init__(self):
        self.events = deque(maxlen=20)
        self._thread = None

    def start(self):
        """Start listening in background thread."""
        try:
            self._start_pynput()
        except ImportError:
            self._start_stdin()

    def _start_pynput(self):
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.right:
                    self.events.append("fp")
                elif key == keyboard.Key.left:
                    self.events.append("fn")
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()

    def _start_stdin(self):
        """Fallback: read from stdin in raw mode."""
        import tty
        import termios
        import select

        self._old_settings = termios.tcgetattr(sys.stdin)

        def reader():
            try:
                tty.setcbreak(sys.stdin.fileno())
                while True:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        ch = sys.stdin.read(1)
                        if ch == '\x1b':
                            ch2 = sys.stdin.read(1)
                            if ch2 == '[':
                                ch3 = sys.stdin.read(1)
                                if ch3 == 'C':  # right arrow
                                    self.events.append("fp")
                                elif ch3 == 'D':  # left arrow
                                    self.events.append("fn")
            except Exception:
                pass

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def poll(self):
        """Return next event or None."""
        if self.events:
            return self.events.popleft()
        return None

    def stop(self):
        if hasattr(self, '_old_settings'):
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


def find_peak_in_buffer(raw_data, t_now, t0, lookback=LOOKBACK_SEC):
    """Search the last `lookback` seconds of raw data for a frontal peak.

    Returns the monotonic time of the peak, or None if not enough data.
    """
    cutoff = t_now - lookback

    # Collect recent frontal samples (AF7=idx1, AF8=idx2)
    recent = [(t, ch, uv) for t, ch, uv in raw_data
              if t >= cutoff and ch in ("AF7", "AF8")]

    if len(recent) < 20:
        return None

    # Build time-aligned frontal signal
    af7 = [(t, uv) for t, ch, uv in recent if ch == "AF7"]
    af8 = [(t, uv) for t, ch, uv in recent if ch == "AF8"]

    if not af7 or not af8:
        return None

    # Use AF7+AF8 absolute amplitude to find peak
    times_7 = np.array([t for t, _ in af7])
    vals_7 = np.array([abs(v) for _, v in af7])
    times_8 = np.array([t for t, _ in af8])
    vals_8 = np.array([abs(v) for _, v in af8])

    # Interpolate to common timeline
    all_times = np.union1d(times_7, times_8)
    interp_7 = np.interp(all_times, times_7, vals_7)
    interp_8 = np.interp(all_times, times_8, vals_8)
    combined = interp_7 + interp_8

    peak_idx = np.argmax(combined)
    return float(all_times[peak_idx])


async def main():
    parser = argparse.ArgumentParser(description="Collect raise data with live model")
    parser.add_argument("--model", type=str, default="data/raise_data/raise_model.pkl")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--cooldown", type=float, default=0.8)
    parser.add_argument("--avg-window", type=int, default=3)
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
    raw_data = []
    buffers = {ch: deque(maxlen=WINDOW * 4) for ch in CHANNELS}
    lock = threading.Lock()
    packet_count = [0]
    last_packet_time = [time.monotonic()]
    prob_history = deque(maxlen=args.avg_window)

    # Label tracking
    raise_times = []     # monotonic times of confirmed raise events
    fp_count = [0]
    fn_count = [0]

    def make_handler(ch):
        def callback(sender, data):
            now = time.monotonic()
            packet_count[0] += 1
            last_packet_time[0] = now
            samples = parse_packet(data)
            with lock:
                buffers[ch].extend(samples)
            for j, uv in enumerate(samples):
                sample_t = now - (len(samples) - 1 - j) / FS
                raw_data.append((sample_t, ch, uv))
        return callback

    # Start key listener
    keys = KeyListener()
    keys.start()

    print(f"\nConnecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        for ch, uuid in EEG_UUIDS.items():
            await client.start_notify(uuid, make_handler(ch))
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

        print("Waiting for BLE data...")
        for _ in range(20):
            await asyncio.sleep(0.5)
            if packet_count[0] > 100:
                break
        if packet_count[0] < 50:
            print(f"ERROR: Only {packet_count[0]} packets. Power-cycle the Muse.")
            keys.stop()
            return
        print(f"Stream OK ({packet_count[0]} packets)")

        await asyncio.sleep(2)

        print()
        print("=" * 60)
        print(f"COLLECTING WITH LIVE MODEL ({args.duration}s)")
        print(f"Model: {model_name} | Thresh: {args.threshold} | Avg: {args.avg_window}")
        print()
        print("  Do raises naturally. Model auto-labels detections.")
        print("  RIGHT ARROW = undo last (false positive)")
        print("  LEFT ARROW  = mark missed raise (false negative)")
        print("  Ctrl+C = stop early")
        print("=" * 60)
        print()

        t0 = time.monotonic()
        raise_count = 0
        last_raise_time = 0.0

        try:
            while (time.monotonic() - t0) < args.duration:
                await asyncio.sleep(0.05)

                # ─── Check for key corrections ───
                key_event = keys.poll()
                if key_event == "fp":
                    # False positive — remove most recent detection
                    if raise_times:
                        removed_t = raise_times.pop()
                        fp_count[0] += 1
                        elapsed_r = removed_t - t0
                        play(FP_BEEP)
                        print(f"\n  --- UNDO detection at t={elapsed_r:.1f}s "
                              f"(FP #{fp_count[0]}) ---")
                    else:
                        print(f"\n  --- No detection to undo ---")

                elif key_event == "fn":
                    # False negative — find peak in last 2s
                    now_fn = time.monotonic()
                    peak_t = find_peak_in_buffer(raw_data, now_fn, t0)
                    if peak_t is not None:
                        raise_times.append(peak_t)
                        fn_count[0] += 1
                        elapsed_p = peak_t - t0
                        play(FN_BEEP)
                        print(f"\n  +++ ADDED missed raise at t={elapsed_p:.1f}s "
                              f"(FN #{fn_count[0]}) +++")
                    else:
                        print(f"\n  +++ Not enough data to find peak +++")

                # ─── Model inference ───
                now = time.monotonic()
                stale = now - last_packet_time[0]
                if stale > 3.0:
                    prob_history.clear()
                    print(f"\r  !! STALL: no data for {stale:.1f}s          ",
                          end="", flush=True)
                    continue

                with lock:
                    sizes = {ch: len(buffers[ch]) for ch in CHANNELS}
                    if any(s < WINDOW for s in sizes.values()):
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
                prev_avg = float(np.mean(prob_history)) if len(prob_history) > 0 else 0.0
                prob_history.append(p_raise)
                p_avg = float(np.mean(prob_history))

                elapsed = time.monotonic() - t0
                remaining = args.duration - elapsed
                n_raises = len(raise_times)
                bar = "#" * int(p_avg * 30)
                print(f"\r  [{elapsed:5.1f}s] avg: {p_avg:.2f} |{bar:<30s}| "
                      f"raises: {n_raises}  fp: {fp_count[0]}  fn: {fn_count[0]}  "
                      f"({remaining:.0f}s left)  ",
                      end="", flush=True)

                # Peak detection — rising edge
                now = time.monotonic()
                if (p_avg >= args.threshold and
                        prev_avg < args.threshold and
                        (now - last_raise_time) > args.cooldown):
                    raise_count += 1
                    last_raise_time = now
                    raise_times.append(now)
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
            keys.stop()

    play(DONE_SOUND)
    print(f"\n\nRecording complete. {len(raw_data)} raw samples.")
    print(f"  Detections: {raise_count}")
    print(f"  FP removed: {fp_count[0]}")
    print(f"  FN added:   {fn_count[0]}")
    print(f"  Final raises: {len(raise_times)}")

    # ─── Assemble CSV ───
    print(f"\nProcessing...")
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

    # Label windows around each raise time
    ref_times = np.array(ref_times)
    raw_aligned = np.column_stack([ch_arrays[ch] for ch in CHANNELS])
    labels = np.zeros(len(ref_times), dtype=int)

    for raise_t in raise_times:
        # Find the actual peak near this time in the aligned data
        # Search ±0.5s around the event time
        search_mask = (ref_times >= raise_t - 0.5) & (ref_times < raise_t + 0.5)
        search_idx = np.where(search_mask)[0]

        if len(search_idx) > 0:
            # Peak on frontal channels
            frontal = np.abs(raw_aligned[search_idx, 1]) + np.abs(raw_aligned[search_idx, 2])
            peak_local = np.argmax(frontal)
            peak_idx = search_idx[peak_local]

            # Label 0.5s window centered on peak (150ms before, 350ms after)
            pre_samples = int(0.15 * FS)
            post_samples = int(0.35 * FS)
            start = max(0, peak_idx - pre_samples)
            end = min(len(labels), peak_idx + post_samples)
            labels[start:end] = 1
        else:
            # Fallback: label around the raw time
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
    print(f"Raise events:    {len(raise_times)}")
    print(f"Raise samples:   {n_raise} (label=1)")
    print(f"Rest samples:    {n_rest} (label=0)")


asyncio.run(main())
import os
os._exit(0)
