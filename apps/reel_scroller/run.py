"""
Reel Scroller — scroll Instagram/TikTok Reels with brow raises.

Raise your eyebrows → next reel. Uses the trained raise detector
and simulates a down arrow key press to scroll.

Usage:
    python -m apps.reel_scroller.run [--threshold 0.2] [--key down]

Open Instagram Reels or TikTok in your browser, click on it to focus,
then start this script. Each brow raise scrolls to the next reel.

Requires macOS Accessibility permission for pyautogui.
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
import pyautogui
import tkinter as tk

from core import MUSE_ADDRESS, CONTROL_UUID, EEG_UUIDS, CHANNELS, FS
from core.audio import play, GESTURE_DETECT_SOUNDS
from core.features import preprocess_v2, extract_features_v2, parse_packet

warnings.filterwarnings("ignore", category=RuntimeWarning)
pyautogui.PAUSE = 0

WINDOW = 128


class StatusWidget:
    """Small always-on-top widget showing detection status."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reel Scroller")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)

        # Position: bottom-right corner
        self.root.geometry("260x120+50+50")

        # Title
        tk.Label(self.root, text="REEL SCROLLER", font=("Helvetica", 11, "bold"),
                 fg="#aaaaaa", bg="#1a1a1a").pack(pady=(8, 2))

        # Confidence bar
        bar_frame = tk.Frame(self.root, bg="#1a1a1a")
        bar_frame.pack(fill="x", padx=12, pady=2)

        self.conf_label = tk.Label(bar_frame, text="0.00", font=("Menlo", 11),
                                   fg="#888888", bg="#1a1a1a", width=4, anchor="e")
        self.conf_label.pack(side="left")

        bar_bg = tk.Frame(bar_frame, bg="#333333", height=16)
        bar_bg.pack(side="left", fill="x", expand=True, padx=(6, 0))
        bar_bg.pack_propagate(False)

        self.bar_fill = tk.Frame(bar_bg, bg="#4CAF50", height=16, width=0)
        self.bar_fill.place(x=0, y=0, relheight=1.0)

        self.bar_width = 0
        self.bar_bg = bar_bg

        # Status line: scrolls + BLE
        status_frame = tk.Frame(self.root, bg="#1a1a1a")
        status_frame.pack(fill="x", padx=12, pady=(4, 2))

        self.scroll_label = tk.Label(status_frame, text="Scrolls: 0",
                                     font=("Menlo", 11), fg="#888888", bg="#1a1a1a")
        self.scroll_label.pack(side="left")

        self.ble_label = tk.Label(status_frame, text="BLE: ...",
                                  font=("Menlo", 10), fg="#888888", bg="#1a1a1a")
        self.ble_label.pack(side="right")

        # Flash label (shows SCROLL! briefly)
        self.flash_label = tk.Label(self.root, text="", font=("Helvetica", 13, "bold"),
                                    fg="#1a1a1a", bg="#1a1a1a")
        self.flash_label.pack(pady=(0, 4))

        self._flash_after_id = None

    def update(self, p_avg, scroll_count, ble_status, triggered=False):
        """Update widget from any thread (schedules on main thread)."""
        self.root.after(0, self._do_update, p_avg, scroll_count, ble_status, triggered)

    def _do_update(self, p_avg, scroll_count, ble_status, triggered):
        # Confidence
        self.conf_label.config(text=f"{p_avg:.2f}")

        # Bar color based on level
        if triggered:
            color = "#FF9800"
        elif p_avg > 0.5:
            color = "#4CAF50"
        elif p_avg > 0.2:
            color = "#8BC34A"
        else:
            color = "#555555"

        self.bar_bg.update_idletasks()
        max_w = self.bar_bg.winfo_width()
        fill_w = int(p_avg * max_w)
        self.bar_fill.config(bg=color)
        self.bar_fill.place(x=0, y=0, relheight=1.0, width=fill_w)

        # Scroll count
        self.scroll_label.config(text=f"Scrolls: {scroll_count}")

        # BLE status
        if ble_status == "OK":
            self.ble_label.config(text="BLE: OK", fg="#4CAF50")
        elif ble_status == "STALL":
            self.ble_label.config(text="BLE: STALL", fg="#f44336")
        else:
            self.ble_label.config(text=f"BLE: {ble_status}", fg="#FF9800")

        # Flash on scroll
        if triggered:
            self.flash_label.config(text="SCROLL!", fg="#FF9800")
            if self._flash_after_id:
                self.root.after_cancel(self._flash_after_id)
            self._flash_after_id = self.root.after(
                500, lambda: self.flash_label.config(text="", fg="#1a1a1a"))

    def run(self):
        """Run the tkinter mainloop (call from main thread)."""
        self.root.mainloop()

    def stop(self):
        self.root.after(0, self.root.destroy)


async def run_detector(args, widget):
    # Load model
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    model_name = bundle["best_name"]
    needs_scaling = model_name != "Random Forest"

    # State
    buffers = {ch: deque(maxlen=WINDOW * 4) for ch in CHANNELS}
    lock = threading.Lock()
    packet_count = [0]
    last_packet_time = [time.monotonic()]
    prob_history = deque(maxlen=args.avg_window)
    scroll_count = 0
    last_scroll_time = 0.0

    def make_handler(ch):
        def callback(sender, data):
            packet_count[0] += 1
            last_packet_time[0] = time.monotonic()
            samples = parse_packet(data)
            with lock:
                buffers[ch].extend(samples)
        return callback

    print(f"Reel Scroller")
    print(f"  Model: {model_name}")
    print(f"  Threshold: {args.threshold} | Cooldown: {args.cooldown}s | Key: {args.key}")
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
            print(f"ERROR: No data. Power-cycle the Muse.")
            widget.stop()
            return

        print("Stream OK. Calibrating (3s)...")
        widget.update(0.0, 0, "OK")
        await asyncio.sleep(3)

        print("ACTIVE — raise eyebrows to scroll\n")

        try:
            while True:
                await asyncio.sleep(0.05)

                stale = time.monotonic() - last_packet_time[0]
                if stale > 3.0:
                    prob_history.clear()
                    widget.update(0.0, scroll_count, "STALL")
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

                # BLE status
                ble_status = "OK" if stale < 1.0 else f"{stale:.0f}s"

                # Peak detection
                now = time.monotonic()
                triggered = (p_avg >= args.threshold and
                             prev_avg < args.threshold and
                             (now - last_scroll_time) > args.cooldown)

                if triggered:
                    scroll_count += 1
                    last_scroll_time = now
                    pyautogui.press(args.key)
                    if not args.no_sound:
                        play(GESTURE_DETECT_SOUNDS[3])
                    print(f"  SCROLL #{scroll_count} (conf={p_avg:.2f})")

                widget.update(p_avg, scroll_count, ble_status, triggered)

        except KeyboardInterrupt:
            pass
        finally:
            try:
                await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
            except Exception:
                pass
            print(f"\nDone. Total scrolls: {scroll_count}")
            widget.stop()


def main():
    parser = argparse.ArgumentParser(description="Reel Scroller — brow raise to scroll")
    parser.add_argument("--model", type=str, default="data/raise_data/raise_model.pkl")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--cooldown", type=float, default=1.5)
    parser.add_argument("--avg-window", type=int, default=3)
    parser.add_argument("--key", type=str, default="down")
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--no-widget", action="store_true",
                        help="Disable the status widget")
    args = parser.parse_args()

    if args.no_widget:
        # Run without widget
        asyncio.run(run_detector(args, type('Dummy', (), {
            'update': lambda *a, **k: None,
            'stop': lambda *a: None})()))
        import os
        os._exit(0)

    # Create widget on main thread, run detector in background
    widget = StatusWidget()

    def run_async():
        asyncio.run(run_detector(args, widget))
        import os
        os._exit(0)

    detector_thread = threading.Thread(target=run_async, daemon=True)
    detector_thread.start()

    # tkinter must run on main thread on macOS
    widget.run()


if __name__ == "__main__":
    main()
