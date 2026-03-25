import asyncio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, detrend
from bleak import BleakClient

plt.style.use("dark_background")

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
RECORD_SEC = 30
b60, a60 = iirnotch(60.0, 200.0, FS)


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


async def record(client, duration):
    buffers = {ch: [] for ch in CHANNELS}

    def make_handler(ch):
        def callback(sender, data):
            buffers[ch].extend(parse_packet(data))
        return callback

    for ch, uuid in EEG_UUIDS.items():
        await client.start_notify(uuid, make_handler(ch))
    await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))
    print(f"Recording {duration}s...")
    await asyncio.sleep(duration)
    await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
    for uuid in EEG_UUIDS.values():
        await client.stop_notify(uuid)

    min_len = min(len(buffers[ch]) for ch in CHANNELS)
    return {ch: np.array(buffers[ch][:min_len]) for ch in CHANNELS}


def detect_blinks(data):
    """
    Anomaly detection on frontal channels.
    Blinks = massive low-freq deflections on AF7/AF8.

    1. Notch filter + detrend
    2. Compute rolling envelope on |AF7| + |AF8|
    3. Z-score the envelope against a robust baseline (median + MAD)
    4. Threshold + group into events
    """
    # Preprocess
    filtered = {}
    for ch in CHANNELS:
        sig = data[ch].copy()
        sig = filtfilt(b60, a60, sig)
        sig = detrend(sig)
        filtered[ch] = sig

    # Frontal envelope: |AF7| + |AF8|, smoothed
    frontal = np.abs(filtered["AF7"]) + np.abs(filtered["AF8"])

    # Rolling RMS envelope (window = 50ms = ~13 samples)
    kernel = int(0.05 * FS)
    envelope = np.convolve(frontal, np.ones(kernel) / kernel, mode="same")

    # Robust z-score using median and MAD
    median = np.median(envelope)
    mad = np.median(np.abs(envelope - median))
    mad_std = mad * 1.4826  # scale MAD to approximate std
    z_scores = (envelope - median) / max(mad_std, 0.001)

    # Threshold: z > 3.5 is an anomaly
    THRESHOLD = 3.5
    anomaly_mask = z_scores > THRESHOLD

    # Group nearby anomalies into events (merge within 0.3s)
    anomaly_idx = np.where(anomaly_mask)[0]
    events = []
    if len(anomaly_idx) > 0:
        merge_gap = int(0.3 * FS)
        groups = np.split(anomaly_idx, np.where(np.diff(anomaly_idx) > merge_gap)[0] + 1)
        for g in groups:
            if len(g) < 3:  # skip tiny glitches
                continue
            start = g[0]
            end = g[-1]
            peak = g[np.argmax(frontal[g])]
            events.append({
                "start_idx": int(start),
                "end_idx": int(end),
                "peak_idx": int(peak),
                "start_sec": start / FS,
                "end_sec": end / FS,
                "peak_sec": peak / FS,
                "duration_ms": (end - start) / FS * 1000,
                "peak_amplitude": float(frontal[peak]),
                "z_score": float(z_scores[peak]),
            })

    return events, filtered, envelope, z_scores


async def main():
    print("=" * 50)
    print("WATER + ANOMALY-BASED BLINK DETECTION")
    print("=" * 50)
    print()
    print("Wet the Muse sensors with water, put it on.")
    print(f"You'll record {RECORD_SEC}s. Blink naturally")
    print("whenever you want — no cues, no timing pressure.")
    print()
    input("Press Enter to start... ")

    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}\n")
        data = await record(client, RECORD_SEC)

    n_samples = len(data[CHANNELS[0]])
    print(f"Recorded {n_samples} samples ({n_samples/FS:.1f}s)\n")

    # Detect blinks
    events, filtered, envelope, z_scores = detect_blinks(data)
    print(f"Detected {len(events)} blink events:\n")
    for i, ev in enumerate(events):
        print(f"  Blink {i+1:2d}:  t={ev['peak_sec']:5.2f}s  "
              f"duration={ev['duration_ms']:.0f}ms  "
              f"amplitude={ev['peak_amplitude']:.0f}  "
              f"z={ev['z_score']:.1f}")

    t = np.arange(n_samples) / FS
    colors = {"TP9": "#e6194b", "AF7": "#3cb44b", "AF8": "#4363d8", "TP10": "#f58231"}

    # --- Plot 1: All channels + detected blinks ---
    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.8]})
    fig.suptitle(f"Water Recording — {len(events)} Blinks Auto-Detected", fontsize=14, fontweight="bold")

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        ax.plot(t, filtered[ch], color=colors[ch], linewidth=0.4)
        for ev in events:
            ax.axvspan(ev["start_sec"], ev["end_sec"], alpha=0.25, color="#00ff88")
            ax.axvline(ev["peak_sec"], color="#00ff88", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"{ch}", fontsize=10, fontweight="bold", color=colors[ch])
        ax.grid(True, alpha=0.1)

    # Bottom: z-score envelope
    ax = axes[4]
    ax.plot(t, z_scores, color="white", linewidth=0.5, alpha=0.8)
    ax.axhline(3.5, color="#ff4444", linewidth=1, linestyle="--", label="Threshold (z=3.5)")
    for ev in events:
        ax.axvspan(ev["start_sec"], ev["end_sec"], alpha=0.25, color="#00ff88")
    ax.set_ylabel("Z-score", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    fig.savefig("water_blinks_detected.png", dpi=150)
    print("\nSaved water_blinks_detected.png")

    # --- Plot 2: Zoom into first 5 blinks ---
    n_zoom = min(5, len(events))
    if n_zoom > 0:
        fig2, axes2 = plt.subplots(n_zoom, 1, figsize=(14, 3 * n_zoom), sharex=False)
        if n_zoom == 1:
            axes2 = [axes2]
        fig2.suptitle("Zoom: Individual Detected Blinks (AF7 + AF8)", fontsize=13, fontweight="bold")

        for i in range(n_zoom):
            ev = events[i]
            ax = axes2[i]
            # 1s window around peak
            win_start = max(0, ev["peak_idx"] - FS)
            win_end = min(n_samples, ev["peak_idx"] + FS)
            t_win = (np.arange(win_start, win_end) - ev["peak_idx"]) / FS * 1000  # ms

            ax.plot(t_win, filtered["AF7"][win_start:win_end], color=colors["AF7"], linewidth=1, label="AF7")
            ax.plot(t_win, filtered["AF8"][win_start:win_end], color=colors["AF8"], linewidth=1, label="AF8")
            ax.axvline(0, color="#00ff88", linewidth=1.5, linestyle="--")
            ax.axvspan((ev["start_idx"] - ev["peak_idx"]) / FS * 1000,
                       (ev["end_idx"] - ev["peak_idx"]) / FS * 1000,
                       alpha=0.15, color="#00ff88")
            ax.set_ylabel(f"Blink {i+1}\n(µV)", fontsize=9)
            ax.set_title(f"t={ev['peak_sec']:.2f}s  |  {ev['duration_ms']:.0f}ms  |  z={ev['z_score']:.1f}",
                         fontsize=9, color="#00ff88")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.1)
            if i == n_zoom - 1:
                ax.set_xlabel("Time relative to peak (ms)", fontsize=10)

        plt.tight_layout()
        fig2.savefig("water_blinks_zoom.png", dpi=150)
        print("Saved water_blinks_zoom.png")

    # --- Save labeled CSV ---
    labels = np.zeros(n_samples, dtype=int)
    for ev in events:
        labels[ev["start_idx"]:ev["end_idx"]+1] = 1

    df = pd.DataFrame({
        "timestamp": t,
        **{ch: filtered[ch] for ch in CHANNELS},
        "label": labels,
    })
    df.to_csv("water_blink_data.csv", index=False)
    print(f"Saved water_blink_data.csv ({(labels==1).sum()} blink samples, {(labels==0).sum()} rest)")

    # --- Summary ---
    if len(events) > 0:
        durations = [ev["duration_ms"] for ev in events]
        amps = [ev["peak_amplitude"] for ev in events]
        zs = [ev["z_score"] for ev in events]
        print(f"\nBlink stats:")
        print(f"  Count:     {len(events)}")
        print(f"  Duration:  {np.mean(durations):.0f}ms avg  ({np.min(durations):.0f}-{np.max(durations):.0f}ms)")
        print(f"  Amplitude: {np.mean(amps):.0f} avg  ({np.min(amps):.0f}-{np.max(amps):.0f})")
        print(f"  Z-score:   {np.mean(zs):.1f} avg  ({np.min(zs):.1f}-{np.max(zs):.1f})")


asyncio.run(main())
