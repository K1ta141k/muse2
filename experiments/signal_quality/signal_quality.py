import asyncio
import time
import threading
import numpy as np
import pandas as pd
import sounddevice as sd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, iirnotch, filtfilt, detrend
from bleak import BleakClient
import os, json

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
b60, a60 = iirnotch(60.0, 200.0, FS)

CONDITIONS = ["dry", "water", "saline"]
DATA_DIR = "signal_quality_data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Sound ---
AUDIO_SR = 44100

def make_tone(freq, duration, volume=0.5):
    t = np.linspace(0, duration, int(AUDIO_SR * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * volume
    fade = int(AUDIO_SR * 0.01)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)

BLINK_BEEP = make_tone(880, 0.15, 0.6)
REST_TONE = make_tone(440, 0.08, 0.3)
COUNTDOWN_BEEP = make_tone(660, 0.08, 0.35)
DONE_SOUND = np.concatenate([make_tone(523, 0.12, 0.4), make_tone(659, 0.12, 0.4), make_tone(784, 0.2, 0.5)])

def play(sound):
    threading.Thread(target=lambda: sd.play(sound, AUDIO_SR), daemon=True).start()


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


async def record_with_cues(client):
    """
    Record ~25s per condition:
    - 3s rest
    - 5 blink cues spaced 3s apart (blink on beep)
    - 3s rest
    Total: ~21s
    """
    buffers = {ch: [] for ch in CHANNELS}
    cue_times = []

    def make_handler(ch):
        def callback(sender, data):
            buffers[ch].extend(parse_packet(data))
        return callback

    for ch, uuid in EEG_UUIDS.items():
        await client.start_notify(uuid, make_handler(ch))
    await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

    t0 = time.monotonic()

    # Rest
    play(REST_TONE)
    print("    REST — stay still (3s)...")
    await asyncio.sleep(3)

    # 5 blink cues
    for i in range(5):
        cue_t = time.monotonic() - t0
        cue_times.append(cue_t)
        play(BLINK_BEEP)
        print(f"    >>> BLINK {i+1}/5 <<<")
        await asyncio.sleep(3)

    # Rest
    play(REST_TONE)
    print("    REST — stay still (3s)...")
    await asyncio.sleep(3)

    await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
    for uuid in EEG_UUIDS.values():
        await client.stop_notify(uuid)

    min_len = min(len(buffers[ch]) for ch in CHANNELS)
    data = {ch: np.array(buffers[ch][:min_len]) for ch in CHANNELS}
    return data, cue_times


def compute_metrics(data, cue_times):
    """Compute metrics using cue-based blink windows."""
    n = len(data[CHANNELS[0]])

    # Preprocess
    filtered = {}
    for ch in CHANNELS:
        sig = data[ch].copy()
        sig = filtfilt(b60, a60, sig)
        sig = detrend(sig)
        filtered[ch] = sig

    # Blink mask: 0.2s–1.0s after each cue (reaction time window)
    blink_mask = np.zeros(n, dtype=bool)
    for ct in cue_times:
        s = int((ct + 0.2) * FS)
        e = int((ct + 1.0) * FS)
        s, e = max(0, s), min(n, e)
        blink_mask[s:e] = True
    rest_mask = ~blink_mask

    metrics = {}
    for ch in CHANNELS:
        sig = filtered[ch]
        rest_sig = sig[rest_mask]
        blink_sig = sig[blink_mask]

        nperseg = min(512, len(rest_sig))
        f, pxx_rest = welch(rest_sig, fs=FS, nperseg=nperseg)

        def bp(pxx, fmin, fmax):
            mask = (f >= fmin) & (f <= fmax)
            return np.trapezoid(pxx[mask], f[mask]) if mask.any() else 0

        # Per-blink peak-to-peak
        blink_ptps = []
        for ct in cue_times:
            s = int((ct + 0.2) * FS)
            e = int((ct + 1.0) * FS)
            s, e = max(0, s), min(n, e)
            if e > s:
                blink_ptps.append(np.ptp(sig[s:e]))

        avg_ptp = np.mean(blink_ptps) if blink_ptps else 0
        rest_std = np.std(rest_sig) if len(rest_sig) > 0 else 1

        metrics[ch] = {
            "rest_std": float(rest_std),
            "blink_ptp_avg": float(avg_ptp),
            "snr": float(avg_ptp / max(rest_std, 0.001)),
            "alpha_power": float(bp(pxx_rest, 8, 12)),
            "beta_power": float(bp(pxx_rest, 12, 30)),
            "noise_60hz": float(bp(pxx_rest, 58, 62)),
            "total_power": float(bp(pxx_rest, 0.5, 50)),
        }

    return metrics, filtered, blink_mask


async def main():
    print("=" * 60)
    print("SIGNAL QUALITY: Dry vs Water vs Saline (cue-based)")
    print("=" * 60)
    print()
    print("Each condition: 3s rest → 5 blink cues (beep) → 3s rest")
    print("Blink HARD on each beep.")
    print()

    all_data = {}
    all_filtered = {}
    all_metrics = {}
    all_cues = {}
    all_blink_masks = {}

    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}\n")

        for cond in CONDITIONS:
            print(f"{'='*40}")
            print(f"  CONDITION: {cond.upper()}")
            print(f"{'='*40}")
            if cond == "dry":
                input("  Put Muse on DRY (no solution). Press Enter... ")
            elif cond == "water":
                input("  Wet sensors with WATER. Press Enter... ")
            elif cond == "saline":
                input("  Apply SALINE to sensors. Press Enter... ")

            input("  Ready? Press Enter to start... ")

            # Countdown
            for i in [3, 2, 1]:
                play(COUNTDOWN_BEEP)
                print(f"    Starting in {i}...")
                await asyncio.sleep(1)

            data, cue_times = await record_with_cues(client)
            metrics, filtered, blink_mask = compute_metrics(data, cue_times)

            all_data[cond] = data
            all_filtered[cond] = filtered
            all_metrics[cond] = metrics
            all_cues[cond] = cue_times
            all_blink_masks[cond] = blink_mask

            # Save CSV
            n = len(data[CHANNELS[0]])
            df = pd.DataFrame({
                "timestamp": np.arange(n) / FS,
                **{ch: data[ch] for ch in CHANNELS},
                "label": blink_mask.astype(int),
            })
            df.to_csv(f"{DATA_DIR}/{cond}_raw.csv", index=False)

            play(DONE_SOUND)
            avg_snr = np.mean([metrics[ch]["snr"] for ch in CHANNELS])
            print(f"    Done! SNR={avg_snr:.1f}  ({n} samples)\n")

    # Save metrics
    with open(f"{DATA_DIR}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ─── Plots ───
    print("Generating plots...")
    colors_cond = {"dry": "#e6194b", "water": "#4363d8", "saline": "#3cb44b"}
    colors_ch = {"TP9": "#e6194b", "AF7": "#3cb44b", "AF8": "#4363d8", "TP10": "#f58231"}

    # --- Plot 1: AF7 waveforms with cue windows ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle("AF7 — Blink Cue Windows (Dry vs Water vs Saline)", fontsize=14, fontweight="bold")

    for i, cond in enumerate(CONDITIONS):
        ax = axes[i]
        sig = all_filtered[cond]["AF7"]
        t = np.arange(len(sig)) / FS
        ax.plot(t, sig, color=colors_cond[cond], linewidth=0.4)
        for ct in all_cues[cond]:
            ax.axvspan(ct + 0.2, ct + 1.0, alpha=0.2, color="#ffcc00")
        snr = np.mean([all_metrics[cond][ch]["snr"] for ch in CHANNELS])
        ax.set_ylabel(f"{cond.upper()}\nSNR={snr:.1f}", fontsize=10,
                       fontweight="bold", color=colors_cond[cond])
        ax.grid(True, alpha=0.1)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{DATA_DIR}/waveforms.png", dpi=150)

    # --- Plot 2: PSD (rest only) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PSD — Rest Segments Only", fontsize=14, fontweight="bold")

    for i, ch in enumerate(CHANNELS):
        ax = axes[i // 2, i % 2]
        for cond in CONDITIONS:
            sig = all_filtered[cond][ch]
            rest = sig[~all_blink_masks[cond]]
            nperseg = min(512, len(rest))
            f, pxx = welch(rest, fs=FS, nperseg=nperseg)
            mask = f <= 60
            ax.semilogy(f[mask], pxx[mask], color=colors_cond[cond], linewidth=1.2, label=cond)
        ax.axvspan(8, 12, alpha=0.15, color="green")
        ax.set_title(ch, fontsize=12, fontweight="bold", color=colors_ch[ch])
        ax.set_xlabel("Hz")
        ax.set_ylabel("µV²/Hz")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.1)
        ax.set_xlim(0, 60)

    plt.tight_layout()
    fig.savefig(f"{DATA_DIR}/psd_comparison.png", dpi=150)

    # --- Plot 3: Metrics bars ---
    metric_names = ["rest_std", "blink_ptp_avg", "snr", "alpha_power"]
    metric_labels = ["Rest Noise (std)", "Blink Amplitude (p2p)", "SNR", "Alpha Power"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Signal Quality Metrics", fontsize=14, fontweight="bold")

    for mi, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[mi]
        x = np.arange(3)
        vals, errs = [], []
        for cond in CONDITIONS:
            ch_vals = [all_metrics[cond][ch][mname] for ch in CHANNELS]
            vals.append(np.mean(ch_vals))
            errs.append(np.std(ch_vals))

        ax.bar(x, vals, yerr=errs, color=[colors_cond[c] for c in CONDITIONS],
               capsize=5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in CONDITIONS])
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.1, axis="y")

        best = CONDITIONS[np.argmin(vals)] if mname == "rest_std" else CONDITIONS[np.argmax(vals)]
        ax.set_xlabel(f"Best: {best.upper()}", fontsize=9, color=colors_cond[best], fontweight="bold")

    plt.tight_layout()
    fig.savefig(f"{DATA_DIR}/metrics_comparison.png", dpi=150)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SIGNAL QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25s}", end="")
    for cond in CONDITIONS:
        print(f"{cond.upper():>12s}", end="")
    print(f"{'BEST':>12s}")
    print("-" * 73)

    for mname, mlabel in zip(metric_names, metric_labels):
        vals = {c: np.mean([all_metrics[c][ch][mname] for ch in CHANNELS]) for c in CONDITIONS}
        best = min(vals, key=vals.get) if mname == "rest_std" else max(vals, key=vals.get)
        print(f"{mlabel:<25s}", end="")
        for cond in CONDITIONS:
            print(f"{vals[cond]:>12.1f}", end="")
        print(f"{best.upper():>12s}")

    print(f"\nPlots saved to {DATA_DIR}/")


asyncio.run(main())
