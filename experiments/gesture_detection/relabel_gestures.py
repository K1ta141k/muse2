"""
Re-label gesture data using peak detection to correct for reaction time.

For each gesture cue, searches for the actual EMG/EEG peak in the appropriate
channels (frontal for blink/furrow/raise, temporal for clench) and re-centers
the label window around the detected peak.

Usage:
    python -m experiments.gesture_detection.relabel_gestures [--input ...] [--output ...]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import CHANNELS, FS, GESTURE_LABELS

plt.style.use("dark_background")

# ─── Config ───
SEARCH_START = 0.1   # ignore first 100ms (too fast for reaction)
SEARCH_END = 1.0     # gesture should happen within 1s of cue
PRE_PEAK = 0.15      # label window: 150ms before peak
POST_PEAK = 0.35     # label window: 350ms after peak

# Which channels to search for each gesture type
GESTURE_CHANNELS = {
    1: ["AF7", "AF8"],     # blink — frontal
    2: ["AF7", "AF8"],     # furrow — frontal
    3: ["AF7", "AF8"],     # raise — frontal
    4: ["TP9", "TP10"],    # clench — temporal
}


def main():
    parser = argparse.ArgumentParser(description="Re-label gesture data using peak detection")
    parser.add_argument("--input", type=str, default="data/gesture_data/gesture_data.csv")
    parser.add_argument("--output", type=str, default="data/gesture_data/gesture_data_relabeled.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    t = df["timestamp"].values
    labels = df["label"].values

    print(f"Loaded {len(df)} samples from {args.input}")

    # Find cue times for each gesture type from label transitions
    new_labels = np.zeros(len(t), dtype=int)
    all_peaks = []  # (peak_time, gesture_id) for visualization

    for gesture_id in [1, 2, 3, 4]:
        name = GESTURE_LABELS[gesture_id]
        channels = GESTURE_CHANNELS[gesture_id]

        # Build detection signal from appropriate channels
        detect_signal = sum(np.abs(df[ch].values) for ch in channels)

        # Find cue start times: where label transitions to this gesture_id
        label_changes = np.diff((labels == gesture_id).astype(int), prepend=0)
        cue_indices = np.where(label_changes == 1)[0]
        cue_times = t[cue_indices]

        print(f"\n{name.upper()} (label={gesture_id}): {len(cue_times)} cues, channels={channels}")

        reactions = []
        for cue_t in cue_times:
            search_mask = (t >= cue_t + SEARCH_START) & (t <= cue_t + SEARCH_END)
            if not search_mask.any():
                print(f"  Cue at {cue_t:.2f}s — no search window, skipping")
                continue

            search_indices = np.where(search_mask)[0]
            peak_idx = search_indices[np.argmax(detect_signal[search_indices])]
            peak_time = t[peak_idx]
            reaction_ms = (peak_time - cue_t) * 1000

            # Apply label window centered on peak
            mask = (t >= peak_time - PRE_PEAK) & (t <= peak_time + POST_PEAK)
            new_labels[mask] = gesture_id

            reactions.append(reaction_ms)
            all_peaks.append((peak_time, gesture_id))
            print(f"  Cue at {cue_t:6.2f}s -> peak at {peak_time:6.2f}s  (reaction: {reaction_ms:.0f}ms)")

        if reactions:
            print(f"  Mean reaction: {np.mean(reactions):.0f}ms, Median: {np.median(reactions):.0f}ms")

    # ─── Save re-labeled CSV ───
    df["label"] = new_labels
    df.to_csv(args.output, index=False)

    print(f"\nSaved re-labeled data to {args.output}")
    for gid, name in sorted(GESTURE_LABELS.items()):
        count = int(np.sum(new_labels == gid))
        print(f"  {name:8s} (label={gid}): {count:5d} samples")

    # ─── Visualization ───
    colors = {"TP9": "#e6194b", "AF7": "#3cb44b", "AF8": "#4363d8", "TP10": "#f58231"}
    gesture_colors = {1: "#ffcc00", 2: "#ff6666", 3: "#66ff66", 4: "#6666ff"}

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    fig.suptitle("Gesture Re-labeling: Cue-based vs Peak-detected", fontsize=14, fontweight="bold")

    # Plot 1: Frontal channels with old labels
    ax = axes[0]
    frontal = np.abs(df["AF7"].values) + np.abs(df["AF8"].values)
    ax.plot(t, frontal, color="#3cb44b", linewidth=0.4, label="|AF7|+|AF8|")
    for gid in [1, 2, 3, 4]:
        old_changes = np.diff((labels == gid).astype(int), prepend=0)
        for start_idx in np.where(old_changes == 1)[0]:
            cue_t = t[start_idx]
            ax.axvspan(cue_t, cue_t + 0.5, alpha=0.25, color=gesture_colors[gid])
    ax.set_ylabel("|AF7|+|AF8|", fontsize=10)
    ax.set_title("OLD labels — window starts at cue", fontsize=11)
    ax.grid(True, alpha=0.1)

    # Plot 2: Temporal channels with old labels
    ax = axes[1]
    temporal = np.abs(df["TP9"].values) + np.abs(df["TP10"].values)
    ax.plot(t, temporal, color="#e6194b", linewidth=0.4, label="|TP9|+|TP10|")
    for gid in [1, 2, 3, 4]:
        old_changes = np.diff((labels == gid).astype(int), prepend=0)
        for start_idx in np.where(old_changes == 1)[0]:
            cue_t = t[start_idx]
            ax.axvspan(cue_t, cue_t + 0.5, alpha=0.25, color=gesture_colors[gid])
    ax.set_ylabel("|TP9|+|TP10|", fontsize=10)
    ax.set_title("Temporal channels with OLD labels", fontsize=11)
    ax.grid(True, alpha=0.1)

    # Plot 3: Re-labeled windows on frontal
    ax = axes[2]
    ax.plot(t, frontal, color="#3cb44b", linewidth=0.4)
    for peak_t, gid in all_peaks:
        ax.axvspan(peak_t - PRE_PEAK, peak_t + POST_PEAK, alpha=0.3, color=gesture_colors[gid])
        ax.axvline(peak_t, color=gesture_colors[gid], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylabel("|AF7|+|AF8|", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title("NEW labels — centered on detected peaks", fontsize=11)
    ax.grid(True, alpha=0.1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=gesture_colors[gid], alpha=0.5, label=GESTURE_LABELS[gid])
                       for gid in [1, 2, 3, 4]]
    axes[0].legend(handles=legend_elements, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig("data/gesture_data/relabel_comparison.png", dpi=150)
    print(f"\nSaved visualization to data/gesture_data/relabel_comparison.png")


if __name__ == "__main__":
    main()
