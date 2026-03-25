import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FS = 256
BLINK_WINDOW = 0.5  # label window duration after detected peak

df = pd.read_csv("blink_data.csv")
t = df["timestamp"].values
labels = df["label"].values

# Find original cue times from existing labels
blink_changes = np.diff(labels, prepend=0)
cue_times = t[blink_changes == 1]
print(f"Found {len(cue_times)} original cues")

# Use frontal channels (AF7 + AF8) — blinks are strongest there
frontal = np.abs(df["AF7"].values) + np.abs(df["AF8"].values)

# For each cue, search 0.1s–1.0s after for the actual spike peak
SEARCH_START = 0.1  # ignore first 100ms (too fast for human reaction)
SEARCH_END = 1.0    # blink should happen within 1s of cue

detected_peaks = []
for cue_t in cue_times:
    search_mask = (t >= cue_t + SEARCH_START) & (t <= cue_t + SEARCH_END)
    if not search_mask.any():
        detected_peaks.append(None)
        continue
    search_indices = np.where(search_mask)[0]
    peak_idx = search_indices[np.argmax(frontal[search_indices])]
    peak_time = t[peak_idx]
    reaction_ms = (peak_time - cue_t) * 1000
    detected_peaks.append(peak_time)
    print(f"  Cue at {cue_t:6.2f}s → peak at {peak_time:6.2f}s  (reaction: {reaction_ms:.0f}ms)")

# Filter out any misses
valid_peaks = [p for p in detected_peaks if p is not None]
reactions = [(p - c) * 1000 for p, c in zip(valid_peaks, cue_times) if p is not None]
print(f"\nMean reaction time: {np.mean(reactions):.0f}ms")
print(f"Median reaction time: {np.median(reactions):.0f}ms")

# Re-label: center a 0.5s window on each detected peak (0.15s before, 0.35s after)
PRE_PEAK = 0.15
POST_PEAK = 0.35
new_labels = np.zeros(len(t), dtype=int)
for peak_t in valid_peaks:
    mask = (t >= peak_t - PRE_PEAK) & (t <= peak_t + POST_PEAK)
    new_labels[mask] = 1

# --- Visualization: compare old vs new labels ---
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
fig.suptitle("Blink Re-labeling: Cue-based vs Peak-detected", fontsize=14, fontweight="bold")

colors = {"TP9": "#e6194b", "AF7": "#3cb44b", "AF8": "#4363d8", "TP10": "#f58231"}

# Plot 1: AF7 signal with OLD labels
ax = axes[0]
ax.plot(t, df["AF7"].values, color=colors["AF7"], linewidth=0.4)
for cue_t in cue_times:
    ax.axvspan(cue_t, cue_t + 0.5, alpha=0.3, color="#ffcc00")
ax.set_ylabel("AF7 (µV)", fontsize=10)
ax.set_title("OLD labels — window starts at cue (yellow)", fontsize=11, color="#ffcc00")
ax.grid(True, alpha=0.1)

# Plot 2: AF7 signal with NEW labels
ax = axes[1]
ax.plot(t, df["AF7"].values, color=colors["AF7"], linewidth=0.4)
for peak_t in valid_peaks:
    ax.axvspan(peak_t - PRE_PEAK, peak_t + POST_PEAK, alpha=0.3, color="#00ff88")
    ax.axvline(peak_t, color="#00ff88", linewidth=1, linestyle="--", alpha=0.6)
ax.set_ylabel("AF7 (µV)", fontsize=10)
ax.set_title("NEW labels — window centered on detected peak (green)", fontsize=11, color="#00ff88")
ax.grid(True, alpha=0.1)

# Plot 3: Overlay both on frontal average
ax = axes[2]
ax.plot(t, frontal, color="white", linewidth=0.3, alpha=0.7, label="|AF7|+|AF8|")
for i, cue_t in enumerate(cue_times):
    ax.axvspan(cue_t, cue_t + 0.5, alpha=0.2, color="#ffcc00",
               label="Old window" if i == 0 else None)
for i, peak_t in enumerate(valid_peaks):
    ax.axvspan(peak_t - PRE_PEAK, peak_t + POST_PEAK, alpha=0.2, color="#00ff88",
               label="New window" if i == 0 else None)
    ax.axvline(peak_t, color="#00ff88", linewidth=1, linestyle="--", alpha=0.5)
ax.set_ylabel("|AF7|+|AF8|", fontsize=10)
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_title("Comparison: Old (yellow) vs New (green) on frontal magnitude", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.1)

plt.tight_layout()
fig.savefig("relabel_comparison.png", dpi=150)
print(f"\nSaved visualization to relabel_comparison.png")

# Also zoom into 3 individual blinks for detail
pick = [0, len(valid_peaks)//2, len(valid_peaks)-1]
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("Zoom: Individual Blink Re-labeling", fontsize=13, fontweight="bold")

for col, idx in enumerate(pick):
    ax = axes2[col]
    cue_t = cue_times[idx]
    peak_t = valid_peaks[idx]
    win_start = cue_t - 0.3
    win_end = cue_t + 1.5
    mask = (t >= win_start) & (t <= win_end)

    ax.plot(t[mask] - cue_t, df["AF7"].values[mask], color=colors["AF7"], linewidth=1, label="AF7")
    ax.plot(t[mask] - cue_t, df["AF8"].values[mask], color=colors["AF8"], linewidth=1, label="AF8")

    # Old window
    ax.axvspan(0, 0.5, alpha=0.2, color="#ffcc00", label="Old window")
    # New window
    rt = peak_t - cue_t
    ax.axvspan(rt - PRE_PEAK, rt + POST_PEAK, alpha=0.2, color="#00ff88", label="New window")
    ax.axvline(rt, color="#00ff88", linewidth=2, linestyle="--", label=f"Peak ({rt*1000:.0f}ms)")
    ax.axvline(0, color="#ffcc00", linewidth=1, linestyle=":", alpha=0.7)

    ax.set_xlabel("Time rel. to cue (s)", fontsize=9)
    ax.set_title(f"Blink {idx+1}", fontsize=11)
    if col == 0:
        ax.set_ylabel("µV", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.1)

fig2.tight_layout()
fig2.savefig("relabel_zoom.png", dpi=150)
print("Saved zoom to relabel_zoom.png")

# Save re-labeled CSV
df["label"] = new_labels
df.to_csv("blink_data_relabeled.csv", index=False)
n_new_blink = int(new_labels.sum())
n_new_rest = int((new_labels == 0).sum())
print(f"\nSaved re-labeled data to blink_data_relabeled.csv")
print(f"  Blink samples: {n_new_blink}")
print(f"  Rest samples:  {n_new_rest}")

plt.show()
