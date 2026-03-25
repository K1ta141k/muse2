import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

df = pd.read_csv("blink_data.csv")

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
COLORS = {"TP9": "#e6194b", "AF7": "#3cb44b", "AF8": "#4363d8", "TP10": "#f58231"}
FS = 256

t = df["timestamp"].values
labels = df["label"].values

# Find blink window start/end times
blink_changes = np.diff(labels, prepend=0)
blink_starts = t[blink_changes == 1]
blink_ends = t[blink_changes == -1] if np.any(blink_changes == -1) else np.array([])
# Handle case where recording ends during a blink
if len(blink_starts) > len(blink_ends):
    blink_ends = np.append(blink_ends, t[-1])

n_blinks = len(blink_starts)

# --- Figure ---
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0f0f1a")
gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1.3], hspace=0.35, wspace=0.3)

fig.suptitle(f"Blink Experiment — {n_blinks} cues, {len(df)} samples, {t[-1]:.0f}s",
             fontsize=15, fontweight="bold", color="white")

# --- Top 4 rows: one channel each, full timeline with blink windows highlighted ---
for i, ch in enumerate(CHANNELS):
    ax = fig.add_subplot(gs[i, :])
    ax.set_facecolor("#16213e")
    signal = df[ch].values

    ax.plot(t, signal, color=COLORS[ch], linewidth=0.4, alpha=0.9, label=f"{ch} signal")

    # Shade blink windows
    for idx, (s, e) in enumerate(zip(blink_starts, blink_ends)):
        ax.axvspan(s, e, alpha=0.25, color="#ffcc00", zorder=0,
                   label="Blink cue" if idx == 0 else None)

    ax.legend(fontsize=7, loc="upper right", facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="#ddd")

    ax.set_ylabel(f"{ch} (µV)", color=COLORS[ch], fontsize=10, fontweight="bold")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.1, color="#fff")
    for spine in ax.spines.values():
        spine.set_color("#333")
    if i < 3:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Time (s)", color="#aaa", fontsize=10)

# --- Bottom row: PSD comparison (blink vs rest) for AF7 and AF8 ---
blink_mask = labels == 1
rest_mask = labels == 0

for j, ch in enumerate(["AF7", "AF8"]):
    ax = fig.add_subplot(gs[4, j])
    ax.set_facecolor("#16213e")

    sig_blink = df[ch].values[blink_mask]
    sig_rest = df[ch].values[rest_mask]

    nperseg = min(512, len(sig_blink), len(sig_rest))

    if len(sig_blink) >= nperseg:
        f_b, pxx_b = welch(sig_blink, fs=FS, nperseg=nperseg)
        ax.semilogy(f_b[f_b <= 60], pxx_b[f_b <= 60], color="#ffcc00", linewidth=1.2, label="Blink windows")

    if len(sig_rest) >= nperseg:
        f_r, pxx_r = welch(sig_rest, fs=FS, nperseg=nperseg)
        ax.semilogy(f_r[f_r <= 60], pxx_r[f_r <= 60], color="#66ccff", linewidth=1.2, label="Rest")

    ax.axvspan(8, 12, alpha=0.15, color="green")
    ax.set_xlim(0, 60)
    ax.set_xlabel("Frequency (Hz)", color="#aaa", fontsize=10)
    ax.set_ylabel("µV²/Hz", color="#aaa", fontsize=10)
    ax.set_title(f"{ch} — PSD: Blink vs Rest", color="white", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.grid(True, alpha=0.1, color="#fff")
    for spine in ax.spines.values():
        spine.set_color("#333")

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("blink_viz.png", dpi=150, facecolor=fig.get_facecolor())
print(f"Saved to blink_viz.png")
print(f"  {n_blinks} blink cues detected in labels")
print(f"  {blink_mask.sum()} blink samples, {rest_mask.sum()} rest samples")
plt.show()
