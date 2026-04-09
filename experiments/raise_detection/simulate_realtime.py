"""
Simulate real-time brow-raise detection by sliding a window across recorded data.

Mimics the actual real-time loop: every ~50ms (13 samples at 256Hz), slide the
128-sample window forward, run inference, apply threshold + cooldown, and log
detections. Then compare against ground-truth raise events.

Usage:
    python -m experiments.raise_detection.simulate_realtime [--threshold 0.5] [--cooldown 0.8]
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import CHANNELS, FS
from core.features import preprocess_v2, extract_features_v2

WINDOW = 128  # 0.5s
STEP = 13     # ~50ms at 256Hz (matches real-time loop)
RAISE_LABEL = 3


def find_ground_truth_events(labels, raw, label_id=RAISE_LABEL):
    """Find raise event peaks (same logic as training)."""
    changes = np.diff((labels == label_id).astype(int), prepend=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) > len(ends):
        ends = np.append(ends, len(labels) - 1)

    detect = np.abs(raw[:, 1]) + np.abs(raw[:, 2])  # frontal
    peaks = []
    for s, e in zip(starts, ends):
        peak = s + np.argmax(detect[s:e + 1])
        peaks.append(peak)
    return peaks


def main():
    parser = argparse.ArgumentParser(description="Simulate real-time raise detection")
    parser.add_argument("--model", type=str, default="data/raise_data/raise_model.pkl")
    parser.add_argument("--data", type=str,
                        default="data/gesture_data/gesture_combined_relabeled.csv")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--cooldown", type=float, default=0.8)
    parser.add_argument("--avg-window", type=int, default=5,
                        help="Moving average window for predictions")
    parser.add_argument("--mode", choices=["avg", "peak"], default="peak",
                        help="Detection mode: 'avg' or 'peak' (rising edge)")
    args = parser.parse_args()

    # Load model
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    model_name = bundle["best_name"]
    needs_scaling = model_name != "Random Forest"
    print(f"Model: {model_name}")
    print(f"Threshold: {args.threshold}, Cooldown: {args.cooldown}s")

    # Load data
    df = pd.read_csv(args.data)
    raw = df[CHANNELS].values
    labels = df["label"].values
    n_samples = len(raw)
    duration = n_samples / FS
    print(f"Data: {n_samples} samples ({duration:.1f}s)")

    # Ground truth raise events
    gt_peaks = find_ground_truth_events(labels, raw)
    print(f"Ground truth raises: {len(gt_peaks)}")

    # ─── Simulate sliding window ───
    from collections import deque as deque_
    detections = []      # (sample_idx, confidence)
    all_probs = []       # (sample_idx, raw_prob) for plotting
    all_avgs = []        # (sample_idx, avg_prob) for plotting
    cooldown_samples = int(args.cooldown * FS)
    last_detection = -cooldown_samples - 1
    prob_history = deque_(maxlen=args.avg_window)

    n_windows = (n_samples - WINDOW) // STEP
    print(f"\nSliding {n_windows} windows (step={STEP} samples, ~{STEP/FS*1000:.0f}ms, "
          f"avg_window={args.avg_window})...")

    for i in range(0, n_samples - WINDOW, STEP):
        window = raw[i:i + WINDOW]
        w_proc = preprocess_v2(window)
        feats = np.array(extract_features_v2(w_proc)).reshape(1, -1)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        if needs_scaling:
            feats = scaler.transform(feats)

        probs = model.predict_proba(feats)[0]
        p_raise = float(probs[1])
        prev_avg = float(np.mean(prob_history)) if len(prob_history) > 0 else 0.0
        prob_history.append(p_raise)
        p_avg = float(np.mean(prob_history))
        center = i + WINDOW // 2
        all_probs.append((center, p_raise))
        all_avgs.append((center, p_avg))

        past_cooldown = (center - last_detection) > cooldown_samples
        if args.mode == "peak":
            triggered = (p_avg >= args.threshold and
                         prev_avg < args.threshold and past_cooldown)
        else:
            triggered = (p_avg >= args.threshold and past_cooldown)

        if triggered:
            detections.append((center, p_avg))
            last_detection = center

    print(f"Detections: {len(detections)}")

    # ─── Match detections to ground truth ───
    MATCH_WINDOW = int(1.0 * FS)  # ±1s tolerance for matching
    matched_gt = set()
    true_positives = []
    false_positives = []

    for det_idx, (det_sample, det_conf) in enumerate(detections):
        matched = False
        for gt_i, gt_peak in enumerate(gt_peaks):
            if gt_i not in matched_gt and abs(det_sample - gt_peak) <= MATCH_WINDOW:
                matched_gt.add(gt_i)
                true_positives.append((det_sample, det_conf, gt_peak))
                matched = True
                break
        if not matched:
            false_positives.append((det_sample, det_conf))

    missed = [gt_peaks[i] for i in range(len(gt_peaks)) if i not in matched_gt]

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(missed)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # False positives per minute
    fp_per_min = fp / (duration / 60)

    print(f"\n{'=' * 50}")
    print(f"SIMULATED REAL-TIME RESULTS")
    print(f"{'=' * 50}")
    print(f"  True positives:  {tp}/{len(gt_peaks)} raises detected")
    print(f"  False positives: {fp} ({fp_per_min:.1f}/min)")
    print(f"  Missed:          {fn}")
    print(f"  Precision:       {precision:.3f}")
    print(f"  Recall:          {recall:.3f}")
    print(f"  F1:              {f1:.3f}")

    if true_positives:
        latencies = [abs(d - g) / FS * 1000 for d, _, g in true_positives]
        print(f"\n  Detection latency:")
        print(f"    Mean:   {np.mean(latencies):.0f}ms")
        print(f"    Median: {np.median(latencies):.0f}ms")
        print(f"    Max:    {np.max(latencies):.0f}ms")

    # ─── Detail log ───
    if true_positives:
        print(f"\n  Hits:")
        for det_s, det_c, gt_s in true_positives:
            t = det_s / FS
            lag = (det_s - gt_s) / FS * 1000
            print(f"    t={t:6.1f}s  conf={det_c:.2f}  lag={lag:+.0f}ms")

    if false_positives:
        print(f"\n  False alarms:")
        for det_s, det_c in false_positives:
            t = det_s / FS
            # Check what label was at this point
            nearby_labels = labels[max(0, det_s-64):det_s+64]
            unique = set(nearby_labels) - {0}
            context = f"  (near: {unique})" if unique else ""
            print(f"    t={t:6.1f}s  conf={det_c:.2f}{context}")

    if missed:
        print(f"\n  Missed raises:")
        for gt_s in missed:
            t = gt_s / FS
            print(f"    t={t:6.1f}s")

    # ─── Plot ───
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    fig.suptitle(f"Simulated Real-Time Raise Detection "
                 f"(thresh={args.threshold}, cooldown={args.cooldown}s)",
                 fontsize=13, fontweight="bold")

    time_axis = np.arange(n_samples) / FS

    # Panel 1: Raw EEG (frontal channels)
    ax = axes[0]
    ax.plot(time_axis, raw[:, 1], alpha=0.6, linewidth=0.5, label="AF7")
    ax.plot(time_axis, raw[:, 2], alpha=0.6, linewidth=0.5, label="AF8")
    for gt_s in gt_peaks:
        ax.axvline(gt_s / FS, color="lime", alpha=0.5, linewidth=1.5, linestyle="--")
    for det_s, det_c in detections:
        color = "#00ff00" if any(det_s == d for d, _, _ in true_positives) else "red"
        ax.axvline(det_s / FS, color=color, alpha=0.7, linewidth=1.5)
    ax.set_ylabel("µV")
    ax.set_title("Frontal EEG (AF7/AF8)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: Raise probability over time (raw + moving average)
    ax = axes[1]
    prob_times = [s / FS for s, _ in all_probs]
    prob_vals = [p for _, p in all_probs]
    avg_times = [s / FS for s, _ in all_avgs]
    avg_vals = [p for _, p in all_avgs]
    ax.plot(prob_times, prob_vals, color="#ff9800", linewidth=0.5, alpha=0.3, label="Raw")
    ax.plot(avg_times, avg_vals, color="#4CAF50", linewidth=1.2, alpha=0.9,
            label=f"Avg (n={args.avg_window})")
    ax.axhline(args.threshold, color="red", linestyle="--", alpha=0.7,
               label=f"Threshold={args.threshold}")

    for det_s, det_c in detections:
        color = "#00ff00" if any(det_s == d for d, _, _ in true_positives) else "red"
        ax.plot(det_s / FS, det_c, "v", color=color, markersize=8)

    ax.set_ylabel("P(raise)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Model Confidence (raw + moving average)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: Ground truth labels
    ax = axes[2]
    label_colors = {0: "#333333", 1: "#2196F3", 2: "#9C27B0", 3: "#4CAF50", 4: "#FF5722"}
    label_names = {0: "rest", 1: "blink", 2: "furrow", 3: "raise", 4: "clench"}
    for lid in [1, 2, 3, 4]:
        mask = labels == lid
        if mask.any():
            ax.fill_between(time_axis, 0, 1, where=mask,
                            color=label_colors[lid], alpha=0.6, label=label_names[lid])
    ax.set_ylabel("Label")
    ax.set_xlabel("Time (s)")
    ax.set_title("Ground Truth")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.set_yticks([])

    plt.tight_layout()
    out_path = "data/raise_data/simulated_realtime.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")

    # ─── Threshold sweep ───
    print(f"\n{'=' * 50}")
    print("THRESHOLD SWEEP")
    print(f"{'=' * 50}")
    print(f"{'Thresh':>8s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'FP/min':>7s}")
    print("-" * 52)

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        dets = []
        last_det = -cooldown_samples - 1
        prev_a = 0.0
        for center, p_a in all_avgs:
            past_cd = (center - last_det) > cooldown_samples
            if args.mode == "peak":
                hit = (p_a >= thresh and prev_a < thresh and past_cd)
            else:
                hit = (p_a >= thresh and past_cd)
            if hit:
                dets.append((center, p_a))
                last_det = center
            prev_a = p_a

        m_gt = set()
        t_tp, t_fp = 0, 0
        for ds, dc in dets:
            hit = False
            for gi, gp in enumerate(gt_peaks):
                if gi not in m_gt and abs(ds - gp) <= MATCH_WINDOW:
                    m_gt.add(gi)
                    hit = True
                    break
            if hit:
                t_tp += 1
            else:
                t_fp += 1
        t_fn = len(gt_peaks) - t_tp
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0
        t_fpm = t_fp / (duration / 60)
        print(f"{thresh:8.1f} {t_tp:4d} {t_fp:4d} {t_fn:4d} {t_prec:7.3f} {t_rec:7.3f} {t_f1:7.3f} {t_fpm:7.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
