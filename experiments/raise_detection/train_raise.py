"""
Train binary brow-raise detector (rest vs raise).

V3 approach: train on sliding-window-style data to match inference conditions.
  - Positives: peak-centered raise windows (augmented)
  - Negatives: randomly sampled windows from entire recordings (matching how
    the real-time detector sees data) + other gesture windows as hard negatives

Data sources:
  - Raise events: dedicated raise-only sessions + gesture data (label=3)
  - Negatives: random windows from all datasets + other gestures

Usage:
    python -m experiments.raise_detection.train_raise [--output ...]
"""

import argparse
import glob
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from core import CHANNELS, FS
from core.features import preprocess_v2, extract_features_v2, V2_FEATURE_NAMES

plt.style.use("dark_background")

WINDOW_SEC = 0.5
WINDOW_SAMPLES = int(FS * WINDOW_SEC)  # 128
AUGMENT_SHIFTS = [-20, -10, 0, 10, 20]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "rest", 1: "raise"}


def find_peaks(labels, raw, label_id):
    """Find event peaks for a given label using amplitude peak detection."""
    changes = np.diff((labels == label_id).astype(int), prepend=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) > len(ends):
        ends = np.append(ends, len(labels) - 1)
    if label_id == 4:  # clench -> temporal
        detect = np.abs(raw[:, 0]) + np.abs(raw[:, 3])
    else:  # blink, furrow, raise -> frontal
        detect = np.abs(raw[:, 1]) + np.abs(raw[:, 2])
    peaks = []
    for s, e in zip(starts, ends):
        peak = s + np.argmax(detect[s:e + 1])
        peaks.append(peak)
    return peaks


def extract_window(raw, center_idx):
    half = WINDOW_SAMPLES // 2
    start = center_idx - half
    end = center_idx + half
    if start < 0 or end > len(raw):
        return None
    return raw[start:end]


def extract_peak_windows(raw, peaks, augment=True):
    """Extract windows centered on peaks, optionally with time-shift augmentation."""
    shifts = AUGMENT_SHIFTS if augment else [0]
    windows = []
    for pi in peaks:
        for shift in shifts:
            w = extract_window(raw, pi + shift)
            if w is not None:
                windows.append(w)
    return windows


def get_random_negative_windows(raw, labels, raise_label, n_windows, rng):
    """Sample random windows avoiding raise events — mimics sliding-window inference.

    Unlike the old approach, this samples from ANY position (including near
    other gestures, noisy regions, etc.) as long as the center isn't a raise.
    This matches what the real-time detector actually sees.
    """
    half = WINDOW_SAMPLES // 2

    # Only exclude ±0.5s around raise events (tight exclusion)
    raise_forbidden = set()
    raise_indices = np.where(labels == raise_label)[0]
    for idx in raise_indices:
        for offset in range(-WINDOW_SAMPLES, WINDOW_SAMPLES + 1):
            raise_forbidden.add(idx + offset)

    candidates = [i for i in range(half, len(raw) - half)
                  if i not in raise_forbidden]

    if len(candidates) == 0:
        return []

    chosen = rng.choice(candidates, size=min(n_windows, len(candidates)), replace=False)
    windows = []
    for idx in chosen:
        w = extract_window(raw, idx)
        if w is not None:
            windows.append(w)
    return windows


def main():
    parser = argparse.ArgumentParser(description="Train binary raise detector (V2)")
    parser.add_argument("--gesture-data", type=str,
                        default="data/gesture_data/gesture_combined_relabeled.csv")
    parser.add_argument("--blink-data", type=str,
                        default="data/blink_data/blink_data_relabeled.csv")
    parser.add_argument("--raise-dir", type=str, default="data/raise_data")
    parser.add_argument("--output", type=str, default="data/raise_data/raise_model.pkl")
    parser.add_argument("--neg-ratio", type=int, default=4,
                        help="Ratio of negative to positive windows")
    args = parser.parse_args()

    rng = np.random.RandomState(42)
    print(f"Device: {DEVICE}")

    # ═══ RAISE EVENTS (positives) ═══

    raise_windows = []

    # Source 1: Dedicated raise-only sessions
    raise_csvs = sorted(glob.glob(f"{args.raise_dir}/raise_*.csv"))
    raise_datasets = []  # store for negative sampling later
    print(f"\n--- Raise-only sessions ({len(raise_csvs)} files) ---")
    for csv_path in raise_csvs:
        df = pd.read_csv(csv_path)
        raw = df[CHANNELS].values
        labels = df["label"].values
        peaks = find_peaks(labels, raw, label_id=1)  # raise=1 in these CSVs
        windows = extract_peak_windows(raw, peaks)
        raise_windows.extend(windows)
        raise_datasets.append((raw, labels, 1))  # (data, labels, raise_label_id)
        print(f"  {csv_path}: {len(peaks)} peaks -> {len(windows)} windows")

    # Source 2: Raise events from gesture data (label=3)
    gdf = pd.read_csv(args.gesture_data)
    raw_gesture = gdf[CHANNELS].values
    labels_gesture = gdf["label"].values
    gesture_raise_peaks = find_peaks(labels_gesture, raw_gesture, label_id=3)
    gesture_raise_windows = extract_peak_windows(raw_gesture, gesture_raise_peaks)
    raise_windows.extend(gesture_raise_windows)
    print(f"\n--- Gesture data raises ---")
    print(f"  {len(gesture_raise_peaks)} peaks -> {len(gesture_raise_windows)} windows")

    n_raise = len(raise_windows)
    print(f"\nTotal raise windows: {n_raise}")

    # ═══ NEGATIVE WINDOWS ═══
    # Key change: sample RANDOM windows (not just clean rest) to match inference

    n_neg_target = n_raise * args.neg_ratio
    neg_windows = []

    # Source 1: Random windows from raise-only sessions
    # These have long natural rest periods — best source
    print(f"\n--- Negative windows (target: {n_neg_target}) ---")
    for raw, labels, rlabel in raise_datasets:
        n = int(n_neg_target * 0.4 / max(len(raise_datasets), 1))
        windows = get_random_negative_windows(raw, labels, rlabel, n, rng)
        neg_windows.extend(windows)
        print(f"  raise session: {len(windows)} random windows")

    # Source 2: Random windows from gesture data (includes blinks, furrows, clenches as noise)
    n_gesture = int(n_neg_target * 0.4)
    gesture_neg = get_random_negative_windows(
        raw_gesture, labels_gesture, 3, n_gesture, rng)
    neg_windows.extend(gesture_neg)
    print(f"  gesture data: {len(gesture_neg)} random windows")

    # Source 3: Other gestures as explicit hard negatives (augmented)
    other_gesture_windows = []
    for gid, gname in [(1, "blink"), (2, "furrow"), (4, "clench")]:
        peaks = find_peaks(labels_gesture, raw_gesture, gid)
        windows = extract_peak_windows(raw_gesture, peaks)
        other_gesture_windows.extend(windows)
    neg_windows.extend(other_gesture_windows)
    print(f"  other gestures (hard neg): {len(other_gesture_windows)} windows")

    # Source 4: Random windows from blink data
    bdf = pd.read_csv(args.blink_data)
    raw_blink = bdf[CHANNELS].values
    labels_blink = bdf["label"].values
    n_blink = int(n_neg_target * 0.1)
    blink_neg = get_random_negative_windows(raw_blink, labels_blink, -1, n_blink, rng)
    neg_windows.extend(blink_neg)
    print(f"  blink data: {len(blink_neg)} random windows")

    print(f"Total negative windows: {len(neg_windows)}")

    # ═══ COMBINE ═══
    all_raw_windows = neg_windows + raise_windows
    all_labels = np.array([0] * len(neg_windows) + [1] * len(raise_windows))

    n_neg = sum(all_labels == 0)
    n_pos = sum(all_labels == 1)
    print(f"\nDataset: {len(all_labels)} windows")
    print(f"  Not-raise (label 0): {n_neg}")
    print(f"  Raise     (label 1): {n_pos}")
    print(f"  Ratio: {n_neg / max(n_pos, 1):.1f}:1")

    # ═══ EXTRACT FEATURES ═══
    print(f"\nExtracting V2 features ({len(V2_FEATURE_NAMES)} per window)...")
    X_features = []
    X_raw_processed = []
    for w_raw in all_raw_windows:
        w_proc = preprocess_v2(w_raw)
        feats = extract_features_v2(w_proc, baseline=None)
        X_features.append(feats)
        X_raw_processed.append(w_proc)

    X_features = np.array(X_features)
    X_raw_arr = np.array(X_raw_processed)
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Feature matrix: {X_features.shape}")

    # ═══ 1D CNN ═══
    class RaiseCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(4, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(64 * 32, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # ═══ 5-FOLD CV ═══
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    metrics_list = ["acc", "prec", "rec", "f1"]
    results = {name: {m: [] for m in metrics_list}
               for name in ["Random Forest", "XGBoost", "SVM", "CNN"]}
    last_cm = {}

    print(f"\n{'=' * 60}")
    print(f"Training with {N_FOLDS}-fold stratified cross-validation")
    print(f"{'=' * 60}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, all_labels)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
        y_train, y_test = all_labels[train_idx], all_labels[test_idx]

        X_tr_f, X_te_f = X_features[train_idx], X_features[test_idx]
        scaler = StandardScaler()
        X_tr_fs = scaler.fit_transform(X_tr_f)
        X_te_fs = scaler.transform(X_te_f)

        classical_models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced"),
            "XGBoost": XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric="logloss"),
            "SVM": SVC(
                kernel="rbf", probability=True, random_state=42,
                class_weight="balanced"),
        }

        for name, model in classical_models.items():
            Xtr = X_tr_f if name == "Random Forest" else X_tr_fs
            Xte = X_te_f if name == "Random Forest" else X_te_fs
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)

            results[name]["acc"].append(accuracy_score(y_test, y_pred))
            results[name]["prec"].append(precision_score(y_test, y_pred, zero_division=0))
            results[name]["rec"].append(recall_score(y_test, y_pred, zero_division=0))
            results[name]["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            last_cm[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])

        # CNN
        X_tr_raw = torch.FloatTensor(X_raw_arr[train_idx].transpose(0, 2, 1)).to(DEVICE)
        X_te_raw = torch.FloatTensor(X_raw_arr[test_idx].transpose(0, 2, 1)).to(DEVICE)
        y_tr_t = torch.LongTensor(y_train).to(DEVICE)

        train_ds = TensorDataset(X_tr_raw, y_tr_t)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        weight = torch.FloatTensor(cw).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight)

        cnn = RaiseCNN().to(DEVICE)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

        cnn.train()
        for epoch in range(80):
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = criterion(cnn(xb), yb)
                loss.backward()
                optimizer.step()

        cnn.eval()
        with torch.no_grad():
            logits = cnn(X_te_raw)
            preds = logits.argmax(dim=1).cpu().numpy()

        results["CNN"]["acc"].append(accuracy_score(y_test, preds))
        results["CNN"]["prec"].append(precision_score(y_test, preds, zero_division=0))
        results["CNN"]["rec"].append(recall_score(y_test, preds, zero_division=0))
        results["CNN"]["f1"].append(f1_score(y_test, preds, zero_division=0))
        last_cm["CNN"] = confusion_matrix(y_test, preds, labels=[0, 1])

        for name in results:
            f = results[name]["f1"][-1]
            a = results[name]["acc"][-1]
            print(f"  {name:25s}  acc={a:.3f}  f1={f:.3f}")

    # ═══ SUMMARY ═══
    print(f"\n{'=' * 60}")
    print(f"RESULTS (mean +/- std across {N_FOLDS} folds)")
    print(f"{'=' * 60}")
    print(f"{'Model':25s} {'Acc':>12s} {'Prec':>12s} {'Rec':>12s} {'F1':>12s}")
    print("-" * 73)

    best_model_name = None
    best_f1 = 0

    for name in results:
        row = f"{name:25s}"
        for m in metrics_list:
            vals = results[name][m]
            row += f" {np.mean(vals):.3f}+/-{np.std(vals):.3f}"
        print(row)
        mean_f1 = np.mean(results[name]["f1"])
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_model_name = name

    print(f"\nBest: {best_model_name} (F1={best_f1:.3f})")

    # ═══ PLOTS ═══
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Brow Raise Detection — Confusion Matrices (last fold)",
                 fontsize=13, fontweight="bold")
    label_names = ["rest", "raise"]
    for ax, name in zip(axes, results.keys()):
        cm = last_cm[name]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=label_names, yticklabels=label_names)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig("data/raise_data/confusion_matrices.png", dpi=150)
    print("Saved data/raise_data/confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
    for i, metric in enumerate(metrics_list):
        means = [np.mean(results[m][metric]) for m in model_names]
        stds = [np.std(results[m][metric]) for m in model_names]
        ax.bar(x + i * width, means, width, yerr=stds, label=metric.upper(),
               color=colors[i], alpha=0.85, capsize=3)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(f"Brow Raise Detection — {N_FOLDS}-Fold CV", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.1, axis="y")
    plt.tight_layout()
    fig.savefig("data/raise_data/training_results.png", dpi=150)
    print("Saved data/raise_data/training_results.png")

    # ═══ RETRAIN BEST ON FULL DATA ═══
    print(f"\nRetraining {best_model_name} on full dataset...")

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_features)

    if best_model_name == "CNN":
        X_full = torch.FloatTensor(X_raw_arr.transpose(0, 2, 1)).to(DEVICE)
        y_full = torch.LongTensor(all_labels).to(DEVICE)
        ds = TensorDataset(X_full, y_full)
        dl = DataLoader(ds, batch_size=32, shuffle=True)

        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=all_labels)
        weight = torch.FloatTensor(cw).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight)

        final_cnn = RaiseCNN().to(DEVICE)
        optimizer = torch.optim.Adam(final_cnn.parameters(), lr=0.001)
        final_cnn.train()
        for epoch in range(100):
            for xb, yb in dl:
                optimizer.zero_grad()
                loss = criterion(final_cnn(xb), yb)
                loss.backward()
                optimizer.step()

        torch.save(final_cnn.state_dict(), "data/raise_data/raise_cnn.pt")
        print("Saved data/raise_data/raise_cnn.pt")

    # Always save best classical model as pkl (for easy inference)
    # If CNN won, save the best classical as fallback
    classical_rank = sorted(
        [(n, np.mean(results[n]["f1"])) for n in ["Random Forest", "XGBoost", "SVM"]],
        key=lambda x: x[1], reverse=True)
    best_classical = classical_rank[0][0]
    save_model_name = best_model_name if best_model_name != "CNN" else best_classical

    if save_model_name == "Random Forest":
        final_model = RandomForestClassifier(n_estimators=300, random_state=42,
                                             class_weight="balanced")
        final_model.fit(X_features, all_labels)
    elif save_model_name == "XGBoost":
        final_model = XGBClassifier(n_estimators=300, max_depth=6,
                                    learning_rate=0.1, random_state=42,
                                    eval_metric="logloss")
        final_model.fit(X_features, all_labels)
    elif save_model_name == "SVM":
        final_model = SVC(kernel="rbf", probability=True, random_state=42,
                          class_weight="balanced")
        final_model.fit(X_scaled, all_labels)

    bundle = {
        "model": final_model,
        "scaler": scaler_final,
        "feature_names": V2_FEATURE_NAMES,
        "best_name": save_model_name,
        "labels": LABELS,
        "num_classes": 2,
        "pipeline_version": 2,
    }
    with open(args.output, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {args.output} (model: {save_model_name})")

    if hasattr(final_model, "feature_importances_"):
        imp = final_model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:15]
        print("\nTop 15 features:")
        for i in top_idx:
            print(f"  {V2_FEATURE_NAMES[i]:30s}  {imp[i]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
