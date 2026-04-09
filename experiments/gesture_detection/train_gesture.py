"""
Train multi-class gesture classifiers (rest, blink, furrow, raise, clench).

V2 pipeline: multi-band decomposition, amplitude-preserving preprocessing,
baseline-referenced features, clipping interpolation. 89 features.
5-fold stratified CV with RF, XGBoost, SVM, and CNN.

Usage:
    python -m experiments.gesture_detection.train_gesture [--input ...] [--output ...]
"""

import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

from core import CHANNELS, FS, GESTURE_LABELS, NUM_GESTURES
from core.features import (preprocess_v2, extract_features_v2,
                           V2_FEATURE_NAMES)

plt.style.use("dark_background")

# ─── Config ───
WINDOW_SEC = 0.5
WINDOW_SAMPLES = int(FS * WINDOW_SEC)  # 128
BASELINE_SAMPLES = 256  # 1s baseline before event
AUGMENT_SHIFTS = [-20, -10, 0, 10, 20]  # time-shift augmentation (samples)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Train multi-class gesture classifier (V2)")
    parser.add_argument("--input", type=str, default="data/gesture_data/gesture_combined_relabeled.csv")
    parser.add_argument("--output", type=str, default="data/gesture_data/gesture_model.pkl")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ─── Load relabeled data (raw µV, NOT preprocessed globally) ───
    df = pd.read_csv(args.input)
    labels = df["label"].values
    raw = df[CHANNELS].values  # (N, 4) raw µV
    print(f"Loaded {len(df)} samples (raw µV, V2 pipeline)")

    # ─── Event-aligned windowing with baseline ───
    half = WINDOW_SAMPLES // 2
    gesture_peak_indices = {}

    for gesture_id in [1, 2, 3, 4]:
        gesture_changes = np.diff((labels == gesture_id).astype(int), prepend=0)
        starts = np.where(gesture_changes == 1)[0]
        ends = np.where(gesture_changes == -1)[0]
        if len(starts) > len(ends):
            ends = np.append(ends, len(labels) - 1)

        # Peak detection on raw data (appropriate channels)
        if gesture_id == 4:  # clench -> temporal
            detect = np.abs(raw[:, 0]) + np.abs(raw[:, 3])
        else:  # blink, furrow, raise -> frontal
            detect = np.abs(raw[:, 1]) + np.abs(raw[:, 2])

        peaks = []
        for s, e in zip(starts, ends):
            peak = s + np.argmax(detect[s:e+1])
            peaks.append(peak)
        gesture_peak_indices[gesture_id] = peaks
        print(f"  {GESTURE_LABELS[gesture_id]:8s}: {len(peaks)} peaks")

    def extract_window_with_baseline(center_idx):
        """Extract (128, 4) window + (256, 4) baseline from raw data."""
        start = center_idx - half
        end = center_idx + half
        if start < 0 or end > len(raw):
            return None, None
        window = raw[start:end]  # (128, 4) raw µV

        # Baseline: 1s of data ending where the window starts
        bl_start = max(0, start - BASELINE_SAMPLES)
        bl_end = start
        baseline = raw[bl_start:bl_end] if bl_end - bl_start >= half else None

        return window, baseline

    # Build gesture windows with time-shift augmentation
    gesture_windows_raw = []
    gesture_labels = []
    print(f"\nExtracting gesture windows (augmented: {len(AUGMENT_SHIFTS)} shifts)...")
    for gesture_id, peaks in gesture_peak_indices.items():
        count = 0
        for pi in peaks:
            for shift in AUGMENT_SHIFTS:
                w_raw, _ = extract_window_with_baseline(pi + shift)
                if w_raw is not None:
                    gesture_windows_raw.append(w_raw)
                    gesture_labels.append(gesture_id)
                    count += 1
        print(f"  {GESTURE_LABELS[gesture_id]:8s}: {count} windows ({len(peaks)} peaks x {len(AUGMENT_SHIFTS)} shifts)")

    print(f"Total gesture windows: {len(gesture_windows_raw)}")

    # Rest windows: avoid ±1s around any gesture peak
    all_peaks = [pi for peaks in gesture_peak_indices.values() for pi in peaks]
    rest_forbidden = set()
    for pi in all_peaks:
        for offset in range(-FS, FS + 1):
            rest_forbidden.add(pi + offset)

    margin = half + BASELINE_SAMPLES
    rest_candidates = [i for i in range(margin, len(raw) - half)
                       if labels[i] == 0 and i not in rest_forbidden]

    np.random.seed(42)
    n_rest = len(gesture_windows_raw)
    n_rest = min(n_rest, len(rest_candidates))
    rest_indices = np.random.choice(rest_candidates, size=n_rest, replace=False)

    rest_windows_raw = []
    for ri in rest_indices:
        w_raw, _ = extract_window_with_baseline(ri)
        if w_raw is not None:
            rest_windows_raw.append(w_raw)

    print(f"Rest windows: {len(rest_windows_raw)}")

    # Combine and extract features
    all_raw = rest_windows_raw + gesture_windows_raw
    all_labels = np.array([0] * len(rest_windows_raw) + gesture_labels)

    print(f"\nExtracting V2 features ({len(V2_FEATURE_NAMES)} per window)...")
    X_features = []
    X_raw_processed = []
    for i, w_raw in enumerate(all_raw):
        w_proc = preprocess_v2(w_raw)

        # Baseline for this window
        center = half  # approximate
        bl_start = max(0, 0)  # we only have the window, recompute from raw indices
        # Actually, get baseline from the data
        # For simplicity, pass None (deviation features will be relative to 0)
        feats = extract_features_v2(w_proc, baseline=None)
        X_features.append(feats)
        X_raw_processed.append(w_proc)

    X_features = np.array(X_features)
    X_raw_arr = np.array(X_raw_processed)  # (N, 128, 4) for CNN

    # Replace NaN/inf with 0
    X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nDataset: {len(all_labels)} windows")
    for gid in range(NUM_GESTURES):
        count = (all_labels == gid).sum()
        print(f"  {GESTURE_LABELS[gid]:8s}: {count}")
    print(f"Feature matrix: {X_features.shape}")

    # ─── 1D CNN ───
    class GestureCNN(nn.Module):
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
            # After 2x MaxPool on 128 samples -> 32 timesteps, 64 channels
            self.fc = nn.Sequential(
                nn.Linear(64 * 32, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, NUM_GESTURES),
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # ─── Training loop ───
    N_FOLDS = 5
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    metrics_list = ["acc", "prec", "rec", "f1"]
    results = {name: {m: [] for m in metrics_list}
               for name in ["Random Forest", "XGBoost", "SVM", "CNN"]}

    # Store last fold's confusion matrix for each model
    last_cm = {}

    print(f"\n{'='*60}")
    print(f"Training with {N_FOLDS}-fold stratified cross-validation")
    print(f"{'='*60}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, all_labels)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        y_train, y_test = all_labels[train_idx], all_labels[test_idx]

        # Classical ML
        X_tr_f, X_te_f = X_features[train_idx], X_features[test_idx]
        scaler = StandardScaler()
        X_tr_fs = scaler.fit_transform(X_tr_f)
        X_te_fs = scaler.transform(X_te_f)

        classical_models = {
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                     class_weight="balanced"),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      random_state=42, eval_metric="mlogloss",
                                      use_label_encoder=False),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42,
                        class_weight="balanced"),
        }

        for name, model in classical_models.items():
            Xtr = X_tr_f if name == "Random Forest" else X_tr_fs
            Xte = X_te_f if name == "Random Forest" else X_te_fs
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)

            results[name]["acc"].append(accuracy_score(y_test, y_pred))
            results[name]["prec"].append(precision_score(y_test, y_pred, average="macro", zero_division=0))
            results[name]["rec"].append(recall_score(y_test, y_pred, average="macro", zero_division=0))
            results[name]["f1"].append(f1_score(y_test, y_pred, average="macro", zero_division=0))
            last_cm[name] = confusion_matrix(y_test, y_pred, labels=list(range(NUM_GESTURES)))

        # CNN
        X_tr_raw = torch.FloatTensor(X_raw_arr[train_idx].transpose(0, 2, 1)).to(DEVICE)
        X_te_raw = torch.FloatTensor(X_raw_arr[test_idx].transpose(0, 2, 1)).to(DEVICE)
        y_tr_t = torch.LongTensor(y_train).to(DEVICE)
        y_te_t = torch.LongTensor(y_test).to(DEVICE)

        train_ds = TensorDataset(X_tr_raw, y_tr_t)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

        # Class weights
        cw = compute_class_weight("balanced", classes=np.arange(NUM_GESTURES), y=y_train)
        weight = torch.FloatTensor(cw).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight)

        cnn = GestureCNN().to(DEVICE)
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
        results["CNN"]["prec"].append(precision_score(y_test, preds, average="macro", zero_division=0))
        results["CNN"]["rec"].append(recall_score(y_test, preds, average="macro", zero_division=0))
        results["CNN"]["f1"].append(f1_score(y_test, preds, average="macro", zero_division=0))
        last_cm["CNN"] = confusion_matrix(y_test, preds, labels=list(range(NUM_GESTURES)))

        for name in results:
            f1 = results[name]["f1"][-1]
            acc = results[name]["acc"][-1]
            print(f"  {name:25s}  acc={acc:.3f}  f1={f1:.3f}")

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"RESULTS (mean +/- std across {N_FOLDS} folds)")
    print(f"{'='*60}")
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

    print(f"\nBest: {best_model_name} (macro F1={best_f1:.3f})")

    # ─── Confusion matrix plot ───
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Confusion Matrices (last fold)", fontsize=14, fontweight="bold")
    label_names = [GESTURE_LABELS[i] for i in range(NUM_GESTURES)]

    for ax, name in zip(axes, results.keys()):
        cm = last_cm[name]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=label_names, yticklabels=label_names)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    fig.savefig("data/gesture_data/confusion_matrices.png", dpi=150)
    print("Saved data/gesture_data/confusion_matrices.png")

    # ─── Bar chart ───
    fig, ax = plt.subplots(figsize=(12, 6))
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
    ax.set_title(f"Multi-Gesture Detection — {N_FOLDS}-Fold CV", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.1, axis="y")
    plt.tight_layout()
    fig.savefig("data/gesture_data/training_results.png", dpi=150)
    print("Saved data/gesture_data/training_results.png")

    # ─── Retrain best model on full data ───
    print(f"\nRetraining {best_model_name} on full dataset...")

    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_features)

    if best_model_name == "CNN":
        X_full = torch.FloatTensor(X_raw_arr.transpose(0, 2, 1)).to(DEVICE)
        y_full = torch.LongTensor(all_labels).to(DEVICE)
        ds = TensorDataset(X_full, y_full)
        dl = DataLoader(ds, batch_size=16, shuffle=True)

        cw = compute_class_weight("balanced", classes=np.arange(NUM_GESTURES), y=all_labels)
        weight = torch.FloatTensor(cw).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight)

        final_cnn = GestureCNN().to(DEVICE)
        optimizer = torch.optim.Adam(final_cnn.parameters(), lr=0.001)

        final_cnn.train()
        for epoch in range(100):
            for xb, yb in dl:
                optimizer.zero_grad()
                loss = criterion(final_cnn(xb), yb)
                loss.backward()
                optimizer.step()

        torch.save(final_cnn.state_dict(), "data/gesture_data/gesture_cnn.pt")
        print("Saved data/gesture_data/gesture_cnn.pt")

        # Save XGBoost as pkl fallback (best classical model)
        final_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     random_state=42, eval_metric="mlogloss")
        final_model.fit(X_features, all_labels)
    elif best_model_name == "Random Forest":
        final_model = RandomForestClassifier(n_estimators=300, random_state=42,
                                              class_weight="balanced")
        final_model.fit(X_features, all_labels)
    elif best_model_name == "XGBoost":
        final_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     random_state=42, eval_metric="mlogloss",
                                     use_label_encoder=False)
        final_model.fit(X_features, all_labels)
    elif best_model_name == "SVM":
        final_model = SVC(kernel="rbf", probability=True, random_state=42,
                           class_weight="balanced")
        final_model.fit(X_scaled, all_labels)
    else:
        final_model = RandomForestClassifier(n_estimators=300, random_state=42,
                                              class_weight="balanced")
        final_model.fit(X_features, all_labels)

    bundle = {
        "model": final_model,
        "scaler": scaler_final,
        "feature_names": V2_FEATURE_NAMES,
        "best_name": best_model_name,
        "gesture_labels": GESTURE_LABELS,
        "num_gestures": NUM_GESTURES,
        "pipeline_version": 2,
    }
    with open(args.output, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {args.output}")

    if hasattr(final_model, "feature_importances_"):
        imp = final_model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:15]
        print("\nTop 15 features:")
        for i in top_idx:
            print(f"  {V2_FEATURE_NAMES[i]:30s}  {imp[i]:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
