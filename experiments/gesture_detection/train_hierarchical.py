"""
Train and compare: single-stage 5-class vs two-stage hierarchical gesture classifier.

Two-stage approach:
  Stage 1 (gate):  Binary — rest (0) vs gesture (1)
  Stage 2 (classify): 4-class — blink/furrow/raise/clench (only on gate=1)

Both approaches evaluated on identical 5-fold stratified CV splits for fair comparison.
Saves the winning model to gesture_model.pkl.

Usage:
    python -m experiments.gesture_detection.train_hierarchical [--input ...] [--output ...]
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from core import CHANNELS, FS, GESTURE_LABELS, NUM_GESTURES
from core.features import preprocess_v2, extract_features_v2, V2_FEATURE_NAMES

plt.style.use("dark_background")

# ─── Config ───
WINDOW_SAMPLES = int(FS * 0.5)  # 128
AUGMENT_SHIFTS = [-20, -10, 0, 10, 20]
N_FOLDS = 5


def load_and_prepare_data(csv_path):
    """Load CSV, extract peak-centered windows with augmentation, return features + labels."""
    df = pd.read_csv(csv_path)
    labels = df["label"].values
    raw = df[CHANNELS].values
    half = WINDOW_SAMPLES // 2
    print(f"Loaded {len(df)} samples")

    # Find gesture peaks
    gesture_peak_indices = {}
    for gid in [1, 2, 3, 4]:
        changes = np.diff((labels == gid).astype(int), prepend=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if len(starts) > len(ends):
            ends = np.append(ends, len(labels) - 1)

        if gid == 4:
            detect = np.abs(raw[:, 0]) + np.abs(raw[:, 3])
        else:
            detect = np.abs(raw[:, 1]) + np.abs(raw[:, 2])

        peaks = []
        for s, e in zip(starts, ends):
            peak = s + np.argmax(detect[s:e + 1])
            peaks.append(peak)
        gesture_peak_indices[gid] = peaks
        print(f"  {GESTURE_LABELS[gid]:8s}: {len(peaks)} peaks")

    def extract_window(center_idx):
        start = center_idx - half
        end = center_idx + half
        if start < 0 or end > len(raw):
            return None
        return raw[start:end]

    # Gesture windows with augmentation
    gesture_windows = []
    gesture_labels = []
    for gid, peaks in gesture_peak_indices.items():
        for pi in peaks:
            for shift in AUGMENT_SHIFTS:
                w = extract_window(pi + shift)
                if w is not None:
                    gesture_windows.append(w)
                    gesture_labels.append(gid)

    # Rest windows
    all_peaks = [pi for peaks in gesture_peak_indices.values() for pi in peaks]
    rest_forbidden = set()
    for pi in all_peaks:
        for offset in range(-FS, FS + 1):
            rest_forbidden.add(pi + offset)

    margin = half + 256
    rest_candidates = [i for i in range(margin, len(raw) - half)
                       if labels[i] == 0 and i not in rest_forbidden]

    np.random.seed(42)
    n_rest = min(len(gesture_windows), len(rest_candidates))
    rest_indices = np.random.choice(rest_candidates, size=n_rest, replace=False)
    rest_windows = [w for ri in rest_indices if (w := extract_window(ri)) is not None]

    # Combine
    all_raw = rest_windows + gesture_windows
    all_labels = np.array([0] * len(rest_windows) + gesture_labels)

    # Extract V2 features
    print(f"\nExtracting V2 features ({len(V2_FEATURE_NAMES)} per window)...")
    X_features = []
    for w_raw in all_raw:
        w_proc = preprocess_v2(w_raw)
        feats = extract_features_v2(w_proc)
        X_features.append(feats)

    X = np.nan_to_num(np.array(X_features), nan=0.0, posinf=0.0, neginf=0.0)
    y = all_labels

    print(f"\nDataset: {len(y)} windows")
    for gid in range(NUM_GESTURES):
        print(f"  {GESTURE_LABELS[gid]:8s}: {(y == gid).sum()}")

    return X, y


def evaluate_single_stage(X, y, train_idx, test_idx):
    """Train single-stage 5-class model, return predictions on test set."""
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "1S-RF": RandomForestClassifier(n_estimators=300, random_state=42,
                                         class_weight="balanced"),
        "1S-XGB": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 random_state=42, eval_metric="mlogloss"),
        "1S-SVM": SVC(kernel="rbf", probability=True, random_state=42,
                       class_weight="balanced"),
    }

    results = {}
    for name, model in models.items():
        Xtr = X_tr if "RF" in name else X_tr_s
        Xte = X_te if "RF" in name else X_te_s
        model.fit(Xtr, y_tr)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)
        results[name] = {
            "pred": y_pred,
            "prob": y_prob,
            "model": model,
            "scaler": scaler,
        }
    return results, y_te


def evaluate_two_stage(X, y, train_idx, test_idx):
    """Train two-stage hierarchical model, return predictions on test set."""
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # ── Stage 1: Binary (rest=0 vs gesture=1) ──
    y_tr_binary = (y_tr > 0).astype(int)
    y_te_binary = (y_te > 0).astype(int)

    scaler1 = StandardScaler()
    X_tr_s1 = scaler1.fit_transform(X_tr)
    X_te_s1 = scaler1.transform(X_te)

    gate_models = {
        "RF": RandomForestClassifier(n_estimators=300, random_state=42,
                                      class_weight="balanced"),
        "XGB": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              random_state=42, eval_metric="logloss"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42,
                    class_weight="balanced"),
    }

    # ── Stage 2: 4-class (blink=1, furrow=2, raise=3, clench=4) ──
    # Only train on gesture samples (label > 0)
    gesture_mask_tr = y_tr > 0
    X_tr_g = X_tr[gesture_mask_tr]
    y_tr_g = y_tr[gesture_mask_tr]  # labels 1-4

    scaler2 = StandardScaler()
    X_tr_s2 = scaler2.fit_transform(X_tr_g)

    classify_models = {
        "RF": RandomForestClassifier(n_estimators=300, random_state=42,
                                      class_weight="balanced"),
        "XGB": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                              random_state=42, eval_metric="mlogloss"),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42,
                    class_weight="balanced"),
    }

    # Try all combinations of gate + classifier
    results = {}
    for g_name, g_model in gate_models.items():
        # Fit gate
        Xtr_g = X_tr if g_name == "RF" else X_tr_s1
        Xte_g = X_te if g_name == "RF" else X_te_s1
        g_model.fit(Xtr_g, y_tr_binary)
        gate_pred = g_model.predict(Xte_g)
        gate_prob = g_model.predict_proba(Xte_g)

        for c_name, c_model in classify_models.items():
            # Fit classifier on gesture-only data
            Xtr_c = X_tr_g if c_name == "RF" else X_tr_s2
            # Remap labels 1-4 → 0-3 for XGBoost compatibility
            y_tr_g_remapped = y_tr_g - 1
            if c_name == "XGB":
                c_model.fit(Xtr_c if c_name == "RF" else scaler2.transform(X_tr_g),
                            y_tr_g_remapped)
            else:
                c_model.fit(Xtr_c if c_name == "RF" else scaler2.transform(X_tr_g),
                            y_tr_g)

            # Two-stage prediction
            y_pred = np.zeros(len(X_te), dtype=int)  # default: rest
            gesture_indices = np.where(gate_pred == 1)[0]

            if len(gesture_indices) > 0:
                X_te_gesture = X_te[gesture_indices]
                Xte_c = X_te_gesture if c_name == "RF" else scaler2.transform(X_te_gesture)
                stage2_pred = c_model.predict(Xte_c)
                if c_name == "XGB":
                    stage2_pred = stage2_pred + 1  # remap back 0-3 → 1-4
                y_pred[gesture_indices] = stage2_pred

            # Build probability matrix (5 classes)
            y_prob = np.zeros((len(X_te), NUM_GESTURES))
            # P(rest) = P(gate=rest)
            y_prob[:, 0] = gate_prob[:, 0]

            if len(gesture_indices) > 0:
                X_te_gesture = X_te[gesture_indices]
                Xte_c = X_te_gesture if c_name == "RF" else scaler2.transform(X_te_gesture)
                stage2_prob = c_model.predict_proba(Xte_c)
                # P(gesture_k) = P(gate=gesture) * P(class=k | gesture)
                for i, idx in enumerate(gesture_indices):
                    p_gesture = gate_prob[idx, 1]
                    if c_name == "XGB":
                        # XGB classes are 0-3, map to columns 1-4
                        for k in range(stage2_prob.shape[1]):
                            y_prob[idx, k + 1] = p_gesture * stage2_prob[i, k]
                    else:
                        # SVM/RF classes are 1-4
                        classes = c_model.classes_
                        for k_idx, k in enumerate(classes):
                            y_prob[idx, k] = p_gesture * stage2_prob[i, k_idx]

            combo_name = f"2S-{g_name}+{c_name}"
            results[combo_name] = {
                "pred": y_pred,
                "prob": y_prob,
                "gate_model": g_model,
                "classify_model": c_model,
                "scaler1": scaler1,
                "scaler2": scaler2,
                "gate_name": g_name,
                "classify_name": c_name,
            }

    return results, y_te


def compute_metrics(y_true, y_pred):
    """Compute standard classification metrics."""
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_gate_metrics(y_true, y_pred):
    """Compute binary gate performance (rest vs any gesture)."""
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    return {
        "gate_acc": accuracy_score(y_true_bin, y_pred_bin),
        "gate_prec": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "gate_rec": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "gate_f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
    }


def simulate_realtime(y_prob, y_true, thresholds, description=""):
    """Simulate real-time detection with confidence thresholds.

    Returns stats about detections, false positives, missed gestures.
    """
    n = len(y_true)
    detections = 0
    correct_detections = 0
    false_positives = 0  # predicted gesture during rest
    missed = 0  # actual gesture but predicted rest or below threshold

    for i in range(n):
        pred = np.argmax(y_prob[i])
        conf = y_prob[i, pred]

        if pred > 0 and conf > thresholds.get(pred, 0.5):
            detections += 1
            if y_true[i] == pred:
                correct_detections += 1
            elif y_true[i] == 0:
                false_positives += 1
        elif y_true[i] > 0:
            missed += 1

    total_gestures = (y_true > 0).sum()
    total_rest = (y_true == 0).sum()

    return {
        "detections": detections,
        "correct": correct_detections,
        "false_pos": false_positives,
        "missed": missed,
        "precision": correct_detections / max(detections, 1),
        "recall": correct_detections / max(total_gestures, 1),
        "fp_rate": false_positives / max(total_rest, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical gesture classifier")
    parser.add_argument("--input", type=str, default="data/gesture_data/gesture_combined_relabeled.csv")
    parser.add_argument("--output", type=str, default="data/gesture_data/gesture_model.pkl")
    args = parser.parse_args()

    X, y = load_and_prepare_data(args.input)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Accumulators
    all_results = {}  # name -> {metric: [values per fold]}
    all_cm = {}       # name -> last fold confusion matrix
    all_gate = {}     # name -> {gate_metric: [values per fold]}
    all_rt = {}       # name -> {rt_metric: [values per fold]}

    # Threshold sets to test
    high_thresh = {1: 0.7, 2: 0.6, 3: 0.7, 4: 0.6}
    low_thresh = {1: 0.4, 2: 0.35, 3: 0.4, 4: 0.35}

    print(f"\n{'=' * 70}")
    print(f"SINGLE-STAGE vs TWO-STAGE — {N_FOLDS}-fold stratified CV")
    print(f"{'=' * 70}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'-' * 50}")
        print(f"  Fold {fold + 1}/{N_FOLDS}")
        print(f"{'-' * 50}")

        # Single-stage
        ss_results, y_test = evaluate_single_stage(X, y, train_idx, test_idx)
        for name, r in ss_results.items():
            m = compute_metrics(y_test, r["pred"])
            gm = compute_gate_metrics(y_test, r["pred"])
            rt_high = simulate_realtime(r["prob"], y_test, high_thresh)
            rt_low = simulate_realtime(r["prob"], y_test, low_thresh)

            all_results.setdefault(name, {k: [] for k in m})
            all_gate.setdefault(name, {k: [] for k in gm})
            all_rt.setdefault(name + "_hiT", {k: [] for k in rt_high})
            all_rt.setdefault(name + "_loT", {k: [] for k in rt_low})

            for k, v in m.items():
                all_results[name][k].append(v)
            for k, v in gm.items():
                all_gate[name][k].append(v)
            for k, v in rt_high.items():
                all_rt[name + "_hiT"][k].append(v)
            for k, v in rt_low.items():
                all_rt[name + "_loT"][k].append(v)

            all_cm[name] = confusion_matrix(y_test, r["pred"], labels=list(range(NUM_GESTURES)))
            print(f"  {name:25s}  f1={m['f1']:.3f}  gate_f1={gm['gate_f1']:.3f}")

        # Two-stage
        ts_results, y_test = evaluate_two_stage(X, y, train_idx, test_idx)
        for name, r in ts_results.items():
            m = compute_metrics(y_test, r["pred"])
            gm = compute_gate_metrics(y_test, r["pred"])
            rt_high = simulate_realtime(r["prob"], y_test, high_thresh)
            rt_low = simulate_realtime(r["prob"], y_test, low_thresh)

            all_results.setdefault(name, {k: [] for k in m})
            all_gate.setdefault(name, {k: [] for k in gm})
            all_rt.setdefault(name + "_hiT", {k: [] for k in rt_high})
            all_rt.setdefault(name + "_loT", {k: [] for k in rt_low})

            for k, v in m.items():
                all_results[name][k].append(v)
            for k, v in gm.items():
                all_gate[name][k].append(v)
            for k, v in rt_high.items():
                all_rt[name + "_hiT"][k].append(v)
            for k, v in rt_low.items():
                all_rt[name + "_loT"][k].append(v)

            all_cm[name] = confusion_matrix(y_test, r["pred"], labels=list(range(NUM_GESTURES)))

        # Print top two-stage combos for this fold
        ts_names = [n for n in all_results if n.startswith("2S-")]
        ts_sorted = sorted(ts_names, key=lambda n: all_results[n]["f1"][-1], reverse=True)
        for name in ts_sorted[:3]:
            f1 = all_results[name]["f1"][-1]
            gf1 = all_gate[name]["gate_f1"][-1]
            print(f"  {name:25s}  f1={f1:.3f}  gate_f1={gf1:.3f}")

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print(f"CLASSIFICATION RESULTS (mean +/- std across {N_FOLDS} folds)")
    print(f"{'=' * 70}")
    print(f"{'Model':25s} {'Acc':>12s} {'Prec':>12s} {'Rec':>12s} {'F1':>12s}")
    print("-" * 73)

    # Sort by F1
    sorted_names = sorted(all_results.keys(),
                           key=lambda n: np.mean(all_results[n]["f1"]), reverse=True)

    best_name = sorted_names[0]
    best_f1 = np.mean(all_results[best_name]["f1"])

    for name in sorted_names:
        r = all_results[name]
        row = f"{name:25s}"
        for m in ["acc", "prec", "rec", "f1"]:
            row += f" {np.mean(r[m]):.3f}+/-{np.std(r[m]):.3f}"
        marker = " <-- BEST" if name == best_name else ""
        print(row + marker)

    # Gate performance
    print(f"\n{'=' * 70}")
    print(f"GATE PERFORMANCE (rest vs gesture detection)")
    print(f"{'=' * 70}")
    print(f"{'Model':25s} {'Gate Acc':>12s} {'Gate Prec':>12s} {'Gate Rec':>12s} {'Gate F1':>12s}")
    print("-" * 73)

    for name in sorted_names:
        g = all_gate[name]
        row = f"{name:25s}"
        for m in ["gate_acc", "gate_prec", "gate_rec", "gate_f1"]:
            row += f" {np.mean(g[m]):.3f}+/-{np.std(g[m]):.3f}"
        print(row)

    # Simulated real-time performance
    print(f"\n{'=' * 70}")
    print(f"SIMULATED REAL-TIME (with confidence thresholds)")
    print(f"{'=' * 70}")

    for thresh_label, thresh_suffix in [("HIGH thresholds (0.6-0.7)", "_hiT"),
                                         ("LOW thresholds (0.35-0.4)", "_loT")]:
        print(f"\n  {thresh_label}:")
        print(f"  {'Model':25s} {'Det Prec':>10s} {'Det Rec':>10s} {'FP Rate':>10s} {'Detections':>12s}")
        print("  " + "-" * 69)

        for name in sorted_names:
            rt_name = name + thresh_suffix
            if rt_name not in all_rt:
                continue
            rt = all_rt[rt_name]
            row = f"  {name:25s}"
            row += f" {np.mean(rt['precision']):10.3f}"
            row += f" {np.mean(rt['recall']):10.3f}"
            row += f" {np.mean(rt['fp_rate']):10.4f}"
            row += f" {np.mean(rt['detections']):10.1f}"
            print(row)

    # ─── Confusion matrices for top models ───
    top_single = sorted([n for n in sorted_names if n.startswith("1S-")],
                         key=lambda n: np.mean(all_results[n]["f1"]), reverse=True)[0]
    top_two = sorted([n for n in sorted_names if n.startswith("2S-")],
                      key=lambda n: np.mean(all_results[n]["f1"]), reverse=True)[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Confusion Matrices: Best Single-Stage vs Best Two-Stage (last fold)",
                 fontsize=13, fontweight="bold")
    label_names = [GESTURE_LABELS[i] for i in range(NUM_GESTURES)]

    for ax, name in zip(axes, [top_single, top_two]):
        cm = all_cm[name]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=label_names, yticklabels=label_names)
        f1 = np.mean(all_results[name]["f1"])
        ax.set_title(f"{name} (F1={f1:.3f})", fontsize=11)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    fig.savefig("data/gesture_data/hierarchical_comparison.png", dpi=150)
    print(f"\nSaved data/gesture_data/hierarchical_comparison.png")

    # ─── Save best model ───
    print(f"\n{'=' * 70}")
    print(f"WINNER: {best_name} (macro F1={best_f1:.3f})")
    print(f"{'=' * 70}")

    # Retrain on full data
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)

    if best_name.startswith("2S-"):
        # Two-stage: retrain both models on full data
        g_name = best_name.split("-")[1].split("+")[0]
        c_name = best_name.split("+")[1]
        print(f"\nRetraining two-stage: gate={g_name}, classify={c_name}")

        # Gate model (binary)
        y_binary = (y > 0).astype(int)
        scaler1 = StandardScaler()
        X_s1 = scaler1.fit_transform(X)

        if g_name == "RF":
            gate = RandomForestClassifier(n_estimators=300, random_state=42,
                                           class_weight="balanced")
            gate.fit(X, y_binary)
        elif g_name == "XGB":
            gate = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  random_state=42, eval_metric="logloss")
            gate.fit(X_s1, y_binary)
        else:
            gate = SVC(kernel="rbf", probability=True, random_state=42,
                        class_weight="balanced")
            gate.fit(X_s1, y_binary)

        # Classify model (4-class, gesture only)
        gesture_mask = y > 0
        X_g = X[gesture_mask]
        y_g = y[gesture_mask]
        scaler2 = StandardScaler()
        X_s2 = scaler2.fit_transform(X_g)

        if c_name == "RF":
            classify = RandomForestClassifier(n_estimators=300, random_state=42,
                                               class_weight="balanced")
            classify.fit(X_g, y_g)
        elif c_name == "XGB":
            classify = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      random_state=42, eval_metric="mlogloss")
            classify.fit(X_s2, y_g - 1)  # remap 1-4 → 0-3
        else:
            classify = SVC(kernel="rbf", probability=True, random_state=42,
                            class_weight="balanced")
            classify.fit(X_s2, y_g)

        bundle = {
            "architecture": "two_stage",
            "gate_model": gate,
            "classify_model": classify,
            "gate_name": g_name,
            "classify_name": c_name,
            "scaler1": scaler1,
            "scaler2": scaler2,
            "feature_names": V2_FEATURE_NAMES,
            "best_name": best_name,
            "gesture_labels": GESTURE_LABELS,
            "num_gestures": NUM_GESTURES,
            "pipeline_version": 2,
        }

        # Print gate feature importances
        if hasattr(gate, "feature_importances_"):
            imp = gate.feature_importances_
            top_idx = np.argsort(imp)[::-1][:10]
            print("\nGate top 10 features:")
            for i in top_idx:
                print(f"  {V2_FEATURE_NAMES[i]:30s}  {imp[i]:.4f}")

        if hasattr(classify, "feature_importances_"):
            imp = classify.feature_importances_
            top_idx = np.argsort(imp)[::-1][:10]
            print("\nClassify top 10 features:")
            for i in top_idx:
                print(f"  {V2_FEATURE_NAMES[i]:30s}  {imp[i]:.4f}")

    else:
        # Single-stage
        print(f"\nRetraining single-stage: {best_name}")
        model_type = best_name.split("-")[1]

        if model_type == "RF":
            model = RandomForestClassifier(n_estimators=300, random_state=42,
                                            class_weight="balanced")
            model.fit(X, y)
        elif model_type == "XGB":
            model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   random_state=42, eval_metric="mlogloss")
            model.fit(X_scaled, y)
        else:
            model = SVC(kernel="rbf", probability=True, random_state=42,
                         class_weight="balanced")
            model.fit(X_scaled, y)

        bundle = {
            "architecture": "single_stage",
            "model": model,
            "scaler": scaler_full,
            "feature_names": V2_FEATURE_NAMES,
            "best_name": best_name,
            "gesture_labels": GESTURE_LABELS,
            "num_gestures": NUM_GESTURES,
            "pipeline_version": 2,
        }

    with open(args.output, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
