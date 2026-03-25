import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import welch, iirnotch, filtfilt, detrend
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# ─── Config ───
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FS = 256
WINDOW_SEC = 0.5
WINDOW_SAMPLES = int(FS * WINDOW_SEC)  # 128
PRE_PEAK = 0.15
POST_PEAK = 0.35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ─── Load relabeled data ───
df = pd.read_csv("blink_data_relabeled.csv")
t = df["timestamp"].values
labels = df["label"].values

# ─── Preprocessing: notch 60Hz + detrend (borrowing from neurothink) ───
def preprocess(df):
    b60, a60 = iirnotch(60.0, 200.0, FS)
    for ch in CHANNELS:
        sig = df[ch].values.copy()
        sig = filtfilt(b60, a60, sig)
        sig = detrend(sig)
        # z-score
        mu, std = sig.mean(), sig.std()
        if std > 0:
            sig = (sig - mu) / std
        df[ch] = sig
    return df

df = preprocess(df)
print(f"Loaded {len(df)} samples, preprocessed (notch + detrend + z-score)")

# ─── Event-aligned windowing ───
# Find blink peaks (label=1 regions)
blink_changes = np.diff(labels, prepend=0)
blink_starts_idx = np.where(blink_changes == 1)[0]
blink_ends_idx = np.where(blink_changes == -1)[0]
if len(blink_starts_idx) > len(blink_ends_idx):
    blink_ends_idx = np.append(blink_ends_idx, len(labels) - 1)

# For each blink region, find peak in frontal channels
frontal = np.abs(df["AF7"].values) + np.abs(df["AF8"].values)
blink_peak_indices = []
for s, e in zip(blink_starts_idx, blink_ends_idx):
    peak = s + np.argmax(frontal[s:e+1])
    blink_peak_indices.append(peak)

print(f"Found {len(blink_peak_indices)} blink peaks")

def extract_window(center_idx):
    """Extract a window centered on center_idx."""
    half = WINDOW_SAMPLES // 2
    start = center_idx - half
    end = center_idx + half
    if start < 0 or end > len(df):
        return None
    window = df.iloc[start:end][CHANNELS].values  # (128, 4)
    return window

# Blink windows: centered on each peak
blink_windows = []
for pi in blink_peak_indices:
    w = extract_window(pi)
    if w is not None:
        blink_windows.append(w)

print(f"Blink windows: {len(blink_windows)}")

# Rest windows: sample from non-blink regions, avoiding ±1s around any peak
rest_forbidden = set()
for pi in blink_peak_indices:
    for offset in range(-FS, FS + 1):
        rest_forbidden.add(pi + offset)

rest_candidates = [i for i in range(WINDOW_SAMPLES // 2, len(df) - WINDOW_SAMPLES // 2)
                   if i not in rest_forbidden]

# Sample 3x as many rest windows as blink (for variety), then balance later
np.random.seed(42)
n_rest = min(len(blink_windows) * 3, len(rest_candidates))
rest_indices = np.random.choice(rest_candidates, size=n_rest, replace=False)

rest_windows = []
for ri in rest_indices:
    w = extract_window(ri)
    if w is not None:
        rest_windows.append(w)

print(f"Rest windows: {len(rest_windows)}")

# ─── Feature engineering (for classical ML) ───
def band_power(signal, fs, fmin, fmax):
    nperseg = min(len(signal), 128)
    if nperseg < 4:
        return 0.0
    f, pxx = welch(signal, fs=fs, nperseg=nperseg)
    mask = (f >= fmin) & (f <= fmax)
    return np.trapezoid(pxx[mask], f[mask]) if mask.any() else 0.0

def extract_features(window):
    """window shape: (128, 4)"""
    feats = []
    for i, ch in enumerate(CHANNELS):
        sig = window[:, i]
        feats.extend([
            np.mean(np.abs(sig)),
            np.std(sig),
            np.ptp(sig),
            band_power(sig, FS, 0.5, 5.0),
            band_power(sig, FS, 8.0, 12.0),
        ])
    return feats

FEATURE_NAMES = []
for ch in CHANNELS:
    for f in ["mean_abs", "std", "ptp", "power_0.5_5", "power_8_12"]:
        FEATURE_NAMES.append(f"{ch}_{f}")

# Build feature matrix
all_windows = blink_windows + rest_windows
all_labels = np.array([1] * len(blink_windows) + [0] * len(rest_windows))

X_features = np.array([extract_features(w) for w in all_windows])
X_raw = np.array(all_windows)  # (N, 128, 4) for CNN

print(f"\nDataset: {len(all_labels)} windows ({(all_labels==1).sum()} blink, {(all_labels==0).sum()} rest)")
print(f"Feature matrix: {X_features.shape}")
print(f"Raw matrix: {X_raw.shape}")

# ─── 1D CNN (neurothink style) ───
class BlinkCNN(nn.Module):
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
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x: (batch, 4, 128)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ─── Training loop ───
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

results = {name: {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []}
           for name in ["Random Forest", "SVM", "Logistic Regression", "CNN"]}

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
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    }

    for name, model in classical_models.items():
        Xtr = X_tr_f if name == "Random Forest" else X_tr_fs
        Xte = X_te_f if name == "Random Forest" else X_te_fs
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        results[name]["acc"].append(accuracy_score(y_test, y_pred))
        results[name]["prec"].append(precision_score(y_test, y_pred, zero_division=0))
        results[name]["rec"].append(recall_score(y_test, y_pred, zero_division=0))
        results[name]["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        results[name]["auc"].append(roc_auc_score(y_test, y_prob))

    # CNN
    X_tr_raw = torch.FloatTensor(X_raw[train_idx].transpose(0, 2, 1)).to(DEVICE)  # (N, 4, 128)
    X_te_raw = torch.FloatTensor(X_raw[test_idx].transpose(0, 2, 1)).to(DEVICE)
    y_tr_t = torch.LongTensor(y_train).to(DEVICE)
    y_te_t = torch.LongTensor(y_test).to(DEVICE)

    train_ds = TensorDataset(X_tr_raw, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    cnn = BlinkCNN().to(DEVICE)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    # Class weights for imbalance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Train
    cnn.train()
    for epoch in range(80):
        epoch_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = cnn(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Eval
    cnn.eval()
    with torch.no_grad():
        logits = cnn(X_te_raw)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    results["CNN"]["acc"].append(accuracy_score(y_test, preds))
    results["CNN"]["prec"].append(precision_score(y_test, preds, zero_division=0))
    results["CNN"]["rec"].append(recall_score(y_test, preds, zero_division=0))
    results["CNN"]["f1"].append(f1_score(y_test, preds, zero_division=0))
    results["CNN"]["auc"].append(roc_auc_score(y_test, probs))

    for name in results:
        f1 = results[name]["f1"][-1]
        acc = results[name]["acc"][-1]
        print(f"  {name:25s}  acc={acc:.3f}  f1={f1:.3f}")

# ─── Summary ───
print(f"\n{'='*60}")
print(f"RESULTS (mean ± std across {N_FOLDS} folds)")
print(f"{'='*60}")
print(f"{'Model':25s} {'Acc':>10s} {'Prec':>10s} {'Rec':>10s} {'F1':>10s} {'AUC':>10s}")
print("-" * 75)

best_model_name = None
best_f1 = 0

for name in results:
    metrics = results[name]
    row = f"{name:25s}"
    for m in ["acc", "prec", "rec", "f1", "auc"]:
        vals = metrics[m]
        row += f" {np.mean(vals):.3f}±{np.std(vals):.3f}"
    print(row)
    mean_f1 = np.mean(metrics["f1"])
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_model_name = name

print(f"\nBest: {best_model_name} (F1={best_f1:.3f})")

# ─── Plot results ───
fig, ax = plt.subplots(figsize=(12, 6))
model_names = list(results.keys())
metrics_to_plot = ["acc", "prec", "rec", "f1", "auc"]
x = np.arange(len(model_names))
width = 0.15
colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]

for i, metric in enumerate(metrics_to_plot):
    means = [np.mean(results[m][metric]) for m in model_names]
    stds = [np.std(results[m][metric]) for m in model_names]
    ax.bar(x + i * width, means, width, yerr=stds, label=metric.upper(),
           color=colors[i], alpha=0.85, capsize=3)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(f"Blink Detection — {N_FOLDS}-Fold CV (event-aligned windows, preprocessed)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.1, axis="y")
plt.tight_layout()
fig.savefig("training_results.png", dpi=150)
print("Saved training_results.png")

# ─── Retrain best model on full data and save ───
print(f"\nRetraining {best_model_name} on full dataset...")

if best_model_name == "CNN":
    X_full = torch.FloatTensor(X_raw.transpose(0, 2, 1)).to(DEVICE)
    y_full = torch.LongTensor(all_labels).to(DEVICE)
    ds = TensorDataset(X_full, y_full)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    final_cnn = BlinkCNN().to(DEVICE)
    optimizer = torch.optim.Adam(final_cnn.parameters(), lr=0.001)
    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()
    weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    final_cnn.train()
    for epoch in range(100):
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(final_cnn(xb), yb)
            loss.backward()
            optimizer.step()

    torch.save(final_cnn.state_dict(), "blink_cnn.pt")
    print("Saved blink_cnn.pt")
else:
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_features)

    if best_model_name == "Random Forest":
        final_model = RandomForestClassifier(n_estimators=200, random_state=42)
        final_model.fit(X_features, all_labels)
    elif best_model_name == "SVM":
        final_model = SVC(kernel="rbf", probability=True, random_state=42)
        final_model.fit(X_scaled, all_labels)
    else:
        final_model = LogisticRegression(max_iter=2000, random_state=42)
        final_model.fit(X_scaled, all_labels)

    bundle = {
        "model": final_model,
        "scaler": scaler_final,
        "feature_names": FEATURE_NAMES,
        "best_name": best_model_name,
    }
    with open("blink_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("Saved blink_model.pkl")

    if best_model_name == "Random Forest":
        imp = final_model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:5]
        print("\nTop 5 features:")
        for i in top_idx:
            print(f"  {FEATURE_NAMES[i]:30s}  {imp[i]:.4f}")

print("\nDone.")
