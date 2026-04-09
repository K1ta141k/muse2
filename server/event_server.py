"""
EventBus + GestureDetector — core components for the gesture event server.

Supports both single-stage and two-stage (hierarchical) model architectures.
"""

import asyncio
import time
import pickle
import threading
import numpy as np
from collections import deque

from core import CHANNELS, FS, GESTURE_LABELS
from core.features import preprocess_v2, extract_features_v2, raw_to_uv, parse_packet
from server.config import GESTURE_THRESHOLDS, GESTURE_COOLDOWNS

WINDOW = 128  # 0.5s at 256Hz


class EventBus:
    """Async pub/sub: publish events to all subscriber queues."""

    def __init__(self):
        self._subscribers = set()

    def subscribe(self):
        q = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q):
        self._subscribers.discard(q)

    def publish(self, event: dict):
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


class GestureDetector:
    """Wraps model loading, ring buffers, and gesture prediction.

    Handles both single-stage and two-stage model bundles automatically.
    """

    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.arch = bundle.get("architecture", "single_stage")
        self.model_name = bundle["best_name"]
        self.num_gestures = bundle["num_gestures"]

        if self.arch == "two_stage":
            self.gate = bundle["gate_model"]
            self.classify = bundle["classify_model"]
            self.scaler1 = bundle["scaler1"]
            self.scaler2 = bundle["scaler2"]
            self.gate_name = bundle["gate_name"]
            self.classify_name = bundle["classify_name"]
        else:
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]

        self.buffers = {ch: deque(maxlen=WINDOW * 2) for ch in CHANNELS}
        self.lock = threading.Lock()
        self.last_gesture_time = {gid: 0.0 for gid in range(1, 5)}
        self.connected = False

    def make_handler(self, ch):
        def callback(sender, data):
            samples = parse_packet(data)
            with self.lock:
                self.buffers[ch].extend(samples)
        return callback

    def _predict(self, feats):
        """Return (probs_5, pred, conf) for either architecture."""
        if self.arch == "two_stage":
            return self._predict_two_stage(feats)
        return self._predict_single(feats)

    def _predict_single(self, feats):
        if "RF" in self.model_name:
            f = feats
        else:
            f = self.scaler.transform(feats)
        probs = self.model.predict_proba(f)[0]
        pred = int(np.argmax(probs))
        return probs, pred, float(probs[pred])

    def _predict_two_stage(self, feats):
        # Gate
        feats_g = feats if self.gate_name == "RF" else self.scaler1.transform(feats)
        gate_prob = self.gate.predict_proba(feats_g)[0]
        p_rest, p_gesture = gate_prob[0], gate_prob[1]

        probs = np.zeros(5)
        probs[0] = p_rest

        if p_gesture > 0.4:
            feats_c = feats if self.classify_name == "RF" else self.scaler2.transform(feats)
            s2_prob = self.classify.predict_proba(feats_c)[0]
            if self.classify_name == "XGB":
                for k in range(len(s2_prob)):
                    probs[k + 1] = p_gesture * s2_prob[k]
            else:
                for k_idx, k in enumerate(self.classify.classes_):
                    probs[int(k)] = p_gesture * s2_prob[k_idx]

        pred = int(np.argmax(probs))
        return probs, pred, float(probs[pred])

    def detect(self):
        """Check for gesture. Returns event dict or None."""
        with self.lock:
            if any(len(self.buffers[ch]) < WINDOW for ch in CHANNELS):
                return None
            window = np.column_stack([list(self.buffers[ch])[-WINDOW:] for ch in CHANNELS])

        window = preprocess_v2(window)
        feats = np.array(extract_features_v2(window)).reshape(1, -1)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        probs, pred, conf = self._predict(feats)

        if pred == 0:
            return None

        threshold = GESTURE_THRESHOLDS.get(pred, 0.5)
        cooldown = GESTURE_COOLDOWNS.get(pred, 0.8)
        now = time.monotonic()

        if conf > threshold and (now - self.last_gesture_time[pred]) > cooldown:
            self.last_gesture_time[pred] = now
            return {
                "type": "gesture",
                "gesture": GESTURE_LABELS[pred],
                "gesture_id": pred,
                "confidence": round(conf, 4),
                "timestamp": time.time(),
                "probabilities": {GESTURE_LABELS[i]: round(float(probs[i]), 4)
                                  for i in range(self.num_gestures)},
            }
        return None
