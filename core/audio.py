import threading
import numpy as np

AUDIO_SR = 44100

# Lazy-load sounddevice to avoid COM STA initialization before bleak/WinRT
_sd = None

def _get_sd():
    global _sd
    if _sd is None:
        import sounddevice
        _sd = sounddevice
    return _sd


def make_tone(freq, duration, volume=0.5):
    """Generate a sine wave tone as a numpy array."""
    t = np.linspace(0, duration, int(AUDIO_SR * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * volume
    fade = int(AUDIO_SR * 0.01)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)


def play(sound):
    """Play a sound non-blocking."""
    sd = _get_sd()
    threading.Thread(target=lambda: sd.play(sound, AUDIO_SR), daemon=True).start()


# Pre-generated sounds for blink detection
BLINK_BEEP = make_tone(880, 0.15, 0.6)
REST_TONE = make_tone(440, 0.08, 0.3)
COUNTDOWN_BEEP = make_tone(660, 0.08, 0.35)
DONE_SOUND = np.concatenate([
    make_tone(523, 0.12, 0.4),
    make_tone(659, 0.12, 0.4),
    make_tone(784, 0.2, 0.5),
])

# Gesture-specific cue sounds (distinct frequencies)
GESTURE_CUES = {
    1: make_tone(880, 0.15, 0.6),   # blink  — high beep
    2: make_tone(600, 0.20, 0.6),   # furrow — mid-low tone
    3: make_tone(1200, 0.20, 0.6),  # raise  — high tone
    4: make_tone(350, 0.25, 0.6),   # clench — low tone
}

# Gesture detection feedback sounds
GESTURE_DETECT_SOUNDS = {
    1: np.concatenate([make_tone(1047, 0.08, 0.5), make_tone(1319, 0.08, 0.5)]),  # blink
    2: np.concatenate([make_tone(600, 0.08, 0.5), make_tone(500, 0.08, 0.5)]),    # furrow
    3: np.concatenate([make_tone(1200, 0.08, 0.5), make_tone(1400, 0.08, 0.5)]),  # raise
    4: np.concatenate([make_tone(350, 0.08, 0.5), make_tone(250, 0.08, 0.5)]),    # clench
}
