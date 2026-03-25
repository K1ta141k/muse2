import asyncio
import time
import csv
import threading
import numpy as np
import sounddevice as sd
from bleak import BleakClient

MUSE_ADDRESS = "00:55:DA:B8:35:23"

CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
EEG_UUIDS = {
    "TP9":  "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7":  "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8":  "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FS = 256
TOTAL_SECONDS = 60
REST_DURATION = 3.0
BLINK_WINDOW = 0.5


AUDIO_SR = 44100


def make_tone(freq, duration, volume=0.5):
    """Generate a sine wave tone as a numpy array."""
    t = np.linspace(0, duration, int(AUDIO_SR * duration), endpoint=False)
    # Apply a short fade in/out to avoid clicks
    tone = np.sin(2 * np.pi * freq * t) * volume
    fade = int(AUDIO_SR * 0.01)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    return tone.astype(np.float32)


# Pre-generate sounds
BLINK_BEEP = make_tone(880, 0.15, 0.6)       # short high beep — blink now
REST_TONE = make_tone(440, 0.08, 0.3)         # soft low blip — rest started
DONE_SOUND = np.concatenate([                  # cheerful done jingle
    make_tone(523, 0.12, 0.4),
    make_tone(659, 0.12, 0.4),
    make_tone(784, 0.2, 0.5),
])
COUNTDOWN_BEEP = make_tone(660, 0.08, 0.35)   # countdown tick


def play(sound):
    """Play a sound non-blocking."""
    threading.Thread(target=lambda: sd.play(sound, AUDIO_SR), daemon=True).start()


def raw_to_uv(val):
    if val > 32767:
        return (val - 65536) * 0.48828125
    return val * 0.48828125


def parse_packet(data):
    samples = []
    for i in range(2, len(data), 2):
        val = int.from_bytes(data[i:i+2], byteorder='big')
        samples.append(raw_to_uv(val))
    return samples


async def main():
    # Shared state
    raw_data = []  # list of (timestamp, channel, uv_value)
    lock = asyncio.Lock()
    t0 = None

    def make_handler(ch):
        def callback(sender, data):
            now = time.monotonic()
            samples = parse_packet(data)
            # Approximate per-sample timestamps (12 samples at 256 Hz)
            for j, uv in enumerate(samples):
                sample_t = now - (len(samples) - 1 - j) / FS
                raw_data.append((sample_t, ch, uv))
        return callback

    print(f"Connecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}\n")

        input("Put Muse on your head, press Enter to start... ")
        print()
        print("=" * 50)
        print(f"Recording {TOTAL_SECONDS}s — REST then BLINK on cue")
        print("=" * 50)

        # Countdown
        for i in [3, 2, 1]:
            play(COUNTDOWN_BEEP)
            print(f"  Starting in {i}...")
            await asyncio.sleep(1)
        print()

        # Start notifications
        for ch, uuid in EEG_UUIDS.items():
            await client.start_notify(uuid, make_handler(ch))
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

        t0 = time.monotonic()
        blink_cue_times = []  # monotonic times when blink cue fired
        elapsed = 0.0

        while elapsed < TOTAL_SECONDS:
            # REST phase
            rest_end = elapsed + REST_DURATION
            if rest_end > TOTAL_SECONDS:
                rest_end = TOTAL_SECONDS
            remaining = rest_end - elapsed
            play(REST_TONE)
            print(f"[{elapsed:5.1f}s] REST — stay still...")
            await asyncio.sleep(remaining)
            elapsed = time.monotonic() - t0

            if elapsed >= TOTAL_SECONDS:
                break

            # BLINK cue
            cue_time = time.monotonic()
            blink_cue_times.append(cue_time)
            play(BLINK_BEEP)
            print(f"[{elapsed:5.1f}s] >>> BLINK NOW! <<<")
            await asyncio.sleep(BLINK_WINDOW)
            elapsed = time.monotonic() - t0

        # Stop streaming
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
        for uuid in EEG_UUIDS.values():
            await client.stop_notify(uuid)

    play(DONE_SOUND)
    print(f"\nRecording complete. Processing {len(raw_data)} raw samples...")

    # --- Assemble into aligned rows ---
    # Group by approximate timestamp: bucket into sample indices
    # Sort all samples by timestamp
    raw_data.sort(key=lambda x: x[0])

    # Build per-channel arrays aligned by time
    # Group samples that are within 1/(2*FS) of each other
    rows = []
    ch_buffers = {ch: [] for ch in CHANNELS}

    for ts, ch, uv in raw_data:
        ch_buffers[ch].append((ts, uv))

    # Use the channel with the most samples as the time reference
    ref_ch = max(CHANNELS, key=lambda c: len(ch_buffers[c]))
    ref_times = [t for t, _ in ch_buffers[ref_ch]]

    # For each channel, interpolate to ref_times
    ch_arrays = {}
    for ch in CHANNELS:
        if len(ch_buffers[ch]) < 2:
            ch_arrays[ch] = np.zeros(len(ref_times))
            continue
        times = np.array([t for t, _ in ch_buffers[ch]])
        vals = np.array([v for _, v in ch_buffers[ch]])
        ch_arrays[ch] = np.interp(ref_times, times, vals)

    # Label each sample
    ref_times = np.array(ref_times)
    labels = np.zeros(len(ref_times), dtype=int)
    for cue_t in blink_cue_times:
        mask = (ref_times >= cue_t) & (ref_times < cue_t + BLINK_WINDOW)
        labels[mask] = 1

    # Timestamps relative to start
    rel_times = ref_times - ref_times[0]

    # --- Write CSV ---
    csv_path = "blink_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "TP9", "AF7", "AF8", "TP10", "label"])
        for i in range(len(ref_times)):
            writer.writerow([
                f"{rel_times[i]:.6f}",
                f"{ch_arrays['TP9'][i]:.4f}",
                f"{ch_arrays['AF7'][i]:.4f}",
                f"{ch_arrays['AF8'][i]:.4f}",
                f"{ch_arrays['TP10'][i]:.4f}",
                labels[i],
            ])

    # --- Summary ---
    n_blink = int(np.sum(labels == 1))
    n_rest = int(np.sum(labels == 0))
    n_cues = len(blink_cue_times)

    print(f"\nSaved to {csv_path}")
    print(f"Total samples:   {len(ref_times)}")
    print(f"Blink cues:      {n_cues}")
    print(f"Blink samples:   {n_blink} (label=1, {BLINK_WINDOW}s window each)")
    print(f"Rest samples:    {n_rest} (label=0)")
    print(f"Duration:        {rel_times[-1]:.1f}s")


asyncio.run(main())
