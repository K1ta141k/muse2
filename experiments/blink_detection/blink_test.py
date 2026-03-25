import asyncio
import numpy as np
import matplotlib.pyplot as plt
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
RECORD_SECONDS = 10


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


async def record(client, duration):
    buffers = {ch: [] for ch in CHANNELS}

    def make_handler(ch):
        def callback(sender, data):
            buffers[ch].extend(parse_packet(data))
        return callback

    for ch, uuid in EEG_UUIDS.items():
        await client.start_notify(uuid, make_handler(ch))

    await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))
    print(f"Recording for {duration} seconds...")
    await asyncio.sleep(duration)
    await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))

    for uuid in EEG_UUIDS.values():
        await client.stop_notify(uuid)

    min_len = min(len(buffers[ch]) for ch in CHANNELS)
    return {ch: np.array(buffers[ch][:min_len]) for ch in CHANNELS}


async def main():
    print(f"Connecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}\n")

        print("=" * 50)
        print("BLINK TEST — Signal Validation")
        print("=" * 50)
        print()
        print("Put the Muse on your head. Make sure the")
        print("forehead sensors touch skin (push hair aside).")
        print()
        print("When recording starts, BLINK HARD 5 times,")
        print("spaced about 2 seconds apart.")
        print()
        input("Press Enter to start recording... ")

        data = await record(client, RECORD_SECONDS)
        print("Recording done.\n")

    # --- Analysis ---
    n_samples = len(data[CHANNELS[0]])
    t = np.arange(n_samples) / FS

    # Focus on AF7 and AF8 (forehead — where blinks show up strongest)
    frontal_avg = (data["AF7"] + data["AF8"]) / 2

    # Detect spikes: abs deviation from rolling median
    window = FS  # 1-second window
    abs_signal = np.abs(frontal_avg)
    baseline = np.median(abs_signal)
    threshold = baseline * 4
    spike_mask = abs_signal > threshold

    # Group nearby spike samples into discrete blink events
    spike_indices = np.where(spike_mask)[0]
    blink_times = []
    if len(spike_indices) > 0:
        groups = np.split(spike_indices, np.where(np.diff(spike_indices) > FS // 4)[0] + 1)
        blink_times = [t[g[len(g) // 2]] for g in groups]

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Blink Test — Are You Getting Real EEG?", fontsize=14, fontweight="bold")

    colors = {"TP9": "#e6194b", "AF7": "#3cb44b", "AF8": "#4363d8", "TP10": "#f58231"}

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        ax.plot(t, data[ch], color=colors[ch], linewidth=0.5)
        ax.set_ylabel(f"{ch}\n(µV)", fontsize=10)
        ax.grid(True, alpha=0.2)

        # Mark detected blinks
        for bt in blink_times:
            ax.axvline(bt, color="red", alpha=0.4, linewidth=1.5, linestyle="--")

    axes[-1].set_xlabel("Time (s)", fontsize=11)

    # Verdict
    n_blinks = len(blink_times)
    if n_blinks >= 3:
        verdict = f"PASS — {n_blinks} blink artifacts detected. Signal is real."
        color = "green"
    elif n_blinks >= 1:
        verdict = f"WEAK — only {n_blinks} blink(s) detected. Check sensor contact."
        color = "orange"
    else:
        verdict = "FAIL — no blink artifacts found. Bad contact or no signal."
        color = "red"

    fig.text(0.5, 0.01, verdict, ha="center", fontsize=13, fontweight="bold", color=color)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    fig.savefig("blink_test.png", dpi=150)
    print(f"\n{verdict}")
    print("Saved to blink_test.png")
    plt.show()


asyncio.run(main())
