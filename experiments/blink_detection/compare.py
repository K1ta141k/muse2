import asyncio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
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

    # Trim all channels to same length
    min_len = min(len(buffers[ch]) for ch in CHANNELS)
    return {ch: np.array(buffers[ch][:min_len]) for ch in CHANNELS}


async def main():
    print(f"Connecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}\n")

        input("Put Muse ON your head, press Enter ")
        on_data = await record(client, RECORD_SECONDS)
        print("On-head recording done.\n")

        input("Take Muse OFF, press Enter ")
        off_data = await record(client, RECORD_SECONDS)
        print("Off-head recording done.\n")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Muse EEG: On-Head vs Off-Head", fontsize=14, fontweight="bold")

    colors = {"TP9": "#e6194b", "AF7": "#3cb44b", "AF8": "#4363d8", "TP10": "#f58231"}
    n_show = 1000

    # Top-left: on-head waveform
    ax = axes[0, 0]
    ax.set_title("On-Head — Raw EEG (µV)")
    t = np.arange(n_show) / FS
    for ch in CHANNELS:
        ax.plot(t, on_data[ch][:n_show], color=colors[ch], linewidth=0.6, label=ch)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("µV")
    ax.legend(loc="upper right", fontsize=8)

    # Top-right: off-head waveform
    ax = axes[0, 1]
    ax.set_title("Off-Head — Raw EEG (µV)")
    for ch in CHANNELS:
        ax.plot(t, off_data[ch][:n_show], color=colors[ch], linewidth=0.6, label=ch)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("µV")
    ax.legend(loc="upper right", fontsize=8)

    # Bottom-left: on-head PSD
    ax = axes[1, 0]
    ax.set_title("On-Head — Power Spectral Density")
    for ch in CHANNELS:
        f, pxx = welch(on_data[ch], fs=FS, nperseg=512)
        mask = f <= 60
        ax.semilogy(f[mask], pxx[mask], color=colors[ch], linewidth=1, label=ch)
    ax.axvspan(8, 12, alpha=0.2, color="green", label="Alpha (8–12 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("µV²/Hz")
    ax.set_xlim(0, 60)
    ax.legend(loc="upper right", fontsize=8)

    # Bottom-right: off-head PSD
    ax = axes[1, 1]
    ax.set_title("Off-Head — Power Spectral Density")
    for ch in CHANNELS:
        f, pxx = welch(off_data[ch], fs=FS, nperseg=512)
        mask = f <= 60
        ax.semilogy(f[mask], pxx[mask], color=colors[ch], linewidth=1, label=ch)
    ax.axvspan(8, 12, alpha=0.2, color="green", label="Alpha (8–12 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("µV²/Hz")
    ax.set_xlim(0, 60)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("on_vs_off.png", dpi=150)
    print("Saved to on_vs_off.png")
    plt.show()


asyncio.run(main())
