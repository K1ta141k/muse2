import asyncio
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from bleak import BleakClient

MUSE_ADDRESS = "00:55:DA:B8:35:23"

# Muse EEG channel UUIDs
EEG_UUIDS = {
    "TP9":  "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7":  "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8":  "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}
CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"

SAMPLE_RATE = 256  # Muse sample rate in Hz
DISPLAY_SECONDS = 4
BUFFER_LEN = SAMPLE_RATE * DISPLAY_SECONDS

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]

# Ring buffers for each channel
buffers = {ch: deque([0.0] * BUFFER_LEN, maxlen=BUFFER_LEN) for ch in CHANNELS}
lock = threading.Lock()


def eeg_handler(channel_name):
    def callback(sender, data):
        samples = []
        for i in range(2, len(data), 2):
            val = int.from_bytes(data[i:i+2], byteorder='big')
            samples.append(float(val))
        with lock:
            buffers[channel_name].extend(samples)
    return callback


async def stream_eeg():
    print(f"Connecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        for ch_name, uuid in EEG_UUIDS.items():
            await client.start_notify(uuid, eeg_handler(ch_name))

        # Send 'd' command to start streaming
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))
        print("Streaming EEG... Close the plot window to stop.")

        # Keep streaming until interrupted
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
            print("Stopped streaming.")


def start_ble_thread():
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(stream_eeg())

    t = threading.Thread(target=run, daemon=True)
    t.start()


def main():
    start_ble_thread()

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Muse EEG — Live Stream", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e")

    x = np.arange(BUFFER_LEN) / SAMPLE_RATE

    lines = []
    for i, (ax, ch, color) in enumerate(zip(axes, CHANNELS, COLORS)):
        ax.set_facecolor("#16213e")
        ax.set_ylabel(ch, color=color, fontsize=12, fontweight="bold")
        ax.tick_params(colors="#aaa")
        ax.set_xlim(0, DISPLAY_SECONDS)
        ax.set_ylim(0, 1200)
        ax.grid(True, alpha=0.15, color="#fff")
        for spine in ax.spines.values():
            spine.set_color("#333")
        line, = ax.plot(x, [0] * BUFFER_LEN, color=color, linewidth=0.7)
        lines.append(line)

    axes[-1].set_xlabel("Time (s)", color="#aaa", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame):
        with lock:
            for i, ch in enumerate(CHANNELS):
                data = list(buffers[ch])
                lines[i].set_ydata(data)
                # Auto-scale y-axis based on recent data
                recent = data[-SAMPLE_RATE:]  # last 1 second
                if max(recent) > 0:
                    mn, mx = min(recent), max(recent)
                    margin = max((mx - mn) * 0.2, 20)
                    axes[i].set_ylim(mn - margin, mx + margin)
        return lines

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
