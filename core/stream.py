import asyncio
from bleak import BleakClient

MUSE_ADDRESS = "00:55:DA:B8:35:23"

# Muse EEG channel UUIDs
EEG_TP9  = "273e0003-4c4d-454d-96be-f03bac821358"
EEG_AF7  = "273e0004-4c4d-454d-96be-f03bac821358"
EEG_AF8  = "273e0005-4c4d-454d-96be-f03bac821358"
EEG_TP10 = "273e0006-4c4d-454d-96be-f03bac821358"

samples = []

def eeg_handler(channel_name):
    def callback(sender, data):
        # Muse sends 12 samples per packet, 2 bytes each, big endian
        raw = []
        for i in range(2, len(data), 2):
            val = int.from_bytes(data[i:i+2], byteorder='big')
            raw.append(val)
        print(f"{channel_name}: {raw[:6]}")
        samples.append((channel_name, raw))
    return callback

async def main():
    print(f"Connecting to Muse-3523...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")

        # Start notifications on EEG channels
        await client.start_notify(EEG_TP9,  eeg_handler("TP9"))
        await client.start_notify(EEG_AF7,  eeg_handler("AF7"))
        await client.start_notify(EEG_AF8,  eeg_handler("AF8"))
        await client.start_notify(EEG_TP10, eeg_handler("TP10"))

        # Need to send 'd' to start streaming
        CONTROL_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

        print("Streaming EEG for 5 seconds...")
        await asyncio.sleep(5)

        # Stop
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))

    print(f"\nTotal packets received: {len(samples)}")

asyncio.run(main())
