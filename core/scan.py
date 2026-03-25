# save as bletest.py
import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=10)
    for d in devices:
        print(f"  {d.address} - {d.name}")
        if d.name and "Muse" in d.name:
            print(f"  ^^^ FOUND YOUR MUSE ^^^")

asyncio.run(main())
