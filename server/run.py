"""
Main entry point: BLE streaming + gesture detection + WebSocket + optional OSC.

Usage:
    python -m server.run [--model data/gesture_data/gesture_model.pkl] [--osc]
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import ctypes
    ctypes.windll.ole32.CoInitializeEx(0, 0x0)  # COINIT_MULTITHREADED

import argparse
import json
import time

from bleak import BleakClient

from core import MUSE_ADDRESS, CONTROL_UUID, EEG_UUIDS, CHANNELS, GESTURE_LABELS
from server.config import (WS_HOST, WS_PORT, OSC_ENABLED, OSC_HOST, OSC_PORT,
                           DETECT_INTERVAL, STATUS_INTERVAL)
from server.event_server import EventBus, GestureDetector

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    raise


async def ws_handler(websocket, event_bus):
    """Handle a single WebSocket client connection."""
    q = event_bus.subscribe()
    remote = websocket.remote_address
    print(f"[WS] Client connected: {remote}")

    # Send welcome
    welcome = {
        "type": "welcome",
        "gestures": {str(k): v for k, v in GESTURE_LABELS.items()},
        "timestamp": time.time(),
    }
    await websocket.send(json.dumps(welcome))

    try:
        while True:
            event = await q.get()
            await websocket.send(json.dumps(event))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        event_bus.unsubscribe(q)
        print(f"[WS] Client disconnected: {remote}")


async def detection_loop(detector, event_bus):
    """Poll the detector and publish events."""
    gesture_counts = {gid: 0 for gid in range(1, 5)}

    while True:
        await asyncio.sleep(DETECT_INTERVAL)
        event = detector.detect()
        if event:
            gid = event["gesture_id"]
            gesture_counts[gid] += 1
            event["count"] = gesture_counts[gid]
            event_bus.publish(event)
            total = sum(gesture_counts.values())
            print(f"  >>> {event['gesture'].upper()} #{gesture_counts[gid]} "
                  f"(conf={event['confidence']:.2f}) [total={total}]")


async def status_loop(detector, event_bus):
    """Broadcast periodic status messages."""
    while True:
        await asyncio.sleep(STATUS_INTERVAL)
        event_bus.publish({
            "type": "status",
            "connected": detector.connected,
            "timestamp": time.time(),
        })


async def osc_sender(event_bus, host, port):
    """Forward gesture events via OSC."""
    try:
        from pythonosc.udp_client import SimpleUDPClient
    except ImportError:
        print("[OSC] python-osc not installed. Run: pip install python-osc")
        return

    client = SimpleUDPClient(host, port)
    q = event_bus.subscribe()
    print(f"[OSC] Sending to {host}:{port}")

    try:
        while True:
            event = await q.get()
            if event["type"] == "gesture":
                client.send_message("/muse/gesture", [
                    event["gesture"],
                    event["confidence"],
                    event["timestamp"],
                ])
                client.send_message(f"/muse/gesture/{event['gesture']}", [
                    event["confidence"],
                ])
    finally:
        event_bus.unsubscribe(q)


async def main():
    parser = argparse.ArgumentParser(description="Muse 2 Gesture Event Server")
    parser.add_argument("--model", type=str, default="data/gesture_data/gesture_model.pkl")
    parser.add_argument("--osc", action="store_true", help="Enable OSC output")
    parser.add_argument("--ws-port", type=int, default=WS_PORT)
    parser.add_argument("--osc-port", type=int, default=OSC_PORT)
    args = parser.parse_args()

    # Initialize
    detector = GestureDetector(args.model)
    event_bus = EventBus()

    print(f"Model: {detector.model_name} ({detector.num_gestures} classes)")
    print(f"WebSocket: ws://{WS_HOST}:{args.ws_port}")
    if args.osc:
        print(f"OSC: {OSC_HOST}:{args.osc_port}")

    # Connect to Muse
    print(f"\nConnecting to Muse at {MUSE_ADDRESS}...")
    async with BleakClient(MUSE_ADDRESS) as client:
        print(f"Connected: {client.is_connected}")
        detector.connected = True

        # Start BLE streaming
        for ch, uuid in EEG_UUIDS.items():
            await client.start_notify(uuid, detector.make_handler(ch))
        await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x64, 0x0a]))

        print("Calibrating... keep still for 3 seconds...")
        await asyncio.sleep(3)

        print("\n" + "=" * 55)
        print("GESTURE EVENT SERVER RUNNING")
        print(f"WebSocket: ws://localhost:{args.ws_port}")
        print("Press Ctrl+C to stop")
        print("=" * 55 + "\n")

        # Start WebSocket server
        ws_server = await websockets.serve(
            lambda ws: ws_handler(ws, event_bus),
            WS_HOST, args.ws_port,
        )

        # Build task list
        tasks = [
            asyncio.create_task(detection_loop(detector, event_bus)),
            asyncio.create_task(status_loop(detector, event_bus)),
        ]
        if args.osc:
            tasks.append(asyncio.create_task(osc_sender(event_bus, OSC_HOST, args.osc_port)))

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            pass
        finally:
            detector.connected = False
            await client.write_gatt_char(CONTROL_UUID, bytes([0x02, 0x68, 0x0a]))
            ws_server.close()
            await ws_server.wait_closed()
            print("\nServer stopped.")


asyncio.run(main())
