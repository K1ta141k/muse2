# Event Server: WebSocket/OSC Gesture Streaming

**Date:** April 7-8, 2026
**Status:** Built, not yet tested with real Muse

## Architecture

```
Muse 2 (BLE) → Ring Buffers → GestureDetector → EventBus → WebSocket clients
                                                          → OSC clients (optional)
```

## Components

### EventBus (`server/event_server.py`)
Async pub/sub using `asyncio.Queue` per subscriber. Non-blocking — if a subscriber is slow, events are dropped (not queued up).

### GestureDetector (`server/event_server.py`)
Wraps the gesture model with ring buffers and prediction logic:
- Receives raw BLE packets via `make_handler(ch)` callbacks
- Maintains per-channel ring buffers (256 samples deep)
- Every 50ms, extracts a 128-sample window, runs V2 preprocessing + feature extraction, classifies
- Applies per-gesture confidence thresholds and cooldowns to avoid false positives

### WebSocket Server (`server/run.py`)
- `websockets` library on port 8765
- Sends JSON events to all connected clients
- Welcome message on connect: `{"type": "welcome", "gestures": {...}}`
- Periodic status: `{"type": "status", "connected": true}`

### OSC Output (optional)
- `python-osc` UDP client
- Address: `/muse/gesture` with `[name, confidence, timestamp]`
- Per-gesture address: `/muse/gesture/blink` with `[confidence]`

## Event Format

```json
{
  "type": "gesture",
  "gesture": "blink",
  "gesture_id": 1,
  "confidence": 0.87,
  "count": 3,
  "timestamp": 1712345678.123,
  "probabilities": {
    "rest": 0.05,
    "blink": 0.87,
    "furrow": 0.03,
    "raise": 0.02,
    "clench": 0.03
  }
}
```

## Configuration (`server/config.py`)

```python
WS_HOST = "0.0.0.0"
WS_PORT = 8765
OSC_HOST = "127.0.0.1"
OSC_PORT = 9000
DETECT_INTERVAL = 0.05   # 50ms polling
STATUS_INTERVAL = 5.0    # status broadcast every 5s

GESTURE_THRESHOLDS = {1: 0.8, 2: 0.7, 3: 0.8, 4: 0.7}
GESTURE_COOLDOWNS  = {1: 0.6, 2: 0.8, 3: 0.8, 4: 0.8}
```

## Usage

```bash
# Basic (WebSocket only)
python -m server.run

# With OSC
python -m server.run --osc

# Custom ports
python -m server.run --ws-port 9000 --osc-port 9001

# Custom model
python -m server.run --model path/to/model.pkl
```

## Connecting from a Browser

```javascript
const ws = new WebSocket("ws://localhost:8765");
ws.onmessage = (e) => {
  const event = JSON.parse(e.data);
  if (event.type === "gesture") {
    console.log(`${event.gesture} (${event.confidence})`);
  }
};
```

## Windows COM Threading Note

The server uses `CoInitializeEx(0, 0x0)` (MTA) before any imports to avoid the Windows COM STA/MTA conflict between sounddevice (PortAudio → STA) and bleak (WinRT → MTA). The `WindowsSelectorEventLoopPolicy` is set for the asyncio loop. See `server/run.py` header.
