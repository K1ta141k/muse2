# WebSocket server
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# OSC output (optional)
OSC_ENABLED = False
OSC_HOST = "127.0.0.1"
OSC_PORT = 9000

# Per-gesture detection thresholds and cooldowns
GESTURE_THRESHOLDS = {1: 0.6, 2: 0.5, 3: 0.5, 4: 0.5}
GESTURE_COOLDOWNS = {1: 0.6, 2: 0.8, 3: 0.8, 4: 0.8}

# Detection polling interval (seconds)
DETECT_INTERVAL = 0.05  # 50ms = 20Hz

# Status broadcast interval (seconds)
STATUS_INTERVAL = 2.0
