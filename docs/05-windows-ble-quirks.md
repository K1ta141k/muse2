# Windows BLE Quirks & Fixes

Hard-won lessons from getting Muse 2 BLE working on Windows 11 ARM.

## COM Threading: STA vs MTA

**Problem:** bleak uses WinRT (requires COM MTA), but sounddevice's PortAudio initializes COM as STA on import. Once COM is initialized as STA in a thread, it can't be changed to MTA. bleak then fails with:

```
Thread is configured for Windows GUI but callbacks are not working
```

**Fix:** Force MTA before ANY imports that touch COM:

```python
import ctypes
ctypes.windll.ole32.CoInitializeEx(0, 0x0)  # COINIT_MULTITHREADED
```

This must be at the very top of the script, before importing sounddevice, bleak, or anything that transitively imports them.

**Additionally:** Lazy-load sounddevice in `core/audio.py`:

```python
_sd = None
def _get_sd():
    global _sd
    if _sd is None:
        import sounddevice
        _sd = sounddevice
    return _sd
```

## Event Loop Policy

**For data collection** (`collect_gestures.py`): Use `WindowsSelectorEventLoopPolicy`. This works because we only need BLE connect/disconnect, not continuous notification callbacks.

```python
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

**For real-time streaming** (`realtime_gesture.py`, `server/run.py`): Do NOT use `WindowsSelectorEventLoopPolicy`. The SelectorEventLoop doesn't deliver BLE notification callbacks on Windows, causing the "stuck at buffering" issue. Use the default ProactorEventLoop + MTA CoInitializeEx instead.

## `input()` Blocks the Event Loop

**Problem:** In real-time scripts, calling `input()` in an async context blocks the entire event loop, stopping BLE notifications.

**Fix:** Use a non-blocking wrapper:

```python
async def async_input(prompt=""):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)
```

## BLE Disconnect During Long Operations

**Problem:** The Muse 2 disconnects if there's no BLE activity for ~10 seconds. Audio tone preview or user prompts before starting streaming can exceed this timeout.

**Fix:** Start BLE notification callbacks immediately after connecting, before any user interaction. The streaming keeps the connection alive.

## Muse 2 BLE Protocol

- **Control UUID:** `273e0001-...` — send `[0x02, 0x64, 0x0a]` to start, `[0x02, 0x68, 0x0a]` to stop
- **EEG UUIDs:** `273e0003-0006` for TP9, AF7, AF8, TP10
- **Packet format:** 12 samples per packet, 2 bytes each big-endian, first 2 bytes are counter
- **Scale:** 0.48828125 µV/LSB (raw 16-bit signed to microvolts)
- **Sample rate:** 256 Hz
