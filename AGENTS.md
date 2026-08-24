# Lingo 2.0 - AI Coding Instructions

## Project Overview
Lingo 2.0 provides **local speech-to-text** via whisper.cpp with two active apps:
1. **web-app/** - Browser-based TTS/STT with FastAPI backend (port 8009)
2. **qt-app/** - System-wide voice typing for Linux (types into any focused app)

**gtk-app/** is the original GTK4 version of the desktop app, kept as a
**deprecated** project for anyone who needs a GTK build. It is frozen: make
changes in `qt-app/` instead, and do not port fixes back unless asked.

**Philosophy**: Framework-free. No React/Vue/build systems. Vanilla HTML/CSS/JS + Python.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ web-app (Browser)                    │ qt-app (Linux Desktop)               │
│ lingo.html/js/css                    │ voice_typer/ (PyQt6)                 │
│     │                                │     │                                │
│     ▼                                │     ▼                                │
│ FastAPI (whisper_server.py:8009)     │ sounddevice → whisper-cli            │
│     │                                │     │                                │
│     └──────────┬─────────────────────┴─────┘                                │
│                ▼                                                            │
│          whisper-model/whisper.cpp/build/bin/whisper-cli                    │
│          whisper-model/whisper.cpp/models/ggml-base.en.bin                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Developer Commands

```bash
./setup-whisper.sh      # First-time: builds whisper.cpp, downloads base.en model

# Web app
cd web-app && ./run.sh  # Start server → http://localhost:8009/lingo.html
./kill.sh               # Stop server

# Qt app (system-wide voice typing)
cd qt-app && ./run.sh   # Launch floating mic button
```

## Code Conventions

### Web Frontend (web-app/lingo.js)
- **Section markers**: `// ============` blocks organize code (TTS State, STT State, etc.)
- **Button sync**: `updateReadButton()` / `updateMicButton()` keep UI in sync with state
- **Storage keys**: `tts_` prefix or `*_v1` suffix for localStorage versioning
- **Status bar**: All ops call `setStatus()` for user feedback

### Python Backend (whisper_server.py, voice_typer/)
- **Paths**: Always use `Path` from pathlib, relative to `SCRIPT_DIR`
- **Temp files**: UUID-prefixed, cleaned in `finally` block
- **Whisper paths** (relative to each app):
  - Binary: `../whisper-model/whisper.cpp/build/bin/whisper-cli`
  - Model: `../whisper-model/whisper.cpp/models/ggml-base.en.bin`

### Qt App Specifics (qt-app/voice_typer/)
- **Audio pipeline**: sounddevice (48kHz) → resample to 16kHz → normalize → whisper-cli
- **Keyboard injection**: XDG Remote Desktop Portal over **QtDBus** (Wayland-safe)
- **Logging**: Writes to `qt-app/voice_typer.log` (overwritten each run)
- **Config file**: `~/.config/lingo-gtk.yaml` stores user preferences (microphone selection)
- **Device selection**: GUI dropdown lets user pick microphone; saved to config file
- **Thread bridge**: the PortAudio callback and whisper worker threads reach the
  UI through `pyqtSignal`, never by touching widgets directly

#### Deliberate non-obvious choices - do not "fix" these

- **The config file is still named `lingo-gtk.yaml`.** Renaming it would orphan
  the saved `portal_restore_token` and force users through the Remote Desktop
  permission dialog again.
- **`_send_key` blocks on `bus.call()` instead of using fire-and-forget
  `bus.send()`.** The round-trip is load-bearing back-pressure: without it, key
  events outrun the compositor and arrive scrambled and truncated.
- **uint32 D-Bus arguments must be `QDBusArgument(v, QMetaType.Type.UInt)`.** A
  plain Python int marshals as int32 and the portal rejects the message. Since
  key events are sent without checking a reply, getting this wrong fails silently.
- **`a{sv}` options are plain Python dicts.** Wrapping the values in
  `QDBusVariant` crashes xdg-desktop-portal outright.
- **The RMS silence-detection state machine in `audio.py` is tuned**, not
  arbitrary. Changing the constants or the branch structure causes missed or
  spurious utterances.
- **`_enforce_size()` re-`resize()`s the window from `resizeEvent`.** Mutter
  replays a stale size in the configure that comes with focus changes, and Qt
  applies it instead of clamping to the window's min/max, which collapses the
  Settings panel and makes the widgets overlap. `setFixedSize()` cannot undo it
  -- it returns early when the min and max are already correct, which they are.
- **`sounddevice` is imported lazily**, so a missing PortAudio produces the
  friendly dependency message instead of an import traceback.

## Silence Detection Config

Both apps use similar silence detection (adjust for your mic):
```python
# qt-app: DEFAULT_SILENCE_THRESHOLD lives in config.py, the rest in audio.py.
# The threshold is also user-editable at runtime in the Settings panel.
DEFAULT_SILENCE_THRESHOLD = 0.005  # RMS threshold
SILENCE_DURATION_S = 1.0           # Seconds of silence → transcribe
MIN_AUDIO_DURATION_S = 0.5         # Skip very short clips

# web-app/lingo.js (browser)
const SILENCE_THRESHOLD = 0.01;
const SILENCE_DURATION_MS = 1000;
```

## Common Modifications

**Change Whisper model**: Update `WHISPER_MODEL` in whisper_server.py AND qt-app/voice_typer/transcribe.py, plus model download in setup-whisper.sh

**Add web keyboard shortcut** (web-app/lingo.js ~line 640):
```javascript
if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "x") {
  evt.preventDefault();
  // action
}
```

**Add API endpoint**: Insert before static files mount in whisper_server.py (line ~208)

**Tune for quiet mics**: Lower the silence threshold in the Settings panel, check RMS in qt-app/voice_typer.log

## Dependencies

**Shared** (both apps): `ffmpeg`, whisper.cpp (built via `./setup-whisper.sh`)

**web-app**: `fastapi`, `uvicorn`, `python-multipart` (auto-installed by run.sh)

**qt-app** (run `./setup.sh` or manually install):
- System: `portaudio19-dev`, `ffmpeg`, `libxcb-cursor0`, `libxkbcommon-x11-0` (Ubuntu/Debian names)
- Python (via pyproject.toml/uv): `PyQt6`, `sounddevice`, `numpy`, `PyYAML`
- Keyboard injection: XDG Remote Desktop Portal via QtDBus (part of PyQt6; no PyGObject)

## Note to AI Agents

Do not ever commit code to the 'git' repo yourself. This is always only done by human developers.

