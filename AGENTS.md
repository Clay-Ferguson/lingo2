# Lingo 2.0 - AI Coding Instructions

## Project Overview
Lingo 2.0 provides **local speech-to-text** via whisper.cpp with two active apps:
1. **web-app/** - Browser-based TTS/STT with FastAPI backend (port 8009)
2. **qt-app/** - System-wide voice typing for Linux (types into any focused app)

**gtk-app/** is the original GTK4 version of the desktop app, kept as a **deprecated** project for anyone who needs a GTK build. It is frozen: make changes in `qt-app/` instead, and do not port fixes back unless asked.

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
- **Device selection**: dropdown in the Settings dialog lets user pick microphone; saved to config file
- **UI shape**: the main window is one row -- mic checkbox and a gear -- and never changes size. Settings open in a separate modeless dialog (`settings_dialog.py`). There is no in-window close button; the title bar's is the same size and a few pixels away
- **Window chrome**: the title bar and window frame are the platform's own, unstyled. The mic checkbox's enlarged indicator comes from `windowchrome`, a library of ours in its own repository -- see the entries under "Deliberate non-obvious choices" below
- **Thread bridge**: the PortAudio callback and whisper worker threads reach the UI through `pyqtSignal`, never by touching widgets directly

#### Deliberate non-obvious choices - do not "fix" these

- **The config file is still named `lingo-gtk.yaml`.** Renaming it would orphan the saved `portal_restore_token` and force users through the Remote Desktop permission dialog again.
- **`_send_key` blocks on `bus.call()` instead of using fire-and-forget `bus.send()`.** The round-trip is load-bearing back-pressure: without it, key events outrun the compositor and arrive scrambled and truncated.
- **uint32 D-Bus arguments must be `QDBusArgument(v, QMetaType.Type.UInt)`.** A plain Python int marshals as int32 and the portal rejects the message. Since key events are sent without checking a reply, getting this wrong fails silently.
- **`a{sv}` options are plain Python dicts.** Wrapping the values in `QDBusVariant` crashes xdg-desktop-portal outright.
- **The RMS silence-detection state machine in `audio.py` is tuned**, not arbitrary. Changing the constants or the branch structure causes missed or spurious utterances.
- **`_enforce_size()` re-`resize()`s the window from `resizeEvent`.** Mutter replays a stale size in the configure that comes with focus changes, and Qt applies it instead of clamping to the window's min/max, which used to collapse the Settings panel and make the widgets overlap. `setFixedSize()` cannot undo it -- it returns early when the min and max are already correct, which they are. Settings is now a separate dialog and the main window never resizes itself, so this should never fire; it is kept as a belt and logs when it corrects.
- **The main window must never resize itself.** That is why Settings lives in `settings_dialog.py` rather than an expanding panel: a self-resizing, non-resizable Wayland toplevel is the rare path that produced the bug above. The dialog is deliberately *not* fixed-size, so it does not take that path.
- **The settings dialog never touches the config file or the recorder.** It validates input and emits signals; `window.py` applies them. Two of the three settings have live side effects (the threshold reaches a running recorder, a device change stops the mic), and the window is the only thing that owns both.
- **The icon buttons are painted white once, at construction, and never change.** An icon is a pixmap and does not follow the stylesheet's `color` rule, so it has to be painted by hand -- but one fixed color is all this window needs, so it does not track the phase. Only the `-symbolic` theme icons are painted; the full-color fallbacks silhouette into a blob, so they are used as they ship.
- **`on_processing_phase_changed` repolishes the mic checkbox as well as the window.** A repolish reaches exactly one widget, and the rule that flips the checkbox text black for the white phase lives on the checkbox.
- **`sounddevice` is imported lazily**, so a missing PortAudio produces the friendly dependency message instead of an import traceback.
- **`windowchrome` is a sibling library, two levels up.** `[tool.uv.sources]` in `qt-app/pyproject.toml` resolves it at `../../windowchrome` -- **two** levels up, unlike every other app that uses it, because `qt-app` sits a directory deeper than they do. It has to be there or `uv run` fails with an unresolved path dependency. The only thing this app takes from it is the entry below; it has no setup call and no ordering rule.
- **The mic checkbox's enlarged indicator is `windowchrome.apply_checkboxes()`, not a stylesheet rule.** It used to be `QCheckBox::indicator { width: 24px; height: 24px; }` in `build_stylesheet()`; it is now the shared component the other apps use, so the four of them enlarge a check box the same way. It has to be a `QProxyStyle` applied to the widget rather than a rule in the sheet -- `windowchrome/README.md` §12 has the reasoning -- so do not move it back into `build_stylesheet()` when tidying that function. The font size and the checked weight stay in the sheet, which is what a stylesheet is good for. Verified that the enlarged indicator survives the phase repolish below.
- **Checking the mic does not recolor its label.** It went red (`color: #e53935`) until that was removed: the window background is already reporting the pipeline's state in three colors chosen for the job, and a fourth color on the text competed with it. The label now keeps the theme's foreground in every state -- white on a dark desktop, black on a light one -- and the bold on `:checked` is what marks it. Do not pin it to a hex value: the 'typing' phase paints the window white, so a hard-coded white label would disappear into it, which is the very thing the `phase="typing"` rule exists to prevent.
- **The title bar and window frame are the platform's own, and deliberately unstyled.** They used to be colored through `windowchrome`, which on Wayland meant choosing Qt's `bradient` decoration plugin and repurposing the application palette's `Window`/`WindowText` roles and the application font for it, then handing them back to every widget through an application-wide event filter. That was removed as too fragile -- it rested on undocumented plugin internals and leaked into unrelated code (the settings dialog's muted help text had to route around it). Do not reintroduce title-bar or frame coloring, and do not wrap the window in anything to paint a border: the phase colors live on the plain toplevel `QWidget#voiceTyperWindow`.

## Silence Detection Config

Both apps use similar silence detection (adjust for your mic):
```python
# qt-app: DEFAULT_SILENCE_THRESHOLD lives in config.py, the rest in audio.py.
# The threshold is also user-editable at runtime in the Settings dialog.
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

**Tune for quiet mics**: Lower the silence threshold in the Settings dialog, check RMS in qt-app/voice_typer.log

## Dependencies

**Shared** (both apps): `ffmpeg`, whisper.cpp (built via `./setup-whisper.sh`)

**web-app**: `fastapi`, `uvicorn`, `python-multipart` (auto-installed by run.sh)

**qt-app** (run `./setup.sh` or manually install):
- System: `portaudio19-dev`, `ffmpeg`, `libxcb-cursor0`, `libxkbcommon-x11-0` (Ubuntu/Debian names)
- Python (via pyproject.toml/uv): `PyQt6`, `sounddevice`, `numpy`, `PyYAML`, and `windowchrome` -- the last of which is **not on PyPI**: it is resolved by path from `../../windowchrome`, a checkout that must sit beside the `lingo2` directory (not beside `qt-app`)
- Keyboard injection: XDG Remote Desktop Portal via QtDBus (part of PyQt6; no PyGObject)

## Note to AI Agents

Do not ever commit code to the 'git' repo yourself. This is always only done by human developers.

