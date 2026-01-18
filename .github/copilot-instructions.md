# Lingo 2.0 - AI Coding Instructions

## Project Overview
Lingo 2.0 is a **framework-free** web app for text-to-speech (TTS) and speech-to-text (STT). The architecture is intentionally simple: vanilla HTML/CSS/JS frontend + Python FastAPI backend for local Whisper transcription. No React, Vue, or build systems.

## Architecture

### Component Flow
```
Browser (web-app/lingo.html/js/css)  <-->  FastAPI (web-app/whisper_server.py:8009)  <-->  whisper.cpp CLI
                                           |
                                     ffmpeg (audio conversion)
```

### Key Files
- [web-app/lingo.js](web-app/lingo.js) - All frontend logic: TTS via Web Speech API, STT via MediaRecorder + silence detection
- [web-app/whisper_server.py](web-app/whisper_server.py) - FastAPI server: `/transcribe` endpoint, static file serving
- [setup-whisper.sh](setup-whisper.sh) - Builds whisper.cpp, downloads `base.en` model
- [web-app/run.sh](web-app/run.sh) - Creates venv, installs deps, starts server on port 8009
- [web-app/kill.sh](web-app/kill.sh) - Stops server by killing processes on port 8009

### Data Flow (Speech-to-Text)
1. Browser captures audio via `MediaRecorder` (webm/opus format)
2. Silence detection (~1s threshold) triggers chunk submission
3. Server converts webm → 16kHz mono WAV via ffmpeg
4. `whisper-cli` transcribes, returns text via JSON
5. Text inserted at cursor position in textarea

## Developer Commands

```bash
./setup-whisper.sh   # First-time setup: builds whisper.cpp, downloads model
cd web-app
./run.sh             # Start server (auto-creates .venv, installs fastapi/uvicorn)
./kill.sh            # Stop server
```

Server runs at `http://localhost:8009/lingo.html`

## Code Conventions

### Frontend (web-app/lingo.js)
- **Section markers**: Code organized with `// ============` comment blocks (TTS State, STT State, Utility Functions, etc.)
- **Button state management**: `updateReadButton()` and `updateMicButton()` sync UI with app state
- **Storage keys**: Prefixed with `tts_` or pattern `*_v1` for localStorage versioning
- **Status feedback**: All operations update `setStatus()` for user feedback

### Backend (web-app/whisper_server.py)
- **Paths**: Use `Path` from pathlib, all paths relative to `SCRIPT_DIR`
- **Temp files**: UUID-prefixed in system temp dir, cleaned up in `finally` block
- **Binary location**: `../whisper-model/whisper.cpp/build/bin/whisper-cli` (relative to web-app)
- **Model location**: `../whisper-model/whisper.cpp/models/ggml-base.en.bin` (relative to web-app)

## Key Configuration

### Silence Detection (web-app/lingo.js:40-43)
```javascript
const SILENCE_THRESHOLD = 0.01;     // RMS level for silence
const SILENCE_DURATION_MS = 1000;   // Silence before transcription trigger
const MIN_AUDIO_DURATION_MS = 500;  // Skip very short clips
```

### Whisper Settings (web-app/whisper_server.py:150-156)
- Language: English only (`--language en`)
- Threads: 4 (`--threads 4`)
- No timestamps (`--no-timestamps`)

## Common Modifications

**Change Whisper model**: Edit `WHISPER_MODEL` in web-app/whisper_server.py and model download in setup-whisper.sh

**Add keyboard shortcut**: Follow pattern in web-app/lingo.js event handler (~line 640):
```javascript
if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "x") {
  evt.preventDefault();
  // action
}
```

**Add API endpoint**: Add before the static files mount in web-app/whisper_server.py (line 208)
