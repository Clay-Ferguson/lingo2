# Lingo 2.0 🗣️

Local speech-to-text powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp). No cloud APIs, no costs, complete privacy.

## Projects

This mono-repo contains two applications that provide different ways to use whisper.cpp for voice input:

| Project | Description |
|---------|-------------|
| [**web-app**](web-app/README.md) | Browser-based TTS/STT with a FastAPI backend. Access via http://localhost:8009 |
| [**gtk-app**](gtk-app/README.md) | Linux desktop app for system-wide voice typing. Speaks into any focused application |

Both projects share the same whisper.cpp engine located in `whisper-model/`.

## Quick Start

1. **Build whisper.cpp and download the model:**
   ```bash
   ./setup-whisper.sh
   ```

2. **Run whichever app you prefer:**
   ```bash
   # Web app (browser-based)
   cd web-app && ./run.sh
   
   # GTK app (Linux desktop)
   cd gtk-app && ./run.sh
   ```

## Requirements

- Linux (Ubuntu/Debian tested) or macOS
- Python 3 with venv support
- ffmpeg
- Build tools (cmake, git, build-essential)

See each project's README for additional dependencies.
