# Lingo 2.0 🗣️

Local speech-to-text powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp). No cloud APIs, no costs, complete privacy.

This Lingo 2.0 project contains both a Web App and a GTK app, both of which use Whisper. The Web App is almost identical to the original [Lingo](https://github.com/Clay-Ferguson/lingo), under this same Github account (by Clay Ferguson), except the oridinal Lingo uses browser-based Speech API (for Voice Input) rather than Whisper.

For browser-based Speech I do recomment `Lingo`, rather than `Lingo 2.0`, just because, if you're already in a browser, there's no reason to use Whisper. 

## GTK App Screenshot

![](gtk-app/screenshot.png)

## Web App Screenshot

![](web-app/lingo-screenshot.png)

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
