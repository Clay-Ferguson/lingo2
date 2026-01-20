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

## 🔧 First setup Whisper AI

### `setup-whisper.sh` (project root)
One-time setup that:
1. Clones the whisper.cpp repository
2. Builds the whisper-cli binary using cmake
3. Downloads the `base.en` model (~150MB)

## 🧠 Upgrading the AI Model

This project uses the `base.en` model by default, which offers a good balance of speed and accuracy. If you need better accuracy (at the cost of speed) or faster performance (at the cost of accuracy), you can switch to a different model.

### Available Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `tiny.en` | ~75MB | Fastest | Decent | Quick testing, low-powered devices |
| `base.en` | ~150MB | Fast | Good | ⭐ **Default** - good balance |
| `small.en` | ~500MB | Medium | Better | Improved accuracy without too much slowdown |
| `medium.en` | ~1.5GB | Slower | Great | High accuracy needs |
| `large` | ~3GB | Slowest | Best | Maximum accuracy, multilingual |

> **Note**: The `.en` suffix means English-only models, which are smaller and faster. The `large` model is multilingual (no `.en` variant).

### How to Switch Models

You need to edit **two files**:

#### 1. `setup-whisper.sh` (line ~91)

Change the model name in the download command:

```bash
# Change from:
./models/download-ggml-model.sh base.en

# To (for example, small.en):
./models/download-ggml-model.sh small.en
```

#### 2. Set Python Variable

Both the 'gtk-app' and the 'web-app' have this same variable definition which tells it which whisper model to use.

```python
# Change from:
WHISPER_MODEL = WHISPER_DIR / "whisper.cpp" / "models" / "ggml-base.en.bin"

# To (for example, small.en):
WHISPER_MODEL = WHISPER_DIR / "whisper.cpp" / "models" / "ggml-small.en.bin"
```

#### 3. Re-run whisper setup and restart

```bash
# Download the new model
./setup-whisper.sh
```
Next you can restart the app.

> **Tip**: You can have multiple models downloaded. Just change `whisper_server.py` to switch between them without re-downloading.

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
