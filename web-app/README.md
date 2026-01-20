# Lingo 2.0 (Whisper-enabled Version) 🗣️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Vanilla_JS](https://img.shields.io/badge/Vanilla_JS-F7DF1E?logo=javascript&logoColor=black)](web-app/lingo.js)
[![Framework_Free](https://img.shields.io/badge/Framework_Free-brightgreen)](web-app/README.md)
[![Local AI](https://img.shields.io/badge/Local_AI-whisper--cpp-ff6600)](whisper-model/whisper.cpp/README.md)

Local speech-to-text powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp). No cloud APIs, no costs, complete privacy.

The "2.0" version of Lingo uses Whisper instead of Google Speech API for voice input. If you want the pure JavaScript-based Speech version of this app use Lingo instead of Lingo2. Both are on Github.

A powerful web application for text-to-speech (TTS) and speech-to-text (STT) functionality. Lingo provides an intuitive interface for reading text aloud and converting speech to text. **Speech recognition is powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp)** - OpenAI's Whisper model running 100% locally on your machine. No cloud APIs, no costs, complete privacy!

![Lingo Application Screenshot](lingo-screenshot.png)

**Rant to Developers:**
This project was very intentionally designed to be free of any specific frameworks (Vue, React, etc.) so it's easily understandable to all web developers and no porting from one framework to another is required. I also wanted this app to be runnable by non-developers simply by opening an HTML page locally. It seems to be the case that nearly *ALL* AI projects strugle endlessly with TTS/STT never getting it right. Gemini Voice sucks, OpenAI Voice sucks, Github Copilot Voice absolutely sucks, etc and yes voice can be a bit tricky to get right, but trust me this JS has it perfected. You guys no longer have any excuses! This code shows how easy it truly is.

## ✨ Features

### 🔊 Text-to-Speech (TTS)
- **Smart Reading**: Read selected text, from cursor position, or the entire document
- **Cursor Position Reading**: Place your cursor anywhere in the text to start reading from that point
- **Pause & Resume**: Pause speech at any time and resume where you left off
- **Voice Selection**: Choose from all available system voices with language indicators
- **Speed Control**: Adjustable speaking rates from slow (0.85x) to ludicrous (1.35x)
- **Persistent Settings**: Voice and speed preferences automatically saved
- **Cross-Browser Compatible**: Works in Chrome, Firefox, Safari, and other modern browsers

### 🎤 Speech Recognition (Whisper-powered)
- **Local Processing**: Uses whisper.cpp for accurate, private speech-to-text
- **Continuous Dictation**: Speak naturally with automatic silence detection
- **Smart Chunking**: After ~1 second of silence, audio is transcribed and inserted
- **Smart Insertion**: Text appears at cursor position, preserving existing content
- **Visual Feedback**: Textarea highlights red when actively listening
- **No Cloud APIs**: All processing happens on your machine

### ⌨️ Keyboard Shortcuts
- **Ctrl/Cmd + Enter**: Start/stop text reading
- **Ctrl/Cmd + M**: Toggle microphone dictation
- **Escape**: Stop all active operations (TTS or speech recognition)

### 🔗 URL Parameters
- **`?mic=on`**: Automatically start mic dictation when the page loads

### 🎨 User Interface
- **Dark Theme**: Easy-on-the-eyes default dark interface
- **Responsive Design**: Optimized for both desktop and mobile devices
- **Real-time Status**: Live feedback on current operations
- **Accessible**: Full keyboard navigation and screen reader support
- **Simple Architecture**: Clean separation of HTML, CSS, and JavaScript

## 🚀 Quick Start

### Prerequisites

Before running Lingo, you need:

- **Linux** (Ubuntu/Debian tested) or macOS
- **Python 3** with venv support (`sudo apt install python3-venv`)
- **ffmpeg** for audio conversion (`sudo apt install ffmpeg`)
- **Build tools** for compiling whisper.cpp (`sudo apt install build-essential cmake git`)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd whisper
   ```

2. **Run the setup script** to download and build whisper.cpp:
   ```bash
   ./setup-whisper.sh
   ```
   This will:
   - Clone whisper.cpp into `whisper-model/`
   - Build the whisper-cli binary
   - Download the `base.en` model (~150MB)

3. **Start the server**:
   ```bash
   cd web-app
   ./run.sh
   ```
   This will:
   - Create a Python virtual environment (first run only)
   - Install FastAPI and dependencies
   - Start the server on http://localhost:8009

4. **Open your browser** to http://localhost:8009/lingo.html

## 📋 How to Use

### Text-to-Speech
1. **Type or paste text** into the main textarea
2. **Drag and drop text** from other applications directly into the textarea (especially handy on Linux - simply select text from any app and drag it in)
3. **Position your cursor** where you want reading to begin, or **select specific text** to read only that portion
4. **Click "🔊 Read"** or press **Ctrl/Cmd + Enter**
5. **Choose your preferred voice** and speaking speed from the dropdowns
6. **Click "⏸️ Pause"** to pause reading, then **"▶️ Resume"** to continue from where you left off
7. **Click "⏹️ Stop"** or press **Escape** to stop reading completely

> **Tip**: If your cursor is at the very end of the text, clicking Read Aloud will start from the beginning.

### Speech Recognition
1. **Click "🎤 Mic"** or press **Ctrl/Cmd + M**
2. **Speak clearly** - your words will appear in the textarea
3. **The app continues listening** until you stop it manually
4. **Click "⏹️ Stop"** or press **Escape** to stop dictation

> **Tip**: Add `?mic=on` to the URL to automatically start mic dictation when the page loads (e.g., `http://localhost:8009/lingo.html?mic=on`).

## 🛠️ Technical Details

### Architecture
```
Browser (web-app/lingo.html/js)  →  HTTP POST (audio blob)  →  FastAPI Server  →  whisper.cpp  →  text response
```

1. **Browser** captures audio via `MediaRecorder` API
2. **Silence detection** (1 second) triggers transcription
3. **Audio blob** sent to FastAPI server at `/transcribe`
4. **Server** converts to 16kHz mono WAV using ffmpeg
5. **whisper.cpp** transcribes the audio locally
6. **Text** returned and inserted at cursor position

### Browser Compatibility
| Browser | TTS Support | Speech Recognition (Whisper) |
|---------|-------------|-----------------------------|
| Chrome/Chromium | ✅ Full | ✅ Full |
| Firefox | ✅ Full | ✅ Full |
| Safari | ✅ Full | ✅ Full |
| Edge | ✅ Full | ✅ Full |

> **Note**: Since speech recognition now uses a local server + whisper.cpp instead of browser APIs, it works in all modern browsers!

### Technologies Used
- **whisper.cpp**: C++ port of OpenAI's Whisper model for speech-to-text
- **FastAPI**: Python web server handling audio processing
- **ffmpeg**: Audio format conversion (webm → 16kHz mono WAV)
- **Speech Synthesis API**: Browser-native text-to-speech
- **MediaRecorder API**: Browser audio capture
- **Web Audio API**: Silence detection via AudioContext/AnalyserNode
- **localStorage**: Persistent settings

## 📁 Project Structure

```
lingo2/
├── setup-whisper.sh        # Setup script: clones/builds whisper.cpp
├── whisper-model/          # Created by setup-whisper.sh
│   └── whisper.cpp/        # whisper.cpp repo with binary and models
├── web-app/                # Web application
│   ├── lingo.html          # Main HTML structure
│   ├── lingo.css           # Styles and theming  
│   ├── lingo.js            # Frontend: audio capture, silence detection, UI
│   ├── whisper_server.py   # FastAPI server: audio conversion + whisper.cpp
│   ├── run.sh              # Startup script: creates venv, starts server
│   ├── kill.sh             # Stop the local server
│   └── .venv/              # Python virtual environment (created by run.sh)
├── gtk-app/                # GTK desktop application (coming soon)
└── README.md               # This documentation
```

## 🔧 The Scripts

### `setup-whisper.sh` (project root)
One-time setup that:
1. Clones the whisper.cpp repository
2. Builds the whisper-cli binary using cmake
3. Downloads the `base.en` model (~150MB)

### `web-app/run.sh`
Starts the web application:
1. Creates Python virtual environment (first run)
2. Installs FastAPI, uvicorn, python-multipart
3. Kills any existing server on port 8009
4. Starts the FastAPI server with hot-reload

### `web-app/kill.sh`
Stops any running server on port 8009.

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

#### 2. `web-app/whisper_server.py` (line ~35)

Update the model path to match:

```python
# Change from:
WHISPER_MODEL = WHISPER_DIR / "whisper.cpp" / "models" / "ggml-base.en.bin"

# To (for example, small.en):
WHISPER_MODEL = WHISPER_DIR / "whisper.cpp" / "models" / "ggml-small.en.bin"
```

#### 3. Re-run setup and restart

```bash
# Download the new model
./setup-whisper.sh

# Restart the server
cd web-app
./kill.sh
./run.sh
```

> **Tip**: You can have multiple models downloaded. Just change `whisper_server.py` to switch between them without re-downloading.

## 🎯 Use Cases

- **Content Review**: Listen to written content for proofreading
- **Accessibility**: Assist users with reading difficulties
- **Multitasking**: Consume text content while doing other activities
- **Note Taking**: Quickly dictate thoughts and ideas
- **Language Learning**: Hear proper pronunciation of text
- **Voice Memos**: Convert speech to text for documentation

## 🔍 Development Features

### Console Helpers
For testing and debugging, Lingo exposes utility functions:
```javascript
// Speak any text directly
window.__tts.speakNow("Hello world");

// Stop current speech
window.__tts.cancel();
```

### Error Handling
- **Graceful Degradation**: Features disable cleanly when unsupported
- **Auto-Recovery**: Speech recognition restarts automatically after interruptions
- **User Feedback**: Clear status messages for all operations

## 🤝 Contributing

Lingo follows a simple three-file architecture for easy maintenance. When contributing:

1. **Keep files organized** - HTML in `lingo.html`, styles in `lingo.css`, logic in `lingo.js`
2. **Test across browsers** - especially Chrome vs Firefox
3. **Maintain responsive design** - mobile and desktop compatibility
4. **Preserve accessibility** - keyboard navigation and screen readers

## 📝 License

This project is open source. Feel free to use, modify, and distribute as needed.

## 🔮 Future Enhancements

- **Electron app integration**: Package as a standalone desktop app
- **Different Whisper models**: Support for tiny (faster) or medium/large (more accurate)
- **Real-time streaming**: Process audio in smaller chunks for faster feedback
- **Language selection**: Support for non-English Whisper models
- **GPU acceleration**: CUDA/Metal support for faster transcription

---

**Lingo** - Bringing voice to your text and text to your voice! 🎙️✨
