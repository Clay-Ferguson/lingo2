# Voice Typer - GTK4 Desktop App 🎤

A lightweight Linux desktop application that provides system-wide voice-to-text input. Speak naturally and your words appear wherever your cursor is focused - in any application!

## How It Works

1. **Trigger the app** via a keyboard shortcut (you configure this)
2. **A small floating button** appears in the upper-right corner
3. **Speak naturally** - after 1 second of silence, your speech is transcribed
4. **Text is typed** wherever your cursor is focused
5. **Click the button** or press **ESC** to close

## Prerequisites

### 1. Whisper Setup (from project root)

```bash
cd /path/to/lingo2-whisper
./setup-whisper.sh
```

### 2. System Packages

**Ubuntu/Debian:**
```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 python3-venv
```

**Fedora:**
```bash
sudo dnf install python3-gobject gtk4
```

**Arch:**
```bash
sudo pacman -S python-gobject gtk4
```

### 3. Audio Access

Make sure your user is in the `audio` group:
```bash
groups $USER  # Should include 'audio'
```

## Running

```bash
cd gtk-app
./run.sh
```

The first run will create a virtual environment and install Python dependencies.

## Setting Up a Keyboard Shortcut

### GNOME (Ubuntu default)

1. Open **Settings** → **Keyboard** → **Keyboard Shortcuts**
2. Scroll to bottom, click **Custom Shortcuts**
3. Click **+** to add:
   - **Name:** Voice Typer
   - **Command:** `/full/path/to/lingo2-whisper/gtk-app/run.sh`
   - **Shortcut:** Choose your preferred key combo (e.g., `Super+V`)

### KDE Plasma

1. Open **System Settings** → **Shortcuts** → **Custom Shortcuts**
2. Click **Edit** → **New** → **Global Shortcut** → **Command/URL**
3. Set the trigger and command

### i3/Sway

Add to your config:
```
bindsym $mod+v exec /path/to/lingo2-whisper/gtk-app/run.sh
```

### Generic (xbindkeys)

Install xbindkeys and add to `~/.xbindkeysrc`:
```
"/path/to/lingo2-whisper/gtk-app/run.sh"
    Mod4 + v
```

## Configuration

Edit `voice_typer.py` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `SILENCE_THRESHOLD` | 0.01 | RMS level below this is silence |
| `SILENCE_DURATION_S` | 1.0 | Seconds of silence before transcription |
| `MIN_AUDIO_DURATION_S` | 0.5 | Minimum speech length to process |

## Troubleshooting

### "No audio input devices found"

```bash
# List audio devices
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Check PulseAudio/PipeWire
pactl list sources short
```

### Keyboard typing not working (Wayland)

On Wayland, `pynput` may have limited functionality. Options:
1. Use X11 session instead
2. Install `ydotool` as an alternative

### Window doesn't appear in corner (Wayland)

Wayland restricts window positioning. The window will appear but may not be in the exact corner. This is a Wayland security feature.

### Permission denied for audio

```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Log out and back in
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              GTK4 Application                    │
│  ┌──────────────────────────────────────────┐   │
│  │  Floating Window (upper-right corner)     │   │
│  │  [🎤 mic button] ← click or ESC to close  │   │
│  └──────────────────────────────────────────┘   │
│                      │                           │
│              ┌───────┴───────┐                   │
│              ▼               ▼                   │
│  ┌─────────────────┐  ┌─────────────────┐       │
│  │  Audio Thread   │  │  Key Listener   │       │
│  │  (sounddevice)  │  │  (pynput/ESC)   │       │
│  └────────┬────────┘  └─────────────────┘       │
│           │                                      │
│           ▼                                      │
│  ┌─────────────────────────────────────────┐    │
│  │  Silence Detection (RMS < 0.01 for 1s)  │    │
│  └────────┬────────────────────────────────┘    │
│           │                                      │
│           ▼                                      │
│  ┌─────────────────────────────────────────┐    │
│  │  whisper-cli (subprocess)               │    │
│  │  ../whisper-model/whisper.cpp/...       │    │
│  └────────┬────────────────────────────────┘    │
│           │                                      │
│           ▼                                      │
│  ┌─────────────────────────────────────────┐    │
│  │  pynput keyboard.type(text)             │    │
│  │  → Types wherever cursor is focused     │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## Files

```
gtk-app/
├── voice_typer.py    # Main application (~300 lines)
├── requirements.txt  # Python dependencies
├── run.sh           # Launch script (creates venv, installs deps)
└── README.md        # This file
```

## Differences from Web App

| Feature | Web App | GTK App |
|---------|---------|---------|
| Audio capture | MediaRecorder (webm) | sounddevice (PCM) |
| Audio conversion | ffmpeg required | Direct to WAV |
| Text output | Textarea insertion | System-wide typing |
| UI | Full browser page | Tiny floating button |
| Trigger | Always open in browser | Hotkey launches app |

---

**Voice Typer** - Speak anywhere, type everywhere! 🎙️✨
