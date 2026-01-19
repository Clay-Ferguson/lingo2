# Lingo GTK: Voice Typer - GTK4 Desktop App 🎤

A lightweight Linux desktop application that provides system-wide voice-to-text input. Speak naturally and your words appear wherever your cursor is focused - in any application!

![](screenshot.png)

## How It Works

1. **Select Microphone** - select your device for your mic
2. **Click Microphone Checkbox** - when the checkbox is checked aything you speak will be typed into wherever your edit cursor is, in any application system wide.  
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

## Tips

This application works best in a quiet room, with the threshold setting (on the GUI) set so low that it will even pick up mouse clicks and keyboard entry noises from time to time. The background of the app will turn green whenever it has detected something on your microphone, and it will successfully ignore things that aren't speech. 

However if you set the threshold too high, the input can sometimes miss chunks of your speech, so if you run into any problems at all, the first thing to try is to lower the threshold value. On a high quality microphone in a quiet room an ideal threshold may be as low as "0.001" but if you're in a more noisy environment, you might need to set the threshold up to "0.02" or even higher.

Whenever the microphone is detecting sound the background will turn green as a visual cue, however, if you see the application continuing to stay green constantly, that means you've got the threshold set too low. A common way this can happen is when you adjust the microphone threshold to what you think is a perfect value in a perfectly quiet room with total silence, and eventually perhaps your air conditioning system turns on creating some background noise, a low enough threshold setting can cause that to trigger the microphone continuously, which makes the application unable to work. So in general, if you see the green background continuously, then you need to raise the threshold setting.  

However, it's perfectly fine and probably preferable to keep the threshold set so low that random  sounds like setting a cup down on a desk or whatever, can trigger the microphone (indicated by the green background), because it will successfully ignore things that are not detected as speech.

## Troubleshooting

### "No audio input devices found"

```bash
# List audio devices
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Check PulseAudio/PipeWire
pactl list sources short
```

### Keyboard typing not working (Wayland)

This app uses the **XDG Remote Desktop Portal** for keyboard input on Wayland. On first use:

1. A system dialog will appear: *"Allow Voice Typer to remote control your desktop?"*
2. Click **Allow** to grant keyboard access
3. **The permission is saved** - you won't see this dialog again (uses Portal v2 `persist_mode`)

The restore token is stored in `~/.config/lingo-gtk.yaml`. If you ever need to reset permissions (e.g., after a system update breaks things), delete the `portal_restore_token` entry from that file.

If the permission dialog doesn't appear or typing still doesn't work:

```bash
# Check that xdg-desktop-portal is running
systemctl --user status xdg-desktop-portal

# Restart it if needed
systemctl --user restart xdg-desktop-portal
systemctl --user restart xdg-desktop-portal-gnome  # or -gtk, -kde, etc.
```

**Note:** If you deny the permission, transcribed text will still be logged to the console but won't be typed.

### Window doesn't appear in corner (Wayland)

Wayland restricts window positioning. The window will appear but may not be in the exact corner. This is a Wayland security feature.

### Permission denied for audio

```bash
# Add user to audio group
sudo usermod -a -G audio $USER
# Log out and back in
```

### Microphone not working or wrong device selected

If the microphone stops responding or behaves unexpectedly, check if:

1. **Your selected microphone reverts to a different device** - If the GUI dropdown no longer shows your configured microphone as selected, this typically indicates another application has grabbed exclusive access to the device.

2. **Audio is not being captured** - The app may appear to be listening but no transcription occurs.

**Common causes:**
- The **web-based Lingo app** (or another browser tab) is using the microphone
- Another application (video call, screen recorder, etc.) has the microphone open

**Solution:**
1. Close any browser tabs running the web version of Lingo
2. Close any other applications that might be using the microphone
3. Restart this GTK application (`./run.sh`)

Your microphone selection is saved to `~/.config/lingo-gtk.yaml` and will be restored on restart.

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
