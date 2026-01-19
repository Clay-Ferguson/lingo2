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
