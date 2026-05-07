# Lingo GTK: Voice Typer - GTK4 Desktop App 🎤

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](voice_typer.py)
[![GTK4](https://img.shields.io/badge/GTK-4.0-green?logo=gnome&logoColor=white)](voice_typer.py)
[![Framework Free](https://img.shields.io/badge/Framework_Free-orange)](voice_typer.py)
[![Whisper.cpp](https://img.shields.io/badge/Whisper.cpp-Local_STT-purple)](../whisper-model/whisper.cpp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE.md)

A lightweight Linux desktop application that provides system-wide voice-to-text input. Speak naturally and your words appear wherever your cursor is focused - in any application!

*Warning: This application has only been tested on Ubuntu Linux.*

![](screenshot.png)

## How It Works

1. **Select Microphone** - select your device for your mic
2. **Click Microphone Checkbox** - when the checkbox is checked aything you speak will be typed into wherever your edit cursor is, in any application system wide.  
3. **Speak naturally** - after 1 second of silence, your speech is transcribed
4. **Text is typed** wherever your cursor is focused

## Whisper Setup (from project root)

You only need to run this once.

```bash
cd /path/to/lingo2
./setup-whisper.sh
```

## Running the GTK App

This project uses [uv](https://docs.astral.sh/uv/) for Python environment management.

### First-time setup

Install uv (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install system dependencies (GTK4, PortAudio, ffmpeg) and create the venv:

```bash
cd gtk-app
./setup.sh
```

`setup.sh` installs OS-level packages (sudo required) and creates a `.venv`
with `--system-site-packages` so the system PyGObject/GTK4 bindings are
visible to the project.

### Running

```bash
./run.sh
```

…which is equivalent to:

```bash
uv sync --active
uv run --active voice_typer.py
```

### Adding dependencies

```bash
uv add <package>
```

This updates `pyproject.toml` and `uv.lock`. Run `uv sync --active` to
install on machines that already have a `.venv`.

## Troubleshooting & Tips

Troubleshooting and notes are [here](TROUBLESHOOTING.md)


