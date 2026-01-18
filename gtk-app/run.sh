#!/bin/bash
# Start Voice Typer GTK application
#
# This script:
# 1. Creates a Python virtual environment (first run only)
# 2. Installs dependencies
# 3. Launches the voice typer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# Check for GTK4 system packages
if ! python3 -c "import gi; gi.require_version('Gtk', '4.0')" 2>/dev/null; then
    echo "❌ GTK4 Python bindings not found!"
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0"
    echo "  Fedora:        sudo dnf install python3-gobject gtk4"
    echo "  Arch:          sudo pacman -S python-gobject gtk4"
    echo ""
    exit 1
fi

# Check for PortAudio (required by sounddevice)
if ! ldconfig -p | grep -q libportaudio; then
    echo "❌ PortAudio library not found!"
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install libportaudio2 portaudio19-dev"
    echo "  Fedora:        sudo dnf install portaudio portaudio-devel"
    echo "  Arch:          sudo pacman -S portaudio"
    echo ""
    exit 1
fi

# Create virtual environment if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR" --system-site-packages
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check whisper setup
WHISPER_BINARY="../whisper-model/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL="../whisper-model/whisper.cpp/models/ggml-base.en.bin"

if [ ! -f "$WHISPER_BINARY" ] || [ ! -f "$WHISPER_MODEL" ]; then
    echo "❌ Whisper not set up!"
    echo ""
    echo "Run from project root:"
    echo "  ./setup-whisper.sh"
    echo ""
    exit 1
fi

echo "🎤 Starting Voice Typer..."
echo "   Click the button or press ESC to close"
echo ""

python3 voice_typer.py
