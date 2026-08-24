#!/bin/bash
# Setup script for Voice Typer (PyQt6)
#
# Installs system dependencies and the desktop entry. The Python environment
# itself needs no setup step -- run.sh lets uv build it on first launch.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🎤 Voice Typer - PyQt6 App Setup"
echo "================================"
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Detect package manager
if command -v apt &> /dev/null; then
    PKG_MANAGER="apt"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
else
    echo "❌ Unsupported package manager. Please install dependencies manually."
    exit 1
fi

echo "📦 Installing system dependencies..."
echo ""

# No PyGObject/GTK packages here: PyQt6 ships Qt itself in its wheels, which is
# the whole reason the venv no longer needs --system-site-packages. Qt's
# platform plugin does still link against a few system X/xkb libraries that a
# minimal install may lack -- libxcb-cursor0 in particular is required by
# Qt 6.5+, and its absence produces the notorious
# "could not load the Qt platform plugin xcb" failure.
case $PKG_MANAGER in
    apt)
        $SUDO apt update
        $SUDO apt install -y \
            portaudio19-dev \
            ffmpeg \
            libxcb-cursor0 \
            libxkbcommon-x11-0
        ;;
    dnf)
        $SUDO dnf install -y \
            portaudio-devel \
            ffmpeg \
            xcb-util-cursor \
            libxkbcommon-x11
        ;;
    pacman)
        $SUDO pacman -S --noconfirm \
            portaudio \
            ffmpeg \
            xcb-util-cursor \
            libxkbcommon-x11
        ;;
esac

echo ""
echo "✅ System dependencies installed!"
echo ""

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

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check whisper setup
WHISPER_BINARY="../whisper-model/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL="../whisper-model/whisper.cpp/models/ggml-base.en.bin"

if [ ! -f "$WHISPER_BINARY" ] || [ ! -f "$WHISPER_MODEL" ]; then
    echo ""
    echo "⚠️  Whisper not set up yet!"
    echo ""
    echo "Run from project root:"
    echo "  ./setup-whisper.sh"
    echo ""
fi

# Install desktop file for dock/taskbar integration.
# Exec/Path/Icon are absolute paths, so the checked-in template is rewritten
# for wherever this checkout actually lives.
echo "🖼️  Installing desktop integration..."
APP_ID="com.lingo.voicetyper"
DESKTOP_DIR="$HOME/.local/share/applications"

chmod +x "$SCRIPT_DIR/run.sh"
mkdir -p "$DESKTOP_DIR"

sed \
  -e "s|^Exec=.*|Exec=$SCRIPT_DIR/run.sh|" \
  -e "s|^Path=.*|Path=$SCRIPT_DIR|" \
  -e "s|^Icon=.*|Icon=$SCRIPT_DIR/lingo-logo.png|" \
  "$SCRIPT_DIR/$APP_ID.desktop" > "$DESKTOP_DIR/$APP_ID.desktop"

update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

echo "✅ Desktop integration installed!"

echo ""
echo "✅ PyQt6 app setup complete!"
echo ""
echo "Start the app with: ./run.sh"
echo ""
echo "Note: You may need to log out/in for the dock icon to appear."
echo ""
