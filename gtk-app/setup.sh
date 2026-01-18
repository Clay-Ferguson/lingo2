#!/bin/bash
# Setup script for Voice Typer GTK application
#
# Installs system dependencies needed for the app to work.

set -e

echo "🎤 Voice Typer - GTK App Setup"
echo "=============================="
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

case $PKG_MANAGER in
    apt)
        $SUDO apt update
        $SUDO apt install -y \
            python3-gi \
            python3-gi-cairo \
            gir1.2-gtk-4.0 \
            portaudio19-dev \
            ffmpeg
        ;;
    dnf)
        $SUDO dnf install -y \
            python3-gobject \
            gtk4 \
            portaudio-devel \
            ffmpeg
        ;;
    pacman)
        $SUDO pacman -S --noconfirm \
            python-gobject \
            gtk4 \
            portaudio \
            ffmpeg
        ;;
esac

echo ""
echo "✅ System dependencies installed!"
echo ""
echo "Next steps:"
echo "  1. Run the whisper setup (if not done): cd .. && ./setup-whisper.sh"
echo "  2. Start the app: ./run.sh"
echo ""
