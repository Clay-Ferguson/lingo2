#!/bin/bash
# Setup script for Voice Typer GTK application
#
# Installs system dependencies, creates virtual environment,
# and installs Python packages.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

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
            python3-venv \
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
echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

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

# Install desktop file for dock/taskbar integration
echo "🖼️  Installing desktop integration..."
APP_ID="com.lingo.voicetyper"
DESKTOP_DIR="$HOME/.local/share/applications"

mkdir -p "$DESKTOP_DIR"

# Create desktop file (uses icon directly from project)
cat > "$DESKTOP_DIR/$APP_ID.desktop" << EOF
[Desktop Entry]
Name=Lingo Voice Typer
Comment=System-wide voice-to-text input
Exec=python3 $SCRIPT_DIR/voice_typer.py
Icon=$SCRIPT_DIR/lingo-logo.png
Terminal=false
Type=Application
Categories=Utility;Accessibility;
StartupWMClass=$APP_ID
EOF

update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

echo "✅ Desktop integration installed!"

echo ""
echo "✅ GTK app setup complete!"
echo ""
echo "Start the app with: ./run.sh"
echo ""
echo "Note: You may need to log out/in for the dock icon to appear."
echo ""
