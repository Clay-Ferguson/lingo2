#!/bin/bash
# Start Voice Typer GTK application
#
# Automatically checks/repairs the virtual environment if needed.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# Quick check: can we import the required modules?
check_env() {
    "$VENV_DIR/bin/python" -c "import numpy, sounddevice, gi" 2>/dev/null
}

# Recreate venv with system-site-packages (needed for PyGObject/GTK)
setup_venv() {
    echo "🔧 Setting up Python environment..."
    rm -rf "$VENV_DIR"
    python3 -m venv --system-site-packages "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -q -r requirements.txt
    echo "✅ Environment ready"
    echo ""
}

# Check if venv exists and works, otherwise set it up
if [ ! -d "$VENV_DIR" ] || ! check_env; then
    setup_venv
fi

echo "🎤 Starting Voice Typer..."
echo "   Click the button or press ESC to close"
echo ""

"$VENV_DIR/bin/python" voice_typer.py
