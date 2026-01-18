#!/bin/bash
# Start Voice Typer GTK application
#
# Prerequisites: Run ./setup.sh first to install dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"

# Check that setup has been run
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found!"
    echo ""
    echo "Please run ./setup.sh first"
    echo ""
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "🎤 Starting Voice Typer..."
echo "   Click the button or press ESC to close"
echo ""

python3 voice_typer.py
