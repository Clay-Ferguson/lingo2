#!/bin/bash

# Lingo with Whisper Speech-to-Text
# This script starts the FastAPI server that serves the web app and handles transcription

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=8009
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=========================================="
echo "  Lingo - Speech-to-Text with Whisper    "
echo "=========================================="

# Check if required commands are available
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Use the virtual environment's Python explicitly
PYTHON="$VENV_DIR/bin/python"

# Check for uvicorn/fastapi and install if needed
if ! "$PYTHON" -c "import uvicorn" 2>/dev/null; then
    echo "Installing required Python packages..."
    "$VENV_DIR/bin/pip" install --quiet fastapi uvicorn python-multipart
fi

# Check if a server is already running on this port and kill it
echo "Checking for existing server on port $PORT..."
EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null)
if [ ! -z "$EXISTING_PID" ]; then
    echo "Found existing server with PID $EXISTING_PID, killing it..."
    kill $EXISTING_PID 2>/dev/null
    sleep 1
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Start the FastAPI server
echo ""
echo "Starting Whisper server on http://localhost:$PORT"
echo "Open http://localhost:$PORT/lingo.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

"$PYTHON" -m uvicorn whisper_server:app --host 0.0.0.0 --port $PORT --reload
