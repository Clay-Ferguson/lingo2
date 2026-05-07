#!/bin/bash
# Start the Lingo Whisper FastAPI server via uv.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PORT=8009

EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "Killing existing server on port $PORT (PID $EXISTING_PID)..."
    kill $EXISTING_PID 2>/dev/null || true
    sleep 1
fi

uv sync
exec uv run uvicorn whisper_server:app --host 0.0.0.0 --port $PORT --reload
