#!/bin/bash
# Start Voice Typer GTK application via uv.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# GTK4 Python bindings (gi) come from system packages, so the venv must be
# created with --system-site-packages. uv will reuse an existing .venv.
if [ ! -d .venv ]; then
    uv venv --system-site-packages
fi

uv sync --active
exec uv run --active voice_typer.py
