#!/usr/bin/env bash
# Lingo Voice Typer launcher.
#
# uv builds/refreshes the virtualenv from pyproject.toml on every run, so there
# is no install step. Unlike the GTK version this needs no --system-site-packages
# venv: PyQt6 comes from PyPI, whereas PyGObject had to come from the distro.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but was not found in PATH." >&2
  echo "Install it from https://docs.astral.sh/uv/ or via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

exec uv run --directory "${HERE}" python -m voice_typer "$@"
