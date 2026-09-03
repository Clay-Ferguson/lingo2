"""Entry point: python -m voice_typer

Close the window with the title bar close button.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import windowchrome
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from . import APP_ID, APP_NAME, LINGO_THEME
from .transcribe import WHISPER_BINARY, WHISPER_MODEL, cleanup_temp_audio_files

# =============================================================================
# Logging Setup
# =============================================================================

# The log lives beside the package rather than inside it, keeping it at
# qt-app/voice_typer.log where the docs say it is.
APP_DIR = Path(__file__).parent.parent.absolute()
LOG_FILE = APP_DIR / "voice_typer.log"
ICON = APP_DIR / "lingo-logo.png"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),  # Overwrite each run
        logging.StreamHandler(),  # Also print to console
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# Startup Checks
# =============================================================================


def check_dependencies() -> list[str]:
    """Return a list of human-readable problems, empty if everything is present."""
    errors = []

    if not WHISPER_BINARY.exists():
        errors.append(f"whisper-cli not found at {WHISPER_BINARY}")

    if not WHISPER_MODEL.exists():
        errors.append(f"whisper model not found at {WHISPER_MODEL}")

    # Imported here, not at module scope: sounddevice raises at import time
    # when PortAudio is missing, and that is precisely one of the problems
    # this function exists to report.
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if not input_devices:
            errors.append("No audio input devices found")
    except OSError as e:
        errors.append(f"PortAudio not available ({e}). Run ./setup.sh to install it.")
    except Exception as e:
        errors.append(f"Error querying audio devices: {e}")

    return errors


def main() -> int:
    print(f"{APP_NAME} Voice Typer - Starting...")
    print(f"Whisper binary: {WHISPER_BINARY}")
    print(f"Whisper model: {WHISPER_MODEL}")

    errors = check_dependencies()
    if errors:
        print("\n⚠️  Missing dependencies:")
        for error in errors:
            print(f"   - {error}")
        print("\nRun ./setup.sh for system packages, and setup-whisper.sh from the\nproject root to build whisper and download the model.")
        return 1

    print("✅ All dependencies found. Starting application...")

    # Imported only once the checks pass, so a missing audio stack produces the
    # message above rather than an import traceback.
    from .window import VoiceTyperWindow

    # Clear anything a previous run left in /dev/shm
    cleanup_temp_audio_files()

    # Before the QApplication, and it has to be: `configure()` picks the Wayland
    # decoration plugin through an environment variable that the platform plugin
    # reads during that constructor and never again. See windowchrome's README.
    windowchrome.configure(LINGO_THEME)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    # Ties the window to com.lingo.voicetyper.desktop so the dock and alt-tab
    # show our icon. This is what replaces GTK's Gtk.Application(application_id=…);
    # without it the Wayland app_id comes from argv[0] ("python3") and matches
    # nothing.
    app.setDesktopFileName(APP_ID)
    if ICON.is_file():
        app.setWindowIcon(QIcon(str(ICON)))

    # After the QApplication, and after anything that changes the application
    # palette: `install()` captures the body's own surface and text colors at
    # the moment it runs, then hands the title bar those palette roles. Lingo
    # tunes no palette of its own, so here is as late as it gets — but a palette
    # changed after this line is one it never saw, and the settings dialog's
    # muted help color is derived from one of the roles it takes.
    windowchrome.install(app)

    window = VoiceTyperWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
