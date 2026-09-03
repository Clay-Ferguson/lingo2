"""Lingo Voice Typer: system-wide voice-to-text input for Linux.

Listens to the microphone, transcribes speech with whisper.cpp, and injects
the result as keystrokes via the XDG Remote Desktop Portal, so the text lands
in whatever application currently has keyboard focus.

Module layout deliberately keeps the Qt-free logic separate from the GUI:
`config` and `transcribe` import no Qt at all and can be exercised without a
display; `audio` and `keyboard` are QObjects only because they need signals
and D-Bus; `window` is the only module that knows about widgets.
"""

from windowchrome import ChromeTheme

APP_NAME = "Lingo"

# Matches the .desktop filename and its StartupWMClass. Passed to
# QGuiApplication.setDesktopFileName() so Wayland can tie the window to the
# desktop entry and show our icon rather than a generic one.
APP_ID = "com.lingo.voicetyper"

# The window is a small always-visible control, glanced at rather than read,
# so its controls run well above the desktop default point size.
UI_FONT_PX = 20

# The window's title bar, and with it the thin frame the decoration draws down
# the sides and along the bottom. `windowchrome` owns both — see
# `windowchrome/README.md` (a sibling of the `lingo2` directory, not of this
# one) for why they are reachable at all (Wayland only, by repurposing three
# palette roles) and why their *size* is not.
#
# The library ships neutral defaults; this is Lingo's override of them, and it
# deliberately matches the other apps here so they read as one family rather
# than as unrelated windows. Note it is *not* one of the phase colors: those
# report what the pipeline is doing and paint the window itself (see
# `window.build_stylesheet`), while the title bar stays one color throughout.
LINGO_THEME = ChromeTheme(title_bg="#1369da")
