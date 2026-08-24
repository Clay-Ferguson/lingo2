"""The single control window: a mic toggle, a Close button, and settings.

A faithful port of the GTK4 layout, including the feature that makes the app
usable without reading anything -- the whole window changes color to report
what the pipeline is doing (green = hearing you, orange = transcribing,
white = typing). Those four colors are semantic signals rather than theming,
so unlike the rest of the styling they are fixed hex values.
"""

from __future__ import annotations

import logging
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QHBoxLayout,
                             QLabel, QLayout, QLineEdit, QMessageBox,
                             QPushButton, QToolButton, QVBoxLayout, QWidget)

from . import APP_NAME, UI_FONT_PX
from .audio import AudioRecorder
from .config import (DEFAULT_MIC_AUTO_OFF_TIMEOUT_S, DEFAULT_SILENCE_THRESHOLD,
                     get_input_devices, load_config, save_config)
from .keyboard import close_keyboard_injector, get_keyboard_injector, type_text
from .transcribe import cleanup_temp_audio_files

log = logging.getLogger(__name__)

# =============================================================================
# Styling
# =============================================================================

PHASE_COLORS = {
    "speech-detected": "#27AE60",
    "transcribing": "#E67E22",
    "typing": "#FFFFFF",
}

# Width budget for the wrapped help paragraphs, and therefore the width of the
# expanded window. See _help_label for why this is fixed rather than maximum.
HELP_TEXT_WIDTH = 340

THRESHOLD_HELP = (
    "Typical range is 0.001–0.02: lower values make the mic more sensitive; "
    "raise it to ignore background noise. Set the value as low as possible for "
    "best reliablility. If you have any problems with this application try "
    "lowering this threshold value."
)

AUTO_OFF_HELP = (
    "Automatically turn off the microphone after this many seconds without a "
    "successful transcription. Set to 0 to disable auto-off."
)


def build_stylesheet() -> str:
    """QSS for the window.

    The phase backgrounds hang off a dynamic `phase` property rather than
    separate stylesheets, so switching phase is a property set plus a repolish.
    The 'typing' phase forces text black because its background is white, and
    Qt does not repaint child text just because the ancestor's background
    changed -- each child type has to be named.
    """
    muted = QApplication.palette().color(
        QApplication.palette().ColorRole.WindowText
    )
    muted.setAlpha(180)

    rules = [
        f"QCheckBox {{ font-size: {UI_FONT_PX}px; }}",
        "QCheckBox::indicator { width: 24px; height: 24px; }",
        "QCheckBox:checked { color: #e53935; font-weight: bold; }",
        f"QComboBox {{ font-size: {UI_FONT_PX}px; }}",
        "QPushButton#closeButton { min-width: 60px; min-height: 36px;"
        " padding: 6px 16px; font-size: 16px; }",
        f"QLabel#helpText {{ font-size: 14px; color: rgba({muted.red()},"
        f" {muted.green()}, {muted.blue()}, 0.7); }}",
    ]

    for phase, color in PHASE_COLORS.items():
        rules.append(f'QWidget#voiceTyperWindow[phase="{phase}"] {{ background: {color}; }}')

    # White background needs dark text on every child that draws any.
    rules.append(
        'QWidget#voiceTyperWindow[phase="typing"] QLabel,'
        ' QWidget#voiceTyperWindow[phase="typing"] QCheckBox,'
        ' QWidget#voiceTyperWindow[phase="typing"] QToolButton'
        " { color: #000000; }"
    )
    return "\n".join(rules)


# =============================================================================
# Window
# =============================================================================


class VoiceTyperWindow(QWidget):
    """Small always-on-top-of-mind control panel for dictation."""

    def __init__(self) -> None:
        super().__init__()

        self.recorder: AudioRecorder | None = None
        self.is_recording = False
        self.config = load_config()
        self._active_phase: str | None = None

        # Auto-off timer state
        self._last_audio_detection_time: float | None = None

        self.setObjectName("voiceTyperWindow")
        # Without this a plain QWidget ignores stylesheet backgrounds entirely,
        # and every phase color silently does nothing.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle(APP_NAME)
        self.setStyleSheet(build_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(10)
        layout.addLayout(self._build_top_row())
        layout.addWidget(self._build_settings_toggle())
        layout.addWidget(self._build_settings_panel())

        # Reproduces GTK's set_resizable(False) with a -1 height: the window is
        # exactly as tall as its contents, and shrinks again when Settings
        # collapses.
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        self._auto_off_timer = QTimer(self)
        self._auto_off_timer.setInterval(1000)
        self._auto_off_timer.timeout.connect(self._check_auto_off)

        QTimer.singleShot(0, self._auto_start_microphone)

    # -- construction ------------------------------------------------------

    def _build_top_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.mic_checkbox = QCheckBox("Mic")
        self.mic_checkbox.toggled.connect(self.on_mic_toggled)
        row.addWidget(self.mic_checkbox)

        row.addStretch(1)

        close_button = QPushButton("Close")
        close_button.setObjectName("closeButton")
        close_button.clicked.connect(self.close)
        row.addWidget(close_button)
        return row

    def _build_settings_toggle(self) -> QToolButton:
        """Qt has no Gtk.Expander, so this is the usual arrow-button stand-in."""
        self.settings_toggle = QToolButton()
        self.settings_toggle.setText("Settings")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setChecked(False)
        self.settings_toggle.setAutoRaise(True)
        self.settings_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.settings_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.settings_toggle.toggled.connect(self._on_settings_toggled)
        return self.settings_toggle

    def _build_settings_panel(self) -> QWidget:
        self.settings_panel = QWidget()
        panel = QVBoxLayout(self.settings_panel)
        panel.setContentsMargins(4, 6, 4, 4)
        panel.setSpacing(6)

        # Microphone device dropdown
        self.mic_dropdown = QComboBox()
        self._populate_mic_dropdown()
        self.mic_dropdown.currentIndexChanged.connect(self.on_mic_changed)
        panel.addLayout(self._labeled_row("Input device", self.mic_dropdown))

        # Silence threshold
        self.threshold_entry = QLineEdit()
        self.threshold_entry.setMaxLength(8)
        self.threshold_entry.setFixedWidth(80)
        self.threshold_entry.setToolTip(
            "Lower values detect quieter audio; higher values filter background noise."
        )
        self.threshold_entry.setText(self._format_threshold(self._get_silence_threshold()))
        # editingFinished covers Enter *and* focus-out, which GTK needed two
        # separate handlers for.
        self.threshold_entry.editingFinished.connect(self._commit_threshold_entry)
        panel.addLayout(self._labeled_row("Silence threshold", self.threshold_entry))
        panel.addWidget(self._help_label(THRESHOLD_HELP))

        # Auto-off timeout
        self.auto_off_entry = QLineEdit()
        self.auto_off_entry.setMaxLength(6)
        self.auto_off_entry.setFixedWidth(80)
        self.auto_off_entry.setToolTip(
            "Seconds of inactivity before the microphone turns off automatically. 0 to disable."
        )
        self.auto_off_entry.setText(str(self._get_auto_off_timeout()))
        self.auto_off_entry.editingFinished.connect(self._commit_auto_off_entry)
        panel.addLayout(self._labeled_row("Auto-off timeout (s)", self.auto_off_entry))
        panel.addWidget(self._help_label(AUTO_OFF_HELP))

        self.settings_panel.setVisible(False)
        return self.settings_panel

    def _labeled_row(self, text: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(text)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(widget)
        return row

    def _help_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("helpText")
        label.setWordWrap(True)
        # Fixed, not maximum. A wrapped label's height is a function of its
        # width (hasHeightForWidth() is True), so with only a maximum the
        # window's sizeHint depends on how the layout happens to negotiate
        # width. Pinning the width makes the required height deterministic,
        # which is what keeps the window from settling at the wrong height.
        label.setFixedWidth(HELP_TEXT_WIDTH)
        return label

    def _on_settings_toggled(self, expanded: bool) -> None:
        self.settings_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.settings_panel.setVisible(expanded)
        # Resize here rather than leaving it to the layout request Qt posts,
        # so no compositor event can be processed while the window size and
        # the panel disagree.
        layout = self.layout()
        layout.invalidate()
        layout.activate()
        self._enforce_size()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._enforce_size()

    def _enforce_size(self) -> None:
        """Undo any resize the window's own size constraints forbid.

        Wayland compositors reply to a state change -- losing or regaining
        focus, most visibly -- with a configure event carrying a size. Mutter
        sends the size it last configured us at, and that size is stale: when
        Settings expands we resize ourselves, which mutter renders but does
        not record. So the first click on another window arrives with a
        configure for the collapsed size.

        Qt applies that size verbatim instead of clamping it to the min/max
        the window advertised, and the result is the whole reported symptom at
        once: the window shrinks back to collapsed height, the settings panel
        is clipped out of view even though the arrow still points down, and
        the rows above it are squeezed until the Settings button overlaps
        Close. X11 does not do this, which is why it only shows up on Wayland.

        SetFixedSize keeps minimumSize() == maximumSize() == the size the
        layout needs, so any other size can only have come from the
        compositor. Resizing back also teaches mutter the real size, so this
        corrects once per Settings toggle and then stays quiet.

        This cannot use setFixedSize(): it returns early when neither the
        minimum nor the maximum changes, which is exactly this case, and that
        is why re-asserting the constraints alone never fixed it.
        """
        required = self.minimumSize()
        if required != self.maximumSize() or self.size() == required:
            return
        log.debug(
            f"Rejecting {self.size().width()}x{self.size().height()} window size; "
            f"restoring {required.width()}x{required.height()}"
        )
        self.resize(required)

    # -- phase display -----------------------------------------------------

    def on_processing_phase_changed(self, phase: str) -> None:
        """Repaint the window to match the pipeline's current phase."""
        target = phase if phase in PHASE_COLORS else None
        if self._active_phase == target:
            return
        self._active_phase = target

        self.setProperty("phase", target or "")
        # Qt only re-evaluates property-based selectors on a repolish.
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
        log.debug(f"Processing phase active: {phase}")

    # -- settings: silence threshold --------------------------------------

    def _get_silence_threshold(self) -> float:
        return float(self.config.get("silence_threshold", DEFAULT_SILENCE_THRESHOLD))

    def _format_threshold(self, value: float) -> str:
        formatted = f"{float(value):.4f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"

    def _commit_threshold_entry(self) -> None:
        text = self.threshold_entry.text().strip()
        if not text:
            self.threshold_entry.setText(self._format_threshold(self._get_silence_threshold()))
            return
        try:
            value = float(text)
        except ValueError:
            print("⚠️  Silence threshold must be a number.")
            self.threshold_entry.setText(self._format_threshold(self._get_silence_threshold()))
            return
        if value <= 0:
            print("⚠️  Silence threshold must be positive.")
            self.threshold_entry.setText(self._format_threshold(self._get_silence_threshold()))
            return
        self._update_silence_threshold(value)

    def _update_silence_threshold(self, value: float) -> None:
        value = float(value)
        if abs(self._get_silence_threshold() - value) < 1e-6:
            self.threshold_entry.setText(self._format_threshold(value))
            return
        self.config["silence_threshold"] = value
        save_config(self.config)
        if self.recorder:
            self.recorder.set_silence_threshold(value)
        self.threshold_entry.setText(self._format_threshold(value))
        print(f"🎚️ Silence threshold set to {self._format_threshold(value)}")

    # -- settings: auto-off timeout ---------------------------------------

    def _get_auto_off_timeout(self) -> int:
        return int(self.config.get("mic_auto_off_timeout", DEFAULT_MIC_AUTO_OFF_TIMEOUT_S))

    def _commit_auto_off_entry(self) -> None:
        text = self.auto_off_entry.text().strip()
        if not text:
            self.auto_off_entry.setText(str(self._get_auto_off_timeout()))
            return
        try:
            value = int(text)
        except ValueError:
            print("⚠️  Auto-off timeout must be an integer.")
            self.auto_off_entry.setText(str(self._get_auto_off_timeout()))
            return
        if value < 0:
            print("⚠️  Auto-off timeout must be 0 or positive.")
            self.auto_off_entry.setText(str(self._get_auto_off_timeout()))
            return
        self._update_auto_off_timeout(value)

    def _update_auto_off_timeout(self, value: int) -> None:
        value = int(value)
        if self._get_auto_off_timeout() == value:
            self.auto_off_entry.setText(str(value))
            return
        self.config["mic_auto_off_timeout"] = value
        save_config(self.config)
        self.auto_off_entry.setText(str(value))
        if value == 0:
            print("⏱️  Mic auto-off disabled")
        else:
            print(f"⏱️  Mic auto-off timeout set to {value}s")

    # -- microphone --------------------------------------------------------

    def _auto_start_microphone(self) -> None:
        """Turn the mic on at launch, showing orange while it initializes."""
        self.on_processing_phase_changed("transcribing")
        # Let the orange actually paint before the blocking device open.
        QTimer.singleShot(100, self._complete_auto_start)

    def _complete_auto_start(self) -> None:
        self.mic_checkbox.setChecked(True)  # triggers on_mic_toggled
        self.on_processing_phase_changed("idle")

    def _populate_mic_dropdown(self) -> None:
        """Fill the dropdown, selecting the saved device if it is still present."""
        self.mic_dropdown.addItem("System Default")

        # Rescan only at startup; it tears down PortAudio and is not free.
        self.input_devices = get_input_devices(rescan=True)
        for device in self.input_devices:
            self.mic_dropdown.addItem(device["name"])

        saved_device = self.config.get("audio_device")
        if saved_device is None:
            self.mic_dropdown.setCurrentIndex(0)
            return

        for i, device in enumerate(self.input_devices):
            if device["name"] == saved_device:
                self.mic_dropdown.setCurrentIndex(i + 1)  # +1 for "System Default"
                return

        log.warning(f"Saved device '{saved_device}' not found, using default")
        self.mic_dropdown.setCurrentIndex(0)

    def on_mic_changed(self, index: int) -> None:
        """Handle microphone selection change."""
        # The stream is already open on the old device, so stop first.
        if self.mic_checkbox.isChecked():
            self.mic_checkbox.setChecked(False)

        if index <= 0:
            self.config["audio_device"] = None
        else:
            self.config["audio_device"] = self.input_devices[index - 1]["name"]

        save_config(self.config)
        log.info(f"Microphone changed to: {self.config['audio_device'] or 'System Default'}")

    def on_mic_toggled(self, checked: bool) -> None:
        if checked:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        """Start audio recording."""
        if self.is_recording:
            return

        self.recorder = AudioRecorder(
            audio_device=self.config.get("audio_device"),
            silence_threshold=self._get_silence_threshold(),
        )
        self.recorder.phase_changed.connect(self.on_processing_phase_changed)
        self.recorder.speech_transcribed.connect(self.on_speech_transcribed)

        success, error_msg = self.recorder.start()
        if not success:
            self.recorder = None
            self._show_mic_error(error_msg or "Unknown microphone error.")
            self.mic_checkbox.setChecked(False)
            return

        self.is_recording = True
        print("🎤 Microphone ON - listening...")

        self._start_auto_off_timer()

        # First use triggers the portal permission dialog, so get it out of
        # the way now rather than mid-sentence.
        injector = get_keyboard_injector()
        injector.initialize(callback=self._on_keyboard_ready)

    def stop_recording(self) -> None:
        """Stop audio recording."""
        if not self.is_recording:
            return

        self.is_recording = False
        if self.recorder:
            self.recorder.stop()
            self.recorder = None

        self._auto_off_timer.stop()
        print("🔇 Microphone OFF")

    def _show_mic_error(self, message: str) -> None:
        QMessageBox.warning(self, "Microphone Error", message)

    def _on_keyboard_ready(self, success: bool) -> None:
        if success:
            print("⌨️  Keyboard access granted - ready to type!")
        else:
            print("⚠️  Keyboard access denied - text will be logged but not typed")

    # -- transcription results --------------------------------------------

    def on_speech_transcribed(self, text: str) -> None:
        """Type a finished transcription wherever the cursor is."""
        # Marks the mic as actively in use, deferring auto-off.
        self._last_audio_detection_time = time.time()

        recorder = self.recorder

        def finished() -> None:
            if recorder:
                recorder.notify_typing_done()
            else:
                self.on_processing_phase_changed("idle")

        try:
            type_text(text, on_finished=finished)
        except Exception as e:
            log.error(f"Failed to type transcribed text: {e}", exc_info=True)
            finished()

    # -- auto-off inactivity timer ----------------------------------------

    def _start_auto_off_timer(self) -> None:
        self._last_audio_detection_time = time.time()  # reset baseline
        self._auto_off_timer.start()
        log.debug("Auto-off inactivity timer started")

    def _check_auto_off(self) -> None:
        """Disable the mic once it has gone unused for the configured time."""
        if not self.is_recording:
            self._auto_off_timer.stop()
            return

        timeout = self._get_auto_off_timeout()
        if timeout <= 0:
            return  # auto-off disabled; keep ticking in case it is re-enabled

        elapsed = time.time() - (self._last_audio_detection_time or 0)
        if elapsed >= timeout:
            log.info(f"Mic auto-off: no transcription for {elapsed:.0f}s (threshold {timeout}s)")
            print(f"⏱️  Mic auto-off after {timeout}s of inactivity")
            self._auto_off_timer.stop()
            self.mic_checkbox.setChecked(False)  # triggers stop_recording

    # -- teardown ----------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Clean up when the window closes."""
        self.stop_recording()
        close_keyboard_injector()
        cleanup_temp_audio_files()
        super().closeEvent(event)
