#!/usr/bin/env python3
"""
Voice Typer - GTK4 application for system-wide voice-to-text input.

Listens to microphone, detects speech, transcribes via whisper.cpp,
and types the result wherever the cursor is focused.

Usage:
  python3 voice_typer.py

Close by clicking the button or pressing Escape.
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gdk, GLib

import numpy as np
import sounddevice as sd
import threading
import subprocess
import tempfile
import wave
import time
import os
import logging
from pathlib import Path

# =============================================================================
# Logging Setup
# =============================================================================

SCRIPT_DIR_FOR_LOG = Path(__file__).parent.absolute()
LOG_FILE = SCRIPT_DIR_FOR_LOG / "voice_typer.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),  # Overwrite each run
        logging.StreamHandler()  # Also print to console
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
WHISPER_DIR = SCRIPT_DIR.parent / "whisper-model"
WHISPER_BINARY = WHISPER_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = WHISPER_DIR / "whisper.cpp" / "models" / "ggml-base.en.bin"
WHISPER_LIB_DIR = WHISPER_DIR / "whisper.cpp" / "build" / "src"  # For libwhisper.so
GGML_LIB_DIR = WHISPER_DIR / "whisper.cpp" / "build" / "ggml" / "src"  # For libggml.so

# Audio settings
# Recording at 48kHz (common USB mic rate), will resample to 16kHz for whisper
RECORD_SAMPLE_RATE = 48000  # Rate to record from microphone
WHISPER_SAMPLE_RATE = 16000  # Rate whisper.cpp expects
CHANNELS = 1
DTYPE = np.int16

# Audio device - None for default, or specify device name/index
# Run: python3 -c "import sounddevice as sd; print(sd.query_devices())" to list devices
# Set to None to use system default, or a string to match device name
AUDIO_DEVICE = "Shure"  # Shure MV5C USB microphone

# Silence detection
# Lower threshold for quieter mics (0.005-0.01), higher for louder mics (0.01-0.02)
SILENCE_THRESHOLD = 0.01  # RMS level below this is silence
SILENCE_DURATION_S = 1.0  # Seconds of silence before transcription
MIN_AUDIO_DURATION_S = 0.5  # Minimum audio length to process

# =============================================================================
# Audio Processing
# =============================================================================

class AudioRecorder:
    """Handles continuous audio recording with silence detection."""
    
    def __init__(self, on_speech_detected):
        self.on_speech_detected = on_speech_detected
        self.is_running = False
        self.audio_buffer = []
        self.silence_start_time = None
        self.recording_start_time = None
        self.speech_detected = False  # Track if we've heard speech above threshold
        self.stream = None
        self.lock = threading.Lock()
        self.rms_log_counter = 0  # To avoid logging every single callback
    
    def start(self):
        """Start recording from microphone."""
        self.is_running = True
        self.audio_buffer = []
        self.silence_start_time = None
        self.recording_start_time = time.time()
        self.speech_detected = False
        
        # Find the audio device
        device = None
        if AUDIO_DEVICE:
            devices = sd.query_devices()
            log.debug(f"Searching for device containing '{AUDIO_DEVICE}' among {len(devices)} devices")
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    log.debug(f"  Input device [{i}]: {d['name']}")
                if AUDIO_DEVICE.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                    device = i
                    log.info(f"Found audio device: [{i}] {d['name']}")
                    break
            if device is None:
                log.warning(f"Device containing '{AUDIO_DEVICE}' not found, using default")
        
        log.info(f"Starting audio stream: {RECORD_SAMPLE_RATE}Hz, {CHANNELS} channel(s), device={device}")
        
        self.stream = sd.InputStream(
            samplerate=RECORD_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._audio_callback,
            blocksize=1024,
            device=device
        )
        self.stream.start()
        log.info("Audio stream started successfully")
    
    def stop(self):
        """Stop recording."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Called for each audio chunk from the microphone."""
        if status:
            log.warning(f"Audio callback status: {status}")
        
        if not self.is_running:
            return
        
        # Calculate RMS (root mean square) for volume level
        # Normalize int16 to float for RMS calculation
        audio_float = indata.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Log RMS every ~50 callbacks (roughly once per second at 1024 blocksize @ 16kHz)
        self.rms_log_counter += 1
        if self.rms_log_counter >= 15:
            log.debug(f"Audio RMS: {rms:.6f} (threshold: {SILENCE_THRESHOLD}), buffer chunks: {len(self.audio_buffer)}, speech_detected: {self.speech_detected}")
            self.rms_log_counter = 0
        
        with self.lock:
            # Always accumulate audio
            self.audio_buffer.append(indata.copy())
            
            now = time.time()
            
            if rms < SILENCE_THRESHOLD:
                # Below threshold - might be silence
                if self.silence_start_time is None:
                    self.silence_start_time = now
                elif now - self.silence_start_time >= SILENCE_DURATION_S:
                    # We've had enough silence - process if we have audio AND speech was detected
                    audio_duration = now - self.recording_start_time
                    log.info(f"Silence detected! Buffer has {len(self.audio_buffer)} chunks, audio_duration={audio_duration:.2f}s, speech_detected={self.speech_detected}")
                    
                    if len(self.audio_buffer) > 0 and audio_duration >= MIN_AUDIO_DURATION_S and self.speech_detected:
                        # Grab the audio and reset
                        audio_data = np.concatenate(self.audio_buffer)
                        log.info(f"Triggering transcription with {len(audio_data)} samples ({len(audio_data)/RECORD_SAMPLE_RATE:.2f}s)")
                        self.audio_buffer = []
                        self.recording_start_time = now
                        self.silence_start_time = None
                        self.speech_detected = False  # Reset for next utterance
                        
                        # Trigger transcription in separate thread
                        threading.Thread(
                            target=self._process_audio,
                            args=(audio_data,),
                            daemon=True
                        ).start()
                    else:
                        # Not enough audio or no speech detected, just reset
                        if not self.speech_detected:
                            log.debug("No speech detected in buffer, discarding silence")
                        else:
                            log.debug(f"Not enough audio to process (duration={audio_duration:.2f}s < {MIN_AUDIO_DURATION_S}s)")
                        self.audio_buffer = []
                        self.recording_start_time = now
                        self.silence_start_time = None
                        self.speech_detected = False
            else:
                # Above threshold - this is speech!
                self.silence_start_time = None
                self.speech_detected = True
    
    def _process_audio(self, audio_data):
        """Process recorded audio through whisper and type result."""
        log.info(f"_process_audio called with {len(audio_data)} samples")
        try:
            text = transcribe_audio(audio_data)
            log.info(f"Transcription result: '{text}'")
            if text and text.strip():
                log.info(f"Scheduling typing of: '{text.strip()}'")
                # Schedule callback on main thread
                GLib.idle_add(self.on_speech_detected, text.strip())
            else:
                log.warning("Transcription returned empty text")
        except Exception as e:
            log.error(f"Transcription error: {e}", exc_info=True)


def normalize_audio(audio_data):
    """
    Normalize audio to use more of the int16 dynamic range.
    This helps whisper recognize quieter audio.
    
    Args:
        audio_data: numpy array of int16 samples
    
    Returns:
        Normalized numpy array of int16 samples
    """
    # Convert to float for processing
    audio_float = audio_data.astype(np.float32)
    
    # Find the peak amplitude
    peak = max(abs(audio_float.min()), abs(audio_float.max()))
    
    if peak < 100:
        # Audio is basically silent, don't amplify noise
        log.debug(f"Audio too quiet to normalize (peak={peak})")
        return audio_data
    
    # Target peak at 80% of int16 max to avoid clipping
    target_peak = 32767 * 0.8
    gain = target_peak / peak
    
    # Limit gain to avoid amplifying quiet audio too much
    gain = min(gain, 20.0)  # Max 20x amplification
    
    log.debug(f"Normalizing audio: peak={peak:.0f}, gain={gain:.2f}x")
    
    normalized = audio_float * gain
    
    # Clip to int16 range and convert back
    normalized = np.clip(normalized, -32768, 32767)
    return normalized.astype(np.int16)


def resample_audio(audio_data, orig_rate, target_rate):
    """
    Resample audio from orig_rate to target_rate using linear interpolation.
    
    Args:
        audio_data: numpy array of int16 samples (can be 1D or 2D)
        orig_rate: original sample rate (e.g., 48000)
        target_rate: target sample rate (e.g., 16000)
    
    Returns:
        Resampled numpy array of int16 samples (1D)
    """
    # Flatten to 1D if needed (sounddevice returns (N, channels) shape)
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
    
    # Debug: log audio data statistics
    log.debug(f"Audio data before resample: dtype={audio_data.dtype}, shape={audio_data.shape}, "
              f"min={audio_data.min()}, max={audio_data.max()}, mean={audio_data.mean():.2f}")
    
    if orig_rate == target_rate:
        return audio_data
    
    # Calculate the number of samples in the output
    duration = len(audio_data) / orig_rate
    num_samples = int(duration * target_rate)
    
    # Create interpolation indices
    indices = np.linspace(0, len(audio_data) - 1, num_samples)
    
    # Interpolate (convert to float for interpolation, then back to int16)
    resampled = np.interp(indices, np.arange(len(audio_data)), audio_data.astype(np.float32))
    
    # Debug: log resampled data statistics  
    log.debug(f"Audio data after resample: min={resampled.min():.2f}, max={resampled.max():.2f}")
    
    return resampled.astype(np.int16)


def transcribe_audio(audio_data):
    """
    Transcribe audio using whisper.cpp.
    
    Args:
        audio_data: numpy array of int16 audio samples at RECORD_SAMPLE_RATE
    
    Returns:
        Transcribed text string
    """
    log.info(f"transcribe_audio called with {len(audio_data)} samples at {RECORD_SAMPLE_RATE}Hz")
    
    if not WHISPER_BINARY.exists():
        log.error(f"Whisper binary not found at {WHISPER_BINARY}")
        return None
    
    if not WHISPER_MODEL.exists():
        log.error(f"Whisper model not found at {WHISPER_MODEL}")
        return None
    
    # Resample from recording rate to whisper rate (48kHz -> 16kHz)
    if RECORD_SAMPLE_RATE != WHISPER_SAMPLE_RATE:
        log.debug(f"Resampling from {RECORD_SAMPLE_RATE}Hz to {WHISPER_SAMPLE_RATE}Hz")
        audio_data = resample_audio(audio_data, RECORD_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        log.debug(f"Resampled to {len(audio_data)} samples")
    
    # Normalize audio to use more dynamic range (helps whisper with quiet audio)
    audio_data = normalize_audio(audio_data)
    
    # Write audio to temporary WAV file (keep for debugging)
    wav_path = f"/tmp/voice_typer_debug.wav"
    
    log.debug(f"Writing audio to file: {wav_path}")
    
    try:
        # Write WAV file at whisper's expected rate
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(WHISPER_SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        
        log.info(f"WAV file written: {wav_path} ({os.path.getsize(wav_path)} bytes)")
        
        # Run whisper.cpp with library path set
        env = os.environ.copy()
        lib_paths = f"{WHISPER_LIB_DIR}:{GGML_LIB_DIR}"
        env["LD_LIBRARY_PATH"] = lib_paths + ":" + env.get("LD_LIBRARY_PATH", "")
        
        cmd = [
            str(WHISPER_BINARY),
            "-m", str(WHISPER_MODEL),
            "-f", wav_path,
            "--no-timestamps",
            "--language", "en",
            "--threads", "4",
        ]
        log.debug(f"Running whisper command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        log.debug(f"Whisper return code: {result.returncode}")
        log.debug(f"Whisper stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
        log.debug(f"Whisper stderr: {result.stderr[:500] if result.stderr else '(empty)'}")
        
        if result.returncode != 0:
            log.error(f"Whisper error (code {result.returncode}): {result.stderr}")
            return None
        
        # Parse output - clean up whisper metadata
        lines = result.stdout.strip().split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip whisper metadata lines and blank audio markers
            if line.startswith("[") or line.startswith("whisper_"):
                continue
            clean_lines.append(line)
        
        text = " ".join(clean_lines)
        
        # Filter out common whisper hallucinations (appears when audio is mostly silence/noise)
        hallucinations = [
            "(dramatic music)",
            "(upbeat music)", 
            "(music)",
            "(music playing)",
            "(soft music)",
            "(laughing)",
            "(applause)",
            "(silence)",
            "(sighs)",
            "(coughing)",
            "(breathing)",
            "[BLANK_AUDIO]",
            "[silence]",
            "Thank you.",
            "Thanks for watching!",
            "Thanks for watching.",
            "Subscribe to my channel",
            "Please subscribe",
        ]
        
        text_lower = text.lower().strip()
        for hallucination in hallucinations:
            if text_lower == hallucination.lower():
                log.warning(f"Filtered out whisper hallucination: '{text}'")
                return None
        
        return text
    
    except Exception as e:
        log.error(f"Transcription exception: {e}", exc_info=True)
        return None


# =============================================================================
# Keyboard Injection
# =============================================================================

def type_text(text):
    """
    Handle transcribed text.
    
    TODO: Implement clipboard paste functionality.
    For now, just logs the transcribed text.
    """
    log.info(f"TRANSCRIBED: '{text}'")
    print(f"\n🎯 TRANSCRIBED: {text}\n")


# =============================================================================
# GTK4 Application
# =============================================================================

class VoiceTyperWindow(Gtk.ApplicationWindow):
    """Simple dialog window for voice typing control."""
    
    def __init__(self, app):
        super().__init__(application=app)
        
        self.recorder = None
        self.is_recording = False
        
        # Window setup - small dialog with title bar (draggable)
        self.set_title("Voice Typer")
        self.set_default_size(150, -1)  # Minimal width, auto height
        self.set_resizable(False)
        
        # Create main vertical box
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(15)
        vbox.set_margin_end(15)
        
        # Create "Mic" checkbox
        self.mic_checkbox = Gtk.CheckButton(label="Mic")
        self.mic_checkbox.connect("toggled", self.on_mic_toggled)
        vbox.append(self.mic_checkbox)
        
        # Add CSS for styling
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window {
                background: @theme_bg_color;
            }
            checkbutton {
                font-size: 14px;
            }
            checkbutton:checked label {
                color: #e53935;
                font-weight: bold;
            }
        """)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        
        self.set_child(vbox)
        
        # Handle window close
        self.connect("close-request", self.on_close_request)
    
    def on_mic_toggled(self, checkbox):
        """Handle mic checkbox toggle."""
        if checkbox.get_active():
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.recorder = AudioRecorder(self.on_speech_detected)
        self.recorder.start()
        log.info("Recording started...")
        print("🎤 Microphone ON - listening...")
    
    def stop_recording(self):
        """Stop audio recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        log.info("Recording stopped.")
        print("🔇 Microphone OFF")
    
    def on_speech_detected(self, text):
        """Called when speech is transcribed."""
        log.info(f"on_speech_detected callback called with: '{text}'")
        type_text(text)
        return False  # Don't repeat
    
    def on_close_request(self, window):
        """Clean up when window closes."""
        self.stop_recording()
        return False  # Allow close


class VoiceTyperApp(Gtk.Application):
    """GTK Application wrapper."""
    
    def __init__(self):
        super().__init__(application_id="com.lingo.voicetyper")
    
    def do_activate(self):
        """Create and show the main window."""
        win = VoiceTyperWindow(self)
        win.present()


# =============================================================================
# Entry Point
# =============================================================================

def check_dependencies():
    """Check if required dependencies are available."""
    errors = []
    
    if not WHISPER_BINARY.exists():
        errors.append(f"whisper-cli not found at {WHISPER_BINARY}")
    
    if not WHISPER_MODEL.exists():
        errors.append(f"whisper model not found at {WHISPER_MODEL}")
    
    # Check for audio devices
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            errors.append("No audio input devices found")
    except Exception as e:
        errors.append(f"Error querying audio devices: {e}")
    
    return errors


def main():
    """Main entry point."""
    print("Voice Typer - Starting...")
    print(f"Whisper binary: {WHISPER_BINARY}")
    print(f"Whisper model: {WHISPER_MODEL}")
    
    # Check dependencies
    errors = check_dependencies()
    if errors:
        print("\n⚠️  Missing dependencies:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease run setup-whisper.sh from the project root first.")
        return 1
    
    print("✅ All dependencies found. Starting application...")
    
    # Run the GTK application
    app = VoiceTyperApp()
    return app.run(None)


if __name__ == "__main__":
    exit(main())
