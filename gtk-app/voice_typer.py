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
import wave
import time
import os
import logging
from pathlib import Path
import yaml

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

# RAM_CACHE: Use /dev/shm (tmpfs in RAM) instead of /tmp for audio files
# Setting this to True avoids disk I/O and reduces SSD wear
RAM_CACHE = True

# Audio settings
# Recording at 48kHz (common USB mic rate), will resample to 16kHz for whisper
RECORD_SAMPLE_RATE = 48000  # Rate to record from microphone
WHISPER_SAMPLE_RATE = 16000  # Rate whisper.cpp expects
CHANNELS = 1
DTYPE = np.int16

# Silence detection
# Lower threshold for quieter mics (0.001-0.005), higher for louder mics (0.01-0.02)
# Adjust based on your microphone - check the "Audio RMS" log values when speaking
DEFAULT_SILENCE_THRESHOLD = 0.005  # RMS level below this is silence (raised to filter electrical spikes)
SILENCE_DURATION_S = 1.0  # Seconds of silence before transcription
MIN_AUDIO_DURATION_S = 0.5  # Minimum audio length to process
SPEECH_CONFIRM_CHUNKS = 3  # Require this many consecutive above-threshold chunks to confirm speech

# =============================================================================
# Configuration Management
# =============================================================================

CONFIG_DIR = Path.home() / ".config"
CONFIG_FILE = CONFIG_DIR / "lingo-gtk.yaml"

DEFAULT_CONFIG = {
    "audio_device": None,  # None means system default
    "portal_restore_token": None,  # Saved token for XDG Remote Desktop Portal session persistence
    "silence_threshold": DEFAULT_SILENCE_THRESHOLD,
}

def load_config():
    """Load configuration from YAML file, creating defaults if needed."""
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f) or {}
        # Merge with defaults for any missing keys
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        try:
            config["silence_threshold"] = float(config.get("silence_threshold", DEFAULT_SILENCE_THRESHOLD))
        except (TypeError, ValueError):
            config["silence_threshold"] = DEFAULT_SILENCE_THRESHOLD
        return config
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save configuration to YAML file."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        log.info(f"Config saved to {CONFIG_FILE}")
    except Exception as e:
        log.error(f"Failed to save config: {e}")

def get_input_devices():
    """Get list of available audio input devices."""
    # Force PortAudio to re-scan devices (catches hot-plugged USB mics)
    sd._terminate()
    sd._initialize()
    
    devices = sd.query_devices()
    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            input_devices.append({
                'index': i,
                'name': d['name'],
            })
    return input_devices

# =============================================================================
# Audio Processing
# =============================================================================

class AudioRecorder:
    """Handles continuous audio recording with silence detection."""
    
    def __init__(self, on_speech_detected, audio_device=None, on_processing_phase_changed=None,
                 silence_threshold=DEFAULT_SILENCE_THRESHOLD):
        self.on_speech_detected = on_speech_detected
        self.on_processing_phase_changed = on_processing_phase_changed  # Callback for UI phase changes
        self.audio_device = audio_device  # Device name or None for default
        self.is_running = False
        self.audio_buffer = []
        self.silence_start_time = None
        self.recording_start_time = None
        self.speech_detected = False  # Track if we've heard speech above threshold
        self.speech_confirm_count = 0  # Counter for consecutive above-threshold chunks
        self.stream = None
        self.lock = threading.RLock()
        self.current_phase = 'idle'
        self.silence_threshold = float(silence_threshold)
    
    def start(self):
        """Start recording from microphone.
        
        Returns:
            tuple: (success: bool, error_message: str or None)
        """
        self.is_running = True
        self.audio_buffer = []
        self.silence_start_time = None
        self.recording_start_time = time.time()
        self.speech_detected = False
        self.current_phase = 'idle'
        
        # Find the audio device by name
        device = None
        device_not_found = False
        if self.audio_device:
            # Force PortAudio to re-scan devices
            sd._terminate()
            sd._initialize()
            
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['name'] == self.audio_device and d['max_input_channels'] > 0:
                    device = i
                    log.info(f"Using audio device: [{i}] {d['name']}")
                    break
            if device is None:
                log.warning(f"Device '{self.audio_device}' not found in available devices")
                device_not_found = True
                # List what we did find for debugging
                available = [d['name'] for d in devices if d['max_input_channels'] > 0]
                log.info(f"Available input devices: {available}")
                return (False, f"Microphone '{self.audio_device}' not found.\n\nIt may be in use by another application (like Chrome).\n\nTry closing other apps that use the microphone and try again.")
        
        log.info(f"Starting audio stream: {RECORD_SAMPLE_RATE}Hz, {CHANNELS} channel(s), device={device}")
        
        try:
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
            return (True, None)
        except sd.PortAudioError as e:
            error_msg = str(e)
            log.error(f"Failed to open audio stream: {error_msg}")
            self.is_running = False
            return (False, f"Failed to open microphone.\n\nError: {error_msg}\n\nThe device may be in use by another application.")
        except Exception as e:
            error_msg = str(e)
            log.error(f"Unexpected error opening audio stream: {error_msg}")
            self.is_running = False
            return (False, f"Unexpected error opening microphone:\n\n{error_msg}")
    
    def stop(self):
        """Stop recording."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._transition_to_phase('idle')

    def _transition_to_phase(self, phase):
        notify = False
        with self.lock:
            if self.current_phase != phase:
                self.current_phase = phase
                notify = True
        if notify and self.on_processing_phase_changed:
            GLib.idle_add(self.on_processing_phase_changed, phase)

    def set_silence_threshold(self, value):
        with self.lock:
            self.silence_threshold = float(value)
        log.info(f"Silence threshold updated to {self.silence_threshold:.6f}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Called for each audio chunk from the microphone."""
        if status:
            # log.warning(f"Audio callback status: {status}")
            pass
        
        if not self.is_running:
            return
        
        # Calculate RMS (root mean square) for volume level
        # Normalize int16 to float for RMS calculation
        audio_float = indata.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Track peak RMS for diagnostics
        if not hasattr(self, 'peak_rms'):
            self.peak_rms = 0.0
        self.peak_rms = max(self.peak_rms, rms)
        
        with self.lock:
            # Always accumulate audio
            self.audio_buffer.append(indata.copy())
            
            now = time.time()
            threshold = self.silence_threshold
            
            if rms < threshold:
                if self.current_phase == 'speech-detected' and not self.speech_detected:
                    self._transition_to_phase('idle')
                # Below threshold - might be silence
                if self.silence_start_time is None:
                    self.silence_start_time = now
                elif now - self.silence_start_time >= SILENCE_DURATION_S:
                    # We've had enough silence - process if we have audio AND speech was detected
                    audio_duration = now - self.recording_start_time
                    
                    if len(self.audio_buffer) > 0 and audio_duration >= MIN_AUDIO_DURATION_S and self.speech_detected:
                        # Grab the audio and reset
                        audio_data = np.concatenate(self.audio_buffer)
                        log.info(f">>> TRIGGERING WHISPER: {len(audio_data)} samples ({len(audio_data)/RECORD_SAMPLE_RATE:.2f}s), peak_rms={self.peak_rms:.6f}")
                        self.audio_buffer = []
                        self.recording_start_time = now
                        self.silence_start_time = None
                        self.speech_detected = False  # Reset for next utterance
                        self.speech_confirm_count = 0  # Reset confirmation counter
                        self.peak_rms = 0.0  # Reset peak for next utterance
                        
                        # Trigger transcription in separate thread
                        threading.Thread(
                            target=self._process_audio,
                            args=(audio_data,),
                            daemon=True
                        ).start()
                    else:
                        # Not enough audio or no speech detected, just reset silently
                        self.audio_buffer = []
                        self.recording_start_time = now
                        self.silence_start_time = None
                        self.speech_detected = False
                        self.speech_confirm_count = 0  # Reset confirmation counter
                        self.peak_rms = 0.0  # Reset peak
                        if self.current_phase == 'speech-detected':
                            self._transition_to_phase('idle')
            else:
                # Above threshold - might be speech, but require sustained signal
                self.silence_start_time = None
                if self.current_phase in ('idle', 'speech-detected'):
                    self._transition_to_phase('speech-detected')
                self.speech_confirm_count += 1
                
                if not self.speech_detected and self.speech_confirm_count >= SPEECH_CONFIRM_CHUNKS:
                    # Confirmed speech - multiple consecutive chunks above threshold
                    log.info(f">>> VOICE CONFIRMED (RMS={rms:.6f} > threshold={threshold})")
                    self.speech_detected = True
    
    def _process_audio(self, audio_data):
        """Process recorded audio through whisper and type result."""
        duration_s = len(audio_data) / RECORD_SAMPLE_RATE
        log.info(f">>> SUBMITTING TO WHISPER: {len(audio_data)} samples ({duration_s:.2f}s of audio)")
        
        # Signal that audio is headed to whisper (turn background orange)
        self._transition_to_phase('transcribing')
        
        def schedule_processing_complete():
            self._transition_to_phase('idle')

        try:
            text = transcribe_audio(audio_data)
            log.info(f">>> WHISPER RETURNED: '{text}'" if text else ">>> WHISPER RETURNED: (empty/None)")
            if text and text.strip():
                def typing_finished():
                    self._transition_to_phase('idle')
                    return False

                self._transition_to_phase('typing')
                # Schedule callback on main thread (text already has trailing space for flow)
                GLib.idle_add(self.on_speech_detected, text, typing_finished)
            else:
                # log.warning("Transcription returned empty text")
                schedule_processing_complete()
        except Exception as e:
            # log.error(f"Transcription error: {e}", exc_info=True)
            schedule_processing_complete()


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
        # log.debug(f"Audio too quiet to normalize (peak={peak})")
        return audio_data
    
    # Target peak at 80% of int16 max to avoid clipping
    target_peak = 32767 * 0.8
    gain = target_peak / peak
    
    # Limit gain to avoid amplifying quiet audio too much
    gain = min(gain, 20.0)  # Max 20x amplification
    
    # log.debug(f"Normalizing audio: peak={peak:.0f}, gain={gain:.2f}x")
    
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
    # log.debug(f"Audio data before resample: dtype={audio_data.dtype}, shape={audio_data.shape}, "
    #           f"min={audio_data.min()}, max={audio_data.max()}, mean={audio_data.mean():.2f}")
    
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
    # log.debug(f"Audio data after resample: min={resampled.min():.2f}, max={resampled.max():.2f}")
    
    return resampled.astype(np.int16)


def transcribe_audio(audio_data):
    """
    Transcribe audio using whisper.cpp.
    
    Args:
        audio_data: numpy array of int16 audio samples at RECORD_SAMPLE_RATE
    
    Returns:
        Transcribed text string
    """
    # log.info(f"transcribe_audio called with {len(audio_data)} samples at {RECORD_SAMPLE_RATE}Hz")
    
    if not WHISPER_BINARY.exists():
        # log.error(f"Whisper binary not found at {WHISPER_BINARY}")
        return None
    
    if not WHISPER_MODEL.exists():
        # log.error(f"Whisper model not found at {WHISPER_MODEL}")
        return None
    
    # Resample from recording rate to whisper rate (48kHz -> 16kHz)
    if RECORD_SAMPLE_RATE != WHISPER_SAMPLE_RATE:
        # log.debug(f"Resampling from {RECORD_SAMPLE_RATE}Hz to {WHISPER_SAMPLE_RATE}Hz")
        audio_data = resample_audio(audio_data, RECORD_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        # log.debug(f"Resampled to {len(audio_data)} samples")
    
    # Normalize audio to use more dynamic range (helps whisper with quiet audio)
    audio_data = normalize_audio(audio_data)

    # Write audio to a unique temporary WAV file (kept only during processing)
    # Use /dev/shm (RAM) if RAM_CACHE is enabled, otherwise use /tmp
    timestamp_ms = int(time.time() * 1000)
    temp_dir = Path("/dev/shm") if RAM_CACHE else Path("/tmp")
    wav_path = temp_dir / f"lingo2_data_{timestamp_ms}_{threading.get_ident()}.wav"
    
    # log.debug(f"Writing audio to file: {wav_path}")
    
    try:
        # Write WAV file at whisper's expected rate
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(WHISPER_SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        
        # log.info(f"WAV file written: {wav_path} ({os.path.getsize(wav_path)} bytes)")
        
        # Run whisper.cpp with library path set
        env = os.environ.copy()
        lib_paths = f"{WHISPER_LIB_DIR}:{GGML_LIB_DIR}"
        env["LD_LIBRARY_PATH"] = lib_paths + ":" + env.get("LD_LIBRARY_PATH", "")
        
        cmd = [
            str(WHISPER_BINARY),
            "-m", str(WHISPER_MODEL),
            "-f", str(wav_path),
            "--no-timestamps",
            "--language", "en",
            "--threads", "4",
        ]
        # log.debug(f"Running whisper command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        # log.debug(f"Whisper return code: {result.returncode}")
        # log.debug(f"Whisper stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
        # log.debug(f"Whisper stderr: {result.stderr[:500] if result.stderr else '(empty)'}")
        
        if result.returncode != 0:
            # log.error(f"Whisper error (code {result.returncode}): {result.stderr}")
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
        
        # Filter out whisper hallucinations: if the first character is not alphanumeric,
        # it's likely a hallucination like "(music)" or "[BLANK_AUDIO]" - ignore it
        if text and not text[0].isalnum():
            # log.warning(f"Filtered out likely hallucination (non-alphanumeric start): '{text}'")
            return None
        
        # Post-process: remove trailing punctuation, trim, add space for continuous dictation
        if text:
            # Strip trailing punctuation that whisper adds
            while text and text[-1] in '.?!,;:':
                text = text[:-1]
            text = text.strip()
            # Lowercase the first letter (sentence fragments don't need caps)
            if text and text[0].isupper():
                text = text[0].lower() + text[1:]
            if text:
                text = text + ' '  # Add space for smooth sentence flow
        
        return text
    
    except Exception as e:
        # log.error(f"Transcription exception: {e}", exc_info=True)
        return None
    finally:
        try:
            wav_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass


def cleanup_temp_audio_files():
    """
    Clean up temporary audio files created by previous runs.

    Removes all files matching the pattern lingo2_data_* from the
    temporary directory (either /dev/shm or /tmp depending on RAM_CACHE).
    """
    temp_dir = Path("/dev/shm") if RAM_CACHE else Path("/tmp")
    pattern = "lingo2_data_*"

    try:
        # Find all matching files
        files_to_remove = list(temp_dir.glob(pattern))

        if files_to_remove:
            log.info(f"Cleaning up {len(files_to_remove)} temporary audio file(s) from {temp_dir}")
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    log.debug(f"Removed: {file_path}")
                except Exception as e:
                    log.warning(f"Failed to remove {file_path}: {e}")
        else:
            log.debug(f"No temporary audio files to clean up in {temp_dir}")

    except Exception as e:
        log.error(f"Error during cleanup: {e}")


# =============================================================================
# Keyboard Injection via XDG Remote Desktop Portal (Wayland)
# =============================================================================

# Character to X11 Keysym mapping
# See: https://www.cl.cam.ac.uk/~mgk25/ucs/keysyms.txt
CHAR_TO_KEYSYM = {
    # Lowercase letters (XK_a - XK_z)
    'a': 0x0061, 'b': 0x0062, 'c': 0x0063, 'd': 0x0064, 'e': 0x0065,
    'f': 0x0066, 'g': 0x0067, 'h': 0x0068, 'i': 0x0069, 'j': 0x006a,
    'k': 0x006b, 'l': 0x006c, 'm': 0x006d, 'n': 0x006e, 'o': 0x006f,
    'p': 0x0070, 'q': 0x0071, 'r': 0x0072, 's': 0x0073, 't': 0x0074,
    'u': 0x0075, 'v': 0x0076, 'w': 0x0077, 'x': 0x0078, 'y': 0x0079,
    'z': 0x007a,
    
    # Uppercase letters (XK_A - XK_Z)
    'A': 0x0041, 'B': 0x0042, 'C': 0x0043, 'D': 0x0044, 'E': 0x0045,
    'F': 0x0046, 'G': 0x0047, 'H': 0x0048, 'I': 0x0049, 'J': 0x004a,
    'K': 0x004b, 'L': 0x004c, 'M': 0x004d, 'N': 0x004e, 'O': 0x004f,
    'P': 0x0050, 'Q': 0x0051, 'R': 0x0052, 'S': 0x0053, 'T': 0x0054,
    'U': 0x0055, 'V': 0x0056, 'W': 0x0057, 'X': 0x0058, 'Y': 0x0059,
    'Z': 0x005a,
    
    # Numbers (XK_0 - XK_9)
    '0': 0x0030, '1': 0x0031, '2': 0x0032, '3': 0x0033, '4': 0x0034,
    '5': 0x0035, '6': 0x0036, '7': 0x0037, '8': 0x0038, '9': 0x0039,
    
    # Common punctuation and symbols
    ' ': 0x0020,   # space
    '!': 0x0021,   # exclam
    '"': 0x0022,   # quotedbl
    '#': 0x0023,   # numbersign
    '$': 0x0024,   # dollar
    '%': 0x0025,   # percent
    '&': 0x0026,   # ampersand
    "'": 0x0027,   # apostrophe
    '(': 0x0028,   # parenleft
    ')': 0x0029,   # parenright
    '*': 0x002a,   # asterisk
    '+': 0x002b,   # plus
    ',': 0x002c,   # comma
    '-': 0x002d,   # minus
    '.': 0x002e,   # period
    '/': 0x002f,   # slash
    ':': 0x003a,   # colon
    ';': 0x003b,   # semicolon
    '<': 0x003c,   # less
    '=': 0x003d,   # equal
    '>': 0x003e,   # greater
    '?': 0x003f,   # question
    '@': 0x0040,   # at
    '[': 0x005b,   # bracketleft
    '\\': 0x005c,  # backslash
    ']': 0x005d,   # bracketright
    '^': 0x005e,   # asciicircum
    '_': 0x005f,   # underscore
    '`': 0x0060,   # grave
    '{': 0x007b,   # braceleft
    '|': 0x007c,   # bar
    '}': 0x007d,   # braceright
    '~': 0x007e,   # asciitilde
    '\n': 0xff0d,  # Return key
    '\t': 0xff09,  # Tab key
}

# DBus Portal constants
BUS_NAME = "org.freedesktop.portal.Desktop"
OBJ_PATH = "/org/freedesktop/portal/desktop"
REMOTE_DESKTOP_IFACE = "org.freedesktop.portal.RemoteDesktop"
REQUEST_IFACE = "org.freedesktop.portal.Request"

from gi.repository import Gio, GLib
import random
import string


class KeyboardInjector:
    """
    Handles keyboard injection via the XDG Remote Desktop Portal.
    
    This allows Wayland applications to simulate keyboard input without
    requiring root permissions or X11.
    
    Supports session persistence via restore tokens (Portal v2) to avoid
    repeated permission dialogs. The restore token is saved to config.
    """
    
    def __init__(self):
        self.connection = None
        self.session_handle = None
        self.pending_text = None
        self._pending_callback = None
        self._signal_id = None
        self._initializing = False
        self._initialized = False
        self._restore_token = None  # Token for restoring session without dialog
    
    def _generate_token(self):
        """Generate a unique token for portal requests."""
        return 'voicetyper_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
    
    def _get_request_path(self, token):
        """Get the DBus request object path for a given token."""
        sender = self.connection.get_unique_name().replace('.', '_').replace(':', '')
        return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"
    
    def initialize(self, callback=None):
        """
        Initialize the Remote Desktop portal session.
        This will trigger a system permission dialog on first use.
        
        Args:
            callback: Optional callback(success: bool) when initialization completes
        """
        if self._initialized:
            if callback:
                callback(True)
            return
        
        if self._initializing:
            # log.debug("Already initializing keyboard injector")
            return
        
        self._initializing = True
        self._init_callback = callback
        
        # log.info("Initializing Remote Desktop portal session...")
        
        try:
            self.connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            # log.debug("Got DBus session connection")
            
            # Step 1: Create session
            self._create_session()
            
        except Exception as e:
            # log.error(f"Failed to initialize keyboard injector: {e}", exc_info=True)
            self._initializing = False
            if callback:
                callback(False)
    
    def _create_session(self):
        """Create a Remote Desktop session."""
        token = self._generate_token()
        request_path = self._get_request_path(token)
        
        # log.debug(f"Creating session with token: {token}")
        
        # Subscribe to the Response signal before making the call
        self._signal_id = self.connection.signal_subscribe(
            BUS_NAME,
            REQUEST_IFACE,
            "Response",
            request_path,
            None,
            Gio.DBusSignalFlags.NO_MATCH_RULE,
            self._on_create_session_response,
            None
        )
        
        # Call CreateSession
        # Build the options dict for the variant
        session_token = self._generate_token()
        params = GLib.Variant("(a{sv})", ({
            "handle_token": GLib.Variant("s", token),
            "session_handle_token": GLib.Variant("s", session_token),
        },))
        
        self.connection.call(
            BUS_NAME,
            OBJ_PATH,
            REMOTE_DESKTOP_IFACE,
            "CreateSession",
            params,
            GLib.VariantType("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._on_create_session_call_done,
            None
        )
    
    def _on_create_session_call_done(self, connection, result, user_data):
        """Called when CreateSession DBus call completes."""
        try:
            res = connection.call_finish(result)
            request_path = res.unpack()[0]
            # log.debug(f"CreateSession call returned request path: {request_path}")
        except Exception as e:
            # log.error(f"CreateSession call failed: {e}")
            self._cleanup_signal()
            self._initializing = False
            if self._init_callback:
                self._init_callback(False)
    
    def _on_create_session_response(self, connection, sender_name, object_path, 
                                     interface_name, signal_name, parameters, user_data):
        """Handle the Response signal from CreateSession."""
        self._cleanup_signal()
        
        response, results = parameters.unpack()
        # log.debug(f"CreateSession response: {response}, results: {results}")
        
        if response != 0:
            # log.error(f"CreateSession failed with response code: {response}")
            self._initializing = False
            if self._init_callback:
                self._init_callback(False)
            return
        
        self.session_handle = results.get("session_handle", None)
        if not self.session_handle:
            # log.error("No session_handle in CreateSession response")
            self._initializing = False
            if self._init_callback:
                self._init_callback(False)
            return
        
        # log.info(f"Session created: {self.session_handle}")
        
        # Step 2: Select devices (keyboard)
        self._select_devices()
    
    def _select_devices(self):
        """Select keyboard device for the session."""
        token = self._generate_token()
        request_path = self._get_request_path(token)
        
        # log.debug("Selecting keyboard device...")
        
        # Subscribe to Response signal
        self._signal_id = self.connection.signal_subscribe(
            BUS_NAME,
            REQUEST_IFACE,
            "Response",
            request_path,
            None,
            Gio.DBusSignalFlags.NO_MATCH_RULE,
            self._on_select_devices_response,
            None
        )
        
        # types: 1=keyboard, 2=pointer, 3=both
        # persist_mode: 0=don't persist, 1=persist while app runs, 2=persist until revoked
        options = {
            "handle_token": GLib.Variant("s", token),
            "types": GLib.Variant("u", 1),  # Keyboard only
            "persist_mode": GLib.Variant("u", 2),  # Persist until explicitly revoked
        }
        
        # Load saved restore token from config
        config = load_config()
        saved_token = config.get("portal_restore_token")
        if saved_token:
            log.debug(f"Using saved restore token to skip permission dialog")
            options["restore_token"] = GLib.Variant("s", saved_token)
        
        params = GLib.Variant("(oa{sv})", (
            self.session_handle,
            options
        ))
        
        self.connection.call(
            BUS_NAME,
            OBJ_PATH,
            REMOTE_DESKTOP_IFACE,
            "SelectDevices",
            params,
            GLib.VariantType("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._on_select_devices_call_done,
            None
        )
    
    def _on_select_devices_call_done(self, connection, result, user_data):
        """Called when SelectDevices DBus call completes."""
        try:
            res = connection.call_finish(result)
            # log.debug(f"SelectDevices call returned: {res.unpack()}")
        except Exception as e:
            # log.error(f"SelectDevices call failed: {e}")
            self._cleanup_signal()
            self._initializing = False
            if self._init_callback:
                self._init_callback(False)
    
    def _on_select_devices_response(self, connection, sender_name, object_path,
                                     interface_name, signal_name, parameters, user_data):
        """Handle the Response signal from SelectDevices."""
        self._cleanup_signal()
        
        response, results = parameters.unpack()
        # log.debug(f"SelectDevices response: {response}")
        
        if response != 0:
            # log.error(f"SelectDevices failed with response code: {response}")
            self._initializing = False
            if self._init_callback:
                self._init_callback(False)
            return
        
        # Step 3: Start the session
        self._start_session()
    
    def _start_session(self):
        """Start the Remote Desktop session - this triggers the permission dialog."""
        token = self._generate_token()
        request_path = self._get_request_path(token)
        
        # log.debug("Starting session (permission dialog may appear)...")
        
        # Subscribe to Response signal
        self._signal_id = self.connection.signal_subscribe(
            BUS_NAME,
            REQUEST_IFACE,
            "Response",
            request_path,
            None,
            Gio.DBusSignalFlags.NO_MATCH_RULE,
            self._on_start_session_response,
            None
        )
        
        # parent_window is empty string for no parent
        params = GLib.Variant("(osa{sv})", (
            self.session_handle,
            "",
            {
                "handle_token": GLib.Variant("s", token),
            }
        ))
        
        self.connection.call(
            BUS_NAME,
            OBJ_PATH,
            REMOTE_DESKTOP_IFACE,
            "Start",
            params,
            GLib.VariantType("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            self._on_start_session_call_done,
            None
        )
    
    def _on_start_session_call_done(self, connection, result, user_data):
        """Called when Start DBus call completes."""
        try:
            res = connection.call_finish(result)
            # log.debug(f"Start call returned: {res.unpack()}")
        except Exception as e:
            # log.error(f"Start call failed: {e}")
            self._cleanup_signal()
            self._initializing = False
            if self._init_callback:
                self._init_callback(False)
    
    def _on_start_session_response(self, connection, sender_name, object_path,
                                    interface_name, signal_name, parameters, user_data):
        """Handle the Response signal from Start."""
        self._cleanup_signal()
        
        response, results = parameters.unpack()
        # log.debug(f"Start session response: {response}, results: {results}")
        
        if response != 0:
            if response == 1:
                # log.warning("User cancelled the Remote Desktop permission dialog")
                # Clear any saved token since user denied permission
                self._clear_restore_token()
            else:
                # log.error(f"Start session failed with response code: {response}")
                # Token may be invalid/expired, clear it so next attempt prompts fresh
                self._clear_restore_token()
            self._initializing = False
            self._initialized = False
            if self.pending_text:
                # Ensure pending callbacks fire even if we cannot type
                callback = self._pending_callback
                self.pending_text = None
                self._pending_callback = None
                if callback:
                    callback()
            if self._init_callback:
                self._init_callback(False)
            return
        
        # Save the restore token for future sessions (Portal v2 feature)
        # This allows skipping the permission dialog on subsequent runs
        new_token = results.get("restore_token")
        if new_token:
            self._restore_token = new_token
            config = load_config()
            config["portal_restore_token"] = new_token
            save_config(config)
            log.info("Saved portal restore token - future runs won't need permission dialog")
        
        # log.info("✅ Remote Desktop session started successfully!")
        self._initialized = True
        self._initializing = False
        
        if self._init_callback:
            self._init_callback(True)
        
        # If there was pending text, type it now
        if self.pending_text:
            text = self.pending_text
            callback = self._pending_callback
            self.pending_text = None
            self._pending_callback = None
            self._start_typing(text, callback)
    
    def _clear_restore_token(self):
        """Clear the saved restore token from config."""
        self._restore_token = None
        config = load_config()
        if config.get("portal_restore_token"):
            config["portal_restore_token"] = None
            save_config(config)
            log.debug("Cleared saved portal restore token")
    
    def _cleanup_signal(self):
        """Unsubscribe from current signal."""
        if self._signal_id and self.connection:
            self.connection.signal_unsubscribe(self._signal_id)
            self._signal_id = None
    
    def type_text(self, text, on_finished=None):
        """
        Type the given text by simulating keyboard input.
        
        Args:
            text: The string to type
            on_finished: Optional callback invoked when typing completes
        """
        if not self._initialized:
            # log.warning("Keyboard injector not initialized, queuing text")
            self.pending_text = text
            self._pending_callback = on_finished
            if not self._initializing:
                self.initialize()
            return
        
        # log.info(f"Typing text: '{text}'")
        self._start_typing(text, on_finished)

    def _start_typing(self, text, on_finished):
        """Begin typing text with optional completion callback."""
        if not text:
            if on_finished:
                on_finished()
            return

        def type_char_at_index(index):
            if index >= len(text):
                if on_finished:
                    on_finished()
                return False

            char = text[index]
            keysym = CHAR_TO_KEYSYM.get(char)

            if keysym is None:
                # Try Unicode keysym for unsupported characters
                keysym = 0x01000000 | ord(char)
                # log.debug(f"Using Unicode keysym for '{char}': {hex(keysym)}")

            self._send_key(keysym, pressed=True)
            self._send_key(keysym, pressed=False)

            GLib.timeout_add(1, type_char_at_index, index + 1)
            return False

        GLib.idle_add(type_char_at_index, 0)
    
    def _send_key(self, keysym, pressed):
        """
        Send a single key event via the Remote Desktop portal.
        
        Args:
            keysym: The X11 keysym to send
            pressed: True for key down, False for key up
        """
        if not self.session_handle:
            # log.error("No session handle, cannot send key")
            return
        
        state = 1 if pressed else 0
        
        try:
            # NotifyKeyboardKeysym(session_handle, options, keysym, state)
            # keysym is signed int (i), state is unsigned int (u)
            self.connection.call_sync(
                BUS_NAME,
                OBJ_PATH,
                REMOTE_DESKTOP_IFACE,
                "NotifyKeyboardKeysym",
                GLib.Variant("(oa{sv}iu)", (
                    self.session_handle,
                    {},  # options (empty)
                    keysym,
                    state
                )),
                None,
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
        except Exception as e:
            # log.error(f"Failed to send key {hex(keysym)} (pressed={pressed}): {e}")
            pass
    
    def close(self):
        """Close the Remote Desktop session."""
        if self.session_handle and self.connection:
            try:
                # Close the session
                self.connection.call_sync(
                    BUS_NAME,
                    self.session_handle,
                    "org.freedesktop.portal.Session",
                    "Close",
                    None,
                    None,
                    Gio.DBusCallFlags.NONE,
                    -1,
                    None
                )
                # log.info("Remote Desktop session closed")
            except Exception as e:
                # log.debug(f"Error closing session (may already be closed): {e}")
                pass
        
        self._cleanup_signal()
        self.session_handle = None
        self._initialized = False


# Global keyboard injector instance
_keyboard_injector = None


def get_keyboard_injector():
    """Get or create the global keyboard injector instance."""
    global _keyboard_injector
    if _keyboard_injector is None:
        _keyboard_injector = KeyboardInjector()
    return _keyboard_injector


def type_text(text, on_finished=None):
    """
    Type text by simulating keyboard input via XDG Remote Desktop Portal.
    
    This is the main entry point for typing transcribed text.
    On first call, it will trigger a system permission dialog.
    
    Args:
        text: The string to type
        on_finished: Optional callback invoked after typing completes
    """
    # log.info(f"TRANSCRIBED: '{text}'")
    print(f"\n🎯 TRANSCRIBED: {text}\n")
    
    injector = get_keyboard_injector()
    injector.type_text(text, on_finished=on_finished)


# =============================================================================
# GTK4 Application
# =============================================================================

class VoiceTyperWindow(Gtk.ApplicationWindow):
    """Simple dialog window for voice typing control."""

    PHASE_CSS_CLASSES = {
        'speech-detected': 'phase-speech',
        'transcribing': 'phase-transcribing',
        'typing': 'phase-typing',
    }
    
    def __init__(self, app):
        super().__init__(application=app)
        
        self.recorder = None
        self.is_recording = False
        self.config = load_config()
        self._active_phase_class = None
        
        # Window setup - small dialog with title bar (draggable)
        self.set_title("Lingo")
        self.set_default_size(200, -1)  # Wider to fit dropdown
        self.set_resizable(False)
        
        # Create main vertical box
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(15)
        vbox.set_margin_end(15)
        
        # Create microphone dropdown
        self.mic_dropdown = Gtk.ComboBoxText()
        self._populate_mic_dropdown()
        self.mic_dropdown.connect("changed", self.on_mic_changed)
        vbox.append(self.mic_dropdown)
        
        # Create horizontal box for checkbox and close button
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        
        # Create "Microphone" checkbox
        self.mic_checkbox = Gtk.CheckButton(label="Microphone")
        self.mic_checkbox.connect("toggled", self.on_mic_toggled)
        hbox.append(self.mic_checkbox)
        
        # Create close button (pushed to the right)
        close_button = Gtk.Button(label="Close")
        close_button.add_css_class("close-button")
        close_button.set_hexpand(True)  # Take remaining space
        close_button.set_halign(Gtk.Align.END)  # Align to right
        close_button.connect("clicked", lambda btn: self.close())
        hbox.append(close_button)
        
        vbox.append(hbox)

        vbox.append(self._create_advanced_settings_section())
        
        # Store reference to main container for processing state styling
        self.main_container = vbox
        self._active_phase_class = None
        
        # Add CSS for styling
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window {
                background: @theme_bg_color;
            }
            window.phase-speech {
                background: #27AE60;
            }
            window.phase-transcribing {
                background: #E67E22;
            }
            window.phase-typing {
                background: #FFFFFF;
                color: #000000;
            }
            combobox {
                font-size: 20px;
            }
            checkbutton {
                font-size: 20px;
                margin-top: 5px;
            }
            checkbutton check {
                min-width: 24px;
                min-height: 24px;
            }
            checkbutton:checked label {
                color: #e53935;
                font-weight: bold;
            }
            .close-button {
                min-width: 60px;
                min-height: 36px;
                padding: 6px 16px;
                font-size: 16px;
            }
            .help-text {
                font-size: 14px;
                opacity: 0.7;
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
        
        # Auto-start microphone after window is fully shown
        # Show orange background to indicate initialization in progress
        GLib.idle_add(self._auto_start_microphone)
    
    def _set_background_phase(self, phase):
        """Update the window background based on the current processing phase."""
        target_class = self.PHASE_CSS_CLASSES.get(phase)
        if phase in (None, 'idle'):
            target_class = None

        if self._active_phase_class == target_class:
            return

        if self._active_phase_class:
            self.remove_css_class(self._active_phase_class)
            self._active_phase_class = None

        if target_class:
            self.add_css_class(target_class)
            self._active_phase_class = target_class

    def _create_advanced_settings_section(self):
        expander = Gtk.Expander(label="Advanced Settings")
        expander.set_expanded(False)
        expander.set_hexpand(True)

        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        container.set_margin_top(6)
        container.set_margin_start(4)
        container.set_margin_end(4)
        container.set_margin_bottom(4)

        threshold_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        threshold_label = Gtk.Label(label="Silence threshold", xalign=0)
        threshold_label.set_hexpand(True)
        threshold_row.append(threshold_label)

        self.threshold_entry = Gtk.Entry()
        self.threshold_entry.set_width_chars(6)
        self.threshold_entry.set_max_length(8)
        self.threshold_entry.set_input_purpose(Gtk.InputPurpose.NUMBER)
        self.threshold_entry.set_tooltip_text("Lower values detect quieter audio; higher values filter background noise.")
        self.threshold_entry.set_text(self._format_threshold(self._get_silence_threshold()))
        self.threshold_entry.connect("activate", self.on_threshold_entry_activate)

        focus_controller = Gtk.EventControllerFocus()
        focus_controller.connect("leave", self.on_threshold_entry_focus_leave)
        self.threshold_entry.add_controller(focus_controller)

        threshold_row.append(self.threshold_entry)
        container.append(threshold_row)

        help_label = Gtk.Label(
            label="Typical range is 0.001–0.02: lower values make the mic more sensitive; raise it to ignore background noise."+
            " Set the value as low as possible for best reliablility. If you have any problems with this application try lowering this threshold value.",
            xalign=0
        )
        help_label.set_wrap(True)
        help_label.add_css_class("help-text")
        container.append(help_label)

        expander.set_child(container)
        return expander

    def _get_silence_threshold(self):
        return float(self.config.get('silence_threshold', DEFAULT_SILENCE_THRESHOLD))

    def _format_threshold(self, value):
        formatted = f"{float(value):.4f}".rstrip('0').rstrip('.')
        return formatted if formatted else "0"

    def _set_threshold_entry_text(self, value):
        self.threshold_entry.set_text(self._format_threshold(value))

    def _commit_threshold_entry(self):
        text = self.threshold_entry.get_text().strip()
        if not text:
            self._set_threshold_entry_text(self._get_silence_threshold())
            return
        try:
            value = float(text)
        except ValueError:
            print("⚠️  Silence threshold must be a number.")
            self._set_threshold_entry_text(self._get_silence_threshold())
            return
        if value <= 0:
            print("⚠️  Silence threshold must be positive.")
            self._set_threshold_entry_text(self._get_silence_threshold())
            return
        self._update_silence_threshold(value)

    def _update_silence_threshold(self, value):
        value = float(value)
        current = self._get_silence_threshold()
        if abs(current - value) < 1e-6:
            self._set_threshold_entry_text(value)
            return
        self.config['silence_threshold'] = value
        save_config(self.config)
        if self.recorder:
            self.recorder.set_silence_threshold(value)
        self._set_threshold_entry_text(value)
        print(f"🎚️ Silence threshold set to {self._format_threshold(value)}")

    def on_threshold_entry_activate(self, entry):
        self._commit_threshold_entry()

    def on_threshold_entry_focus_leave(self, controller):
        self._commit_threshold_entry()

    def _auto_start_microphone(self):
        """Auto-start the microphone when the application launches."""
        # Show orange background (same as transcribing phase) while initializing
        self._set_background_phase('transcribing')
        
        # Small delay to let the UI render the orange background first
        GLib.timeout_add(100, self._complete_auto_start)
        return False  # Don't repeat idle_add
    
    def _complete_auto_start(self):
        """Complete the auto-start after showing the orange initialization indicator."""
        # Check the mic checkbox (this triggers _activate_microphone via on_mic_toggled)
        self.mic_checkbox.set_active(True)
        
        # Remove the initialization background and let live phases drive the color
        self._set_background_phase('idle')
        return False  # Don't repeat timeout
    
    def _populate_mic_dropdown(self):
        """Populate the microphone dropdown with available input devices."""
        # Add "System Default" as first option
        self.mic_dropdown.append_text("System Default")
        
        # Add all input devices
        self.input_devices = get_input_devices()
        for device in self.input_devices:
            self.mic_dropdown.append_text(device['name'])
        
        # Select saved device or default
        saved_device = self.config.get('audio_device')
        if saved_device is None:
            self.mic_dropdown.set_active(0)  # System Default
        else:
            # Find the device in the list
            found = False
            for i, device in enumerate(self.input_devices):
                if device['name'] == saved_device:
                    self.mic_dropdown.set_active(i + 1)  # +1 because of "System Default"
                    found = True
                    break
            if not found:
                log.warning(f"Saved device '{saved_device}' not found, using default")
                self.mic_dropdown.set_active(0)
    
    def on_mic_changed(self, combo):
        """Handle microphone selection change."""
        # Stop recording if active
        if self.mic_checkbox.get_active():
            self.mic_checkbox.set_active(False)
        
        # Get selected device
        active_index = combo.get_active()
        if active_index == 0:
            # System Default
            self.config['audio_device'] = None
        else:
            # Specific device
            device = self.input_devices[active_index - 1]
            self.config['audio_device'] = device['name']
        
        # Save config
        save_config(self.config)
        log.info(f"Microphone changed to: {self.config['audio_device'] or 'System Default'}")
    
    def on_mic_toggled(self, checkbox):
        """Handle mic checkbox toggle."""
        if checkbox.get_active():
            self._activate_microphone()
        else:
            self.stop_recording()
    
    def _activate_microphone(self):
        """Activate the microphone - shared logic for startup and checkbox toggle."""
        self.start_recording()
    
    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            return
        
        audio_device = self.config.get('audio_device')
        self.recorder = AudioRecorder(
            self.on_speech_detected,
            audio_device=audio_device,
            on_processing_phase_changed=self.on_processing_phase_changed,
            silence_threshold=self._get_silence_threshold()
        )
        success, error_msg = self.recorder.start()
        
        if not success:
            # Show error dialog and uncheck the mic checkbox
            self._show_mic_error(error_msg)
            self.mic_checkbox.set_active(False)
            self.recorder = None
            return
        
        self.is_recording = True
        print("🎤 Microphone ON - listening...")
        
        # Initialize keyboard injector (triggers permission dialog on first use)
        injector = get_keyboard_injector()
        if not injector._initialized and not injector._initializing:
            # log.info("Initializing keyboard injector for first use...")
            injector.initialize(callback=self._on_keyboard_ready)
    
    def _show_mic_error(self, message):
        """Show a warning dialog about microphone issues."""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK,
            text="Microphone Error"
        )
        dialog.set_secondary_text(message)
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()
    
    def _on_keyboard_ready(self, success):
        """Called when keyboard injector initialization completes."""
        if success:
            # log.info("✅ Keyboard injector ready - transcribed text will be typed")
            print("⌨️  Keyboard access granted - ready to type!")
        else:
            # log.warning("❌ Keyboard injector failed - text will only be logged")
            print("⚠️  Keyboard access denied - text will be logged but not typed")
    
    def stop_recording(self):
        """Stop audio recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        # log.info("Recording stopped.")
        print("🔇 Microphone OFF")
    
    def on_processing_phase_changed(self, phase):
        """Called when the recorder reports a processing phase change."""
        if phase in self.PHASE_CSS_CLASSES:
            self._set_background_phase(phase)
            log.debug(f"Processing phase active: {phase}")
        else:
            self._set_background_phase('idle')
            log.debug("Processing phase active: idle")
        return False  # Don't repeat (for GLib.idle_add)

    def on_speech_detected(self, text, completion_callback=None):
        """Called when speech is transcribed."""
        # log.info(f"on_speech_detected callback called with: '{text}'")
        finished_cb = completion_callback or self._on_typing_finished
        try:
            type_text(text, on_finished=finished_cb)
        except Exception:
            finished_cb()
        return False  # Don't repeat

    def _on_typing_finished(self):
        """Reset processing indicator after typing completes."""
        self.on_processing_phase_changed('idle')
    
    def on_close_request(self, window):
        """Clean up when window closes."""
        self.stop_recording()

        # Clean up keyboard injector
        global _keyboard_injector
        if _keyboard_injector:
            _keyboard_injector.close()
            _keyboard_injector = None

        # Clean up temporary audio files
        cleanup_temp_audio_files()

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

    # Clean up any temporary audio files from previous runs
    cleanup_temp_audio_files()

    # Run the GTK application
    app = VoiceTyperApp()
    return app.run(None)


if __name__ == "__main__":
    exit(main())
