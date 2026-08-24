"""Audio -> text via whisper.cpp. Deliberately Qt-free.

Everything here is plain Python and numpy, so the resampling, normalization
and whisper post-processing rules can be exercised without a display or a
microphone attached -- feed `transcribe_audio` samples read from a WAV file.

The commented-out `log.debug` calls are left in place on purpose: they are the
diagnostic scaffolding referenced by TROUBLESHOOTING.md when a quiet mic is
not triggering, and are meant to be uncommented rather than rewritten.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# This file lives at <root>/qt-app/voice_typer/, so the project root -- and the
# whisper-model directory shared with web-app -- is three levels up.
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
WHISPER_DIR = PROJECT_ROOT / "whisper-model"
WHISPER_BINARY = WHISPER_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = WHISPER_DIR / "whisper.cpp" / "models" / "ggml-base.en.bin"
WHISPER_LIB_DIR = WHISPER_DIR / "whisper.cpp" / "build" / "src"  # For libwhisper.so
GGML_LIB_DIR = WHISPER_DIR / "whisper.cpp" / "build" / "ggml" / "src"  # For libggml.so

# RAM_CACHE: Use /dev/shm (tmpfs in RAM) instead of /tmp for audio files.
# Setting this to True avoids disk I/O and reduces SSD wear.
RAM_CACHE = True

# Recording at 48kHz (common USB mic rate), resampled to 16kHz for whisper.
RECORD_SAMPLE_RATE = 48000  # Rate to record from microphone
WHISPER_SAMPLE_RATE = 16000  # Rate whisper.cpp expects
CHANNELS = 1
DTYPE = np.int16

# =============================================================================
# Audio Conditioning
# =============================================================================


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize audio to use more of the int16 dynamic range.

    This helps whisper recognize quieter audio.
    """
    audio_float = audio_data.astype(np.float32)

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

    normalized = np.clip(normalized, -32768, 32767)
    return normalized.astype(np.int16)


def resample_audio(audio_data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio from orig_rate to target_rate using linear interpolation.

    Accepts the (N, channels) shape sounddevice produces as well as 1D, and
    always returns 1D.
    """
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()

    if orig_rate == target_rate:
        return audio_data

    duration = len(audio_data) / orig_rate
    num_samples = int(duration * target_rate)

    indices = np.linspace(0, len(audio_data) - 1, num_samples)

    resampled = np.interp(indices, np.arange(len(audio_data)), audio_data.astype(np.float32))

    return resampled.astype(np.int16)


# =============================================================================
# Transcription
# =============================================================================


def transcribe_audio(audio_data: np.ndarray) -> str | None:
    """Transcribe int16 samples at RECORD_SAMPLE_RATE, returning text or None.

    Returns None rather than raising for every failure mode (missing binary,
    whisper error, hallucination filtered out), because every caller treats
    "no usable text" identically.
    """
    if not WHISPER_BINARY.exists():
        log.error(f"Whisper binary not found at {WHISPER_BINARY}")
        return None

    if not WHISPER_MODEL.exists():
        log.error(f"Whisper model not found at {WHISPER_MODEL}")
        return None

    if RECORD_SAMPLE_RATE != WHISPER_SAMPLE_RATE:
        audio_data = resample_audio(audio_data, RECORD_SAMPLE_RATE, WHISPER_SAMPLE_RATE)

    # Normalize audio to use more dynamic range (helps whisper with quiet audio)
    audio_data = normalize_audio(audio_data)

    # Unique name so concurrent transcriptions cannot collide on the same file.
    timestamp_ms = int(time.time() * 1000)
    temp_dir = Path("/dev/shm") if RAM_CACHE else Path("/tmp")
    wav_path = temp_dir / f"lingo2_data_{timestamp_ms}_{threading.get_ident()}.wav"

    try:
        # Write WAV file at whisper's expected rate
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(WHISPER_SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        # whisper-cli links against libwhisper/libggml in the build tree, which
        # are not on the system loader path.
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

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)

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

        # Filter out whisper hallucinations: if the first character is not
        # alphanumeric it is likely "(music)" or "[BLANK_AUDIO]" - ignore it.
        if text and not text[0].isalnum():
            return None

        # Post-process for continuous dictation: this is a sentence fragment
        # being appended mid-sentence, not a standalone utterance.
        if text:
            while text and text[-1] in ".?!,;:":
                text = text[:-1]
            text = text.strip()
            if text and text[0].isupper():
                text = text[0].lower() + text[1:]
            if text:
                text = text + " "  # Add space for smooth sentence flow

        return text

    except Exception as e:
        log.error(f"Transcription exception: {e}")
        return None
    finally:
        try:
            wav_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass


def cleanup_temp_audio_files() -> None:
    """Remove lingo2_data_* files left behind by previous runs."""
    temp_dir = Path("/dev/shm") if RAM_CACHE else Path("/tmp")
    pattern = "lingo2_data_*"

    try:
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
