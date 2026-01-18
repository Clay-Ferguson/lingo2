#!/usr/bin/env python3
"""
FastAPI server that bridges the browser to whisper.cpp for speech-to-text.

Endpoints:
  POST /transcribe - Receives audio blob, returns transcribed text
  GET /health - Health check endpoint
  Static files - Serves the web app (lingo.html, lingo.js, lingo.css)
"""

import asyncio
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Whisper Speech-to-Text Server")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent  # One level up from web-app/
WHISPER_DIR = PROJECT_ROOT / "whisper-model"
WHISPER_BINARY = WHISPER_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = WHISPER_DIR / "whisper.cpp" / "models" / "ggml-base.en.bin"

# Temp directory for audio files
TEMP_DIR = Path(tempfile.gettempdir()) / "whisper-lingo"
TEMP_DIR.mkdir(exist_ok=True)


def check_dependencies():
    """Check if required dependencies are available."""
    errors = []
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        errors.append("ffmpeg is not installed or not in PATH")
    
    # Check whisper binary
    if not WHISPER_BINARY.exists():
        errors.append(f"whisper.cpp binary not found at {WHISPER_BINARY}")
    
    # Check whisper model
    if not WHISPER_MODEL.exists():
        errors.append(f"whisper model not found at {WHISPER_MODEL}")
    
    return errors


@app.on_event("startup")
async def startup_event():
    """Check dependencies on startup."""
    errors = check_dependencies()
    if errors:
        print("\n⚠️  WARNING: Missing dependencies:")
        for error in errors:
            print(f"   - {error}")
        print("\nThe /transcribe endpoint will not work until these are resolved.")
        print("Run the setup commands from WHISPER_INSTRUCTIONS.md to install whisper.cpp\n")
    else:
        print("\n✅ All dependencies found. Whisper server ready!")
        print(f"   Binary: {WHISPER_BINARY}")
        print(f"   Model: {WHISPER_MODEL}\n")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    errors = check_dependencies()
    return {
        "status": "ok" if not errors else "degraded",
        "whisper_ready": len(errors) == 0,
        "errors": errors
    }


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio to text using whisper.cpp.
    
    Accepts audio in any format that ffmpeg can read (webm, mp3, wav, etc.)
    Returns JSON with the transcribed text.
    """
    # Check dependencies first
    errors = check_dependencies()
    if errors:
        raise HTTPException(
            status_code=503,
            detail=f"Whisper not ready: {'; '.join(errors)}"
        )
    
    # Generate unique filenames
    file_id = str(uuid.uuid4())[:8]
    input_path = TEMP_DIR / f"input_{file_id}.webm"
    wav_path = TEMP_DIR / f"audio_{file_id}.wav"
    
    try:
        # Save uploaded audio to temp file
        content = await audio.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        with open(input_path, "wb") as f:
            f.write(content)
        
        # Convert to 16kHz mono WAV using ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            "-c:a", "pcm_s16le",  # 16-bit PCM
            str(wav_path)
        ]
        
        ffmpeg_result = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, ffmpeg_stderr = await ffmpeg_result.communicate()
        
        if ffmpeg_result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"ffmpeg conversion failed: {ffmpeg_stderr.decode()}"
            )
        
        # Run whisper.cpp
        whisper_cmd = [
            str(WHISPER_BINARY),
            "-m", str(WHISPER_MODEL),
            "-f", str(wav_path),
            "--no-timestamps",
            "--language", "en",
            "--threads", "4",
        ]
        
        whisper_result = await asyncio.create_subprocess_exec(
            *whisper_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        whisper_stdout, whisper_stderr = await whisper_result.communicate()
        
        if whisper_result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"whisper.cpp failed: {whisper_stderr.decode()}"
            )
        
        # Parse output - whisper outputs text to stdout
        # The output format is typically just the transcribed text
        text = whisper_stdout.decode().strip()
        
        # Clean up any whisper metadata lines (they start with timestamps or brackets)
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and metadata
            if not line:
                continue
            # Skip lines that look like whisper metadata
            if line.startswith("[") or line.startswith("whisper_"):
                continue
            clean_lines.append(line)
        
        transcribed_text = " ".join(clean_lines)
        
        return JSONResponse({
            "success": True,
            "text": transcribed_text
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp files
        for path in [input_path, wav_path]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass


# Mount static files AFTER API routes so API takes precedence
# This serves lingo.html, lingo.js, lingo.css from the same directory
app.mount("/", StaticFiles(directory=str(SCRIPT_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
