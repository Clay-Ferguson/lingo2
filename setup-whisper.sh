#!/bin/bash

# Setup script for Whisper.cpp
# This downloads and builds whisper.cpp with the base.en model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$SCRIPT_DIR/whisper-model"

echo "=========================================="
echo "  Whisper.cpp Setup Script               "
echo "=========================================="
echo ""

# Check for required build tools
echo "Checking dependencies..."

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is not installed"
    echo "Install it with: sudo apt install git"
    exit 1
fi

if ! command -v make >/dev/null 2>&1; then
    echo "ERROR: make is not installed"
    echo "Install it with: sudo apt install build-essential"
    exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
    echo "ERROR: g++ is not installed"
    echo "Install it with: sudo apt install build-essential"
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake is not installed"
    echo "Install it with: sudo apt install cmake"
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg is not installed"
    echo "Install it with: sudo apt install ffmpeg"
    echo "ffmpeg is required for audio conversion"
    echo ""
fi

# Create whisper-model directory
echo ""
echo "Creating whisper-model directory..."
mkdir -p "$WHISPER_DIR"
cd "$WHISPER_DIR"

# Clone whisper.cpp if not already present
if [ ! -d "whisper.cpp" ]; then
    echo ""
    echo "Cloning whisper.cpp repository..."
    git clone https://github.com/ggerganov/whisper.cpp.git
else
    echo ""
    echo "whisper.cpp already exists, pulling latest..."
    cd whisper.cpp
    git pull
    cd ..
fi

# Build whisper.cpp
echo ""
echo "Building whisper.cpp (this may take a few minutes)..."
cd whisper.cpp

# Clean previous build if exists
rm -rf build

# Build with cmake
cmake -B build
cmake --build build --config Release -j$(nproc)

# The binary is in build/bin/
if [ ! -f "build/bin/whisper-cli" ]; then
    echo "ERROR: Build failed - 'whisper-cli' binary not found"
    exit 1
fi

# Create a symlink for backward compatibility (main -> whisper-cli)
ln -sf build/bin/whisper-cli main

echo ""
echo "Build successful!"

# Download model if not present
if [ ! -f "models/ggml-base.en.bin" ]; then
    echo ""
    echo "Downloading base.en model (~150MB)..."
    ./models/download-ggml-model.sh base.en
    
    if [ ! -f "models/ggml-base.en.bin" ]; then
        echo "ERROR: Model download failed"
        exit 1
    fi
else
    echo ""
    echo "Model already downloaded: models/ggml-base.en.bin"
fi

# Test the installation
echo ""
echo "Testing whisper.cpp with sample audio..."
if [ -f "samples/jfk.wav" ]; then
    ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav --no-timestamps 2>/dev/null
    echo ""
fi

echo "=========================================="
echo "  Setup Complete!                        "
echo "=========================================="
echo ""
echo "Whisper binary: $WHISPER_DIR/whisper.cpp/build/bin/whisper-cli"
echo "Model file:     $WHISPER_DIR/whisper.cpp/models/ggml-base.en.bin"
echo ""
echo "You can now run the app with: ./run.sh"
echo ""
