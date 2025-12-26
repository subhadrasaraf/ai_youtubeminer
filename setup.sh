#!/bin/bash

# Setup script for YouTube Miner project

echo "Setting up YouTube Miner project..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg found: $(ffmpeg -version | head -n 1)"
else
    echo "WARNING: FFmpeg not found. Please install FFmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt-get install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
fi

echo ""
echo "Setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To run the pipeline: python youtube_miner.py <YOUTUBE_URL>"

