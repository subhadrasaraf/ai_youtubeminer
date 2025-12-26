# YouTube Miner - Data Pipeline

A Python pipeline that downloads YouTube audio, performs Voice Activity Detection (VAD), creates clean 30-second chunks, and compares transcriptions from Whisper with YouTube's auto-generated captions.

## Track Selected
**Task 3: "The YouTube Miner" (Data Pipeline)**

## Features

- ✅ Downloads audio from YouTube videos using `yt-dlp`
- ✅ Voice Activity Detection (VAD) using `pyannote.audio` with fallback to energy-based VAD
- ✅ Automatic chunking into 30-second segments (removing silence/music)
- ✅ Transcription using open-source Whisper model (`faster-whisper`)
- ✅ Extracts YouTube auto-generated captions
- ✅ Compares Whisper transcription with YouTube captions
- ✅ Comprehensive unit test coverage
- ✅ No paid APIs - 100% open-source

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- Internet connection (for downloading YouTube videos)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai_project
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.

### 5. (Optional) Pyannote VAD Model

The pipeline will attempt to use `pyannote.audio` for VAD. If it fails to load, it automatically falls back to a simple energy-based VAD method. For best results, you can accept the HuggingFace model terms at [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection) and use an access token.

## Usage

### Basic Usage

```bash
python youtube_miner.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Advanced Usage

```bash
# Specify chunk index to transcribe
python youtube_miner.py "https://www.youtube.com/watch?v=VIDEO_ID" --chunk 2

# Specify output directory
python youtube_miner.py "https://www.youtube.com/watch?v=VIDEO_ID" --output my_outputs

# Use different Whisper model (tiny, base, small, medium, large)
python youtube_miner.py "https://www.youtube.com/watch?v=VIDEO_ID" --model base
```

### Programmatic Usage

```python
from youtube_miner import YouTubeMiner

# Initialize miner
miner = YouTubeMiner(
    output_dir="outputs",
    chunk_duration=30,
    whisper_model="tiny"
)

# Process a YouTube video
results = miner.process(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    chunk_index=0
)

# Access results
print(f"Video: {results['video_info']['title']}")
print(f"Transcription: {results['transcription']['full_text']}")
print(f"Similarity: {results['comparison']['jaccard_similarity']}")
```

## Output Structure

```
outputs/
├── downloads/
│   └── VIDEO_ID.wav          # Downloaded audio file
├── chunks/
│   ├── chunk_0000.wav        # 30-second chunks
│   ├── chunk_0001.wav
│   └── ...
└── results.json              # Complete results with metadata
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=youtube_miner --cov-report=html

# Run specific test file
pytest tests/test_youtube_miner.py -v
```

## Project Structure

```
ai_project/
├── youtube_miner.py          # Main pipeline script
├── requirements.txt          # Python dependencies
├── pytest.ini               # Pytest configuration
├── README.md                # This file
├── TECHNICAL_DESIGN.md      # Technical design document
├── tests/
│   ├── __init__.py
│   └── test_youtube_miner.py # Unit tests
└── outputs/                 # Generated outputs (gitignored)
```

## How It Works

1. **Audio Download**: Uses `yt-dlp` to download audio from YouTube URL
2. **VAD Processing**: Detects speech segments using `pyannote.audio` (or fallback method)
3. **Chunking**: Creates clean 30-second chunks from speech segments
4. **Transcription**: Transcribes selected chunk using `faster-whisper` (Whisper-Tiny)
5. **Caption Extraction**: Extracts YouTube auto-generated captions
6. **Comparison**: Compares transcriptions using Jaccard similarity and character-level metrics

## Limitations

- Requires FFmpeg for audio processing
- VAD model may require HuggingFace authentication for best results
- YouTube captions may not be available for all videos
- Processing time depends on video length and model size

## Troubleshooting

### OpenMP Error on macOS
If you see "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized", the script automatically sets `KMP_DUPLICATE_LIB_OK=TRUE` to fix this. This is handled in the code, but if you still see the error, you can manually set it:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python youtube_miner.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### NumPy Version Compatibility
If you get NumPy 2.x compatibility errors, reinstall NumPy 1.x:
```bash
pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
```

### FFmpeg not found
Ensure FFmpeg is installed and in your system PATH.

### Pyannote VAD fails to load
The pipeline automatically falls back to energy-based VAD. This is normal and the pipeline will continue to work.

### No captions available
Some videos don't have auto-generated captions. The comparison will indicate this in the results.

### Memory issues with large videos
Use a smaller Whisper model (`tiny` or `base`) or process shorter videos.

## License

This project is created for educational purposes as part of a coding challenge.

## Author

Created for the AI Project Challenge - Task 3: The YouTube Miner

