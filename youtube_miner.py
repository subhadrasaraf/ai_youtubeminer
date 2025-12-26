"""
YouTube Miner - Data Pipeline
Downloads YouTube audio, performs VAD chunking, and compares transcriptions.
"""

import os
# Fix OpenMP library conflict on macOS (must be set before importing torch/whisper)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads to avoid conflicts
os.environ['MKL_NUM_THREADS'] = '1'   # Limit MKL threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Limit NumExpr threads

import json
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yt_dlp
from pyannote.audio import Pipeline
# Delay faster_whisper import to avoid segfault during module import
# Will import when needed in transcribe_chunk method
import torch
import torchaudio
from pydub import AudioSegment
import numpy as np


class YouTubeMiner:
    """Main class for YouTube audio processing and transcription pipeline."""
    
    def __init__(
        self,
        output_dir: str = "outputs",
        chunk_duration: int = 30,
        whisper_model: str = "tiny",
        vad_model_path: Optional[str] = None
    ):
        """
        Initialize YouTube Miner.
        
        Args:
            output_dir: Directory to save outputs
            chunk_duration: Duration of each chunk in seconds (default: 30)
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            vad_model_path: Optional path to custom VAD model
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chunk_duration = chunk_duration
        self.whisper_model_name = whisper_model
        self.whisper_model = None  # Lazy loading - load when needed
        self.youtube_caption_segments = []  # Store timed caption segments
        
        # Initialize VAD pipeline (using pyannote)
        print("Loading VAD pipeline...")
        self.vad_pipeline = None
        try:
            # Using pyannote's pretrained VAD model
            # Note: May require HuggingFace token for some models
            # If it fails, we'll use the fallback method
            try:
                # Try without use_auth_token first (newer versions)
                self.vad_pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection"
                )
                print("VAD pipeline loaded successfully")
            except TypeError:
                # Fallback for older pyannote versions
                self.vad_pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection",
                    use_auth_token=None
                )
                print("VAD pipeline loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pyannote VAD model: {e}")
            print("Falling back to simple energy-based VAD...")
            self.vad_pipeline = None
    
    def download_audio(self, youtube_url: str) -> Tuple[str, Dict]:
        """
        Download audio from YouTube URL.
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            Tuple of (audio_file_path, video_info_dict)
        """
        print(f"Downloading audio from: {youtube_url}")
        
        # Create temporary directory for downloads
        download_dir = self.output_dir / "downloads"
        download_dir.mkdir(exist_ok=True)
        
        # Configure yt-dlp options - download without FFmpeg post-processing to avoid segfault
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio',
            'outtmpl': str(download_dir / '%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info first (without downloading)
                info = ydl.extract_info(youtube_url, download=False)
                video_id = info.get('id', 'unknown')
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                # Now download the audio
                ydl.download([youtube_url])
                
                # Find the downloaded audio file (could be m4a, mp3, webm, opus, etc.)
                audio_file = None
                for ext in ['m4a', 'mp3', 'webm', 'opus', 'wav']:
                    potential_file = download_dir / f"{video_id}.{ext}"
                    if potential_file.exists():
                        audio_file = potential_file
                        break
                
                if audio_file is None:
                    # Try to find any audio file in the directory
                    audio_files = list(download_dir.glob("*"))
                    audio_files = [f for f in audio_files if f.suffix in ['.m4a', '.mp3', '.webm', '.opus', '.wav']]
                    if audio_files:
                        audio_file = audio_files[0]
                    else:
                        raise FileNotFoundError(f"Audio file not found after download in {download_dir}")
                
                # Convert to WAV if needed (using pydub which is safer than FFmpeg post-processing)
                if audio_file.suffix != '.wav':
                    print(f"Converting {audio_file.suffix} to WAV...")
                    try:
                        audio = AudioSegment.from_file(str(audio_file))
                        wav_file = download_dir / f"{video_id}.wav"
                        audio.export(str(wav_file), format="wav")
                        audio_file = wav_file
                        print("Conversion successful")
                    except Exception as conv_error:
                        print(f"Warning: Could not convert to WAV: {conv_error}")
                        print(f"Using original file: {audio_file}")
                
                video_info = {
                    'id': video_id,
                    'title': title,
                    'duration': duration,
                    'url': youtube_url
                }
                
                print(f"Downloaded: {title} ({duration}s)")
                return str(audio_file), video_info
                
        except Exception as e:
            print(f"Error downloading audio: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_youtube_captions(self, youtube_url: str) -> Optional[str]:
        """
        Extract captions from YouTube (both automatic and manual).
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            Caption text as string, or None if not available
        """
        print("Extracting YouTube captions...")
        
        # Create temporary directory for subtitle files
        subtitle_dir = self.output_dir / "subtitles"
        subtitle_dir.mkdir(exist_ok=True)
        
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],  # Try multiple English variants
            'skip_download': True,
            'quiet': False,
            'no_warnings': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First, get video info to check available captions
                info = ydl.extract_info(youtube_url, download=False)
                video_id = info.get('id', 'unknown')
                
                print(f"Checking available captions for video: {video_id}")
                
                # Check for subtitles (manual captions)
                subtitles_available = info.get('subtitles', {})
                # Check for automatic captions
                automatic_captions = info.get('automatic_captions', {})
                
                # Combine both types
                all_captions = {}
                if subtitles_available:
                    all_captions.update(subtitles_available)
                    print(f"Found manual subtitles in: {list(subtitles_available.keys())}")
                if automatic_captions:
                    all_captions.update(automatic_captions)
                    print(f"Found automatic captions in: {list(automatic_captions.keys())}")
                
                if not all_captions:
                    print("No captions available for this video")
                    return None
                
                # Try to download subtitles using yt-dlp
                ydl_opts_download = {
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en', 'en-US', 'en-GB'],
                    'skip_download': True,
                    'outtmpl': str(subtitle_dir / f'{video_id}.%(ext)s'),
                    'quiet': False,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts_download) as ydl_download:
                    try:
                        ydl_download.download([youtube_url])
                    except Exception as e:
                        print(f"Note: Could not download subtitle file: {e}")
                
                # Try to find downloaded subtitle file
                subtitle_file = None
                for ext in ['vtt', 'srt', 'ttml']:
                    potential_file = subtitle_dir / f"{video_id}.en.{ext}"
                    if potential_file.exists():
                        subtitle_file = potential_file
                        break
                    # Try without language code
                    potential_file2 = subtitle_dir / f"{video_id}.{ext}"
                    if potential_file2.exists():
                        subtitle_file = potential_file2
                        break
                
                if subtitle_file:
                    print(f"Found subtitle file: {subtitle_file}")
                    with open(subtitle_file, 'r', encoding='utf-8') as f:
                        caption_data = f.read()
                else:
                    # Fallback: try to get caption URL from info
                    print("Subtitle file not found, trying to fetch from URL...")
                    caption_url = None
                    
                    # Try manual subtitles first (usually better quality)
                    for lang in ['en', 'en-US', 'en-GB']:
                        if lang in subtitles_available and subtitles_available[lang]:
                            caption_url = subtitles_available[lang][0].get('url')
                            if caption_url:
                                print(f"Using manual subtitle URL for {lang}")
                                break
                    
                    # Fallback to automatic captions
                    if not caption_url:
                        for lang in ['en', 'en-US', 'en-GB']:
                            if lang in automatic_captions and automatic_captions[lang]:
                                caption_url = automatic_captions[lang][0].get('url')
                                if caption_url:
                                    print(f"Using automatic caption URL for {lang}")
                                    break
                    
                    if not caption_url:
                        print("No caption URL found")
                        return None
                    
                    import urllib.request
                    with urllib.request.urlopen(caption_url) as response:
                        caption_data = response.read().decode('utf-8')
                
                # Parse VTT/SRT format and extract text with timestamps
                import re
                lines = caption_data.split('\n')
                
                # Store timed segments for time-based extraction
                timed_segments = []
                text_lines = []
                
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Skip empty lines and headers
                    if not line or line.startswith('WEBVTT') or line.startswith('<?xml') or line.startswith('<tt'):
                        i += 1
                        continue
                    
                    # Check for timestamp line (VTT: "00:00:01.000 --> 00:00:03.000" or SRT format)
                    timestamp_pattern = r'(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{3})'
                    match = re.search(timestamp_pattern, line)
                    
                    if match:
                        # Parse timestamp
                        start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
                        end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])
                        
                        start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
                        end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0
                        
                        # Collect text lines until next timestamp or empty line
                        i += 1
                        segment_text = []
                        while i < len(lines):
                            text_line = lines[i].strip()
                            if not text_line:
                                break
                            if re.search(timestamp_pattern, text_line):
                                i -= 1  # Back up to process this timestamp
                                break
                            if text_line.isdigit():  # SRT line number
                                i += 1
                                continue
                            # Remove HTML tags
                            text_line = re.sub(r'<[^>]+>', '', text_line)
                            if text_line:
                                segment_text.append(text_line)
                            i += 1
                        
                        if segment_text:
                            segment_text_str = ' '.join(segment_text)
                            timed_segments.append({
                                'start': start_time,
                                'end': end_time,
                                'text': segment_text_str
                            })
                            text_lines.append(segment_text_str)
                    
                    i += 1
                
                # Store timed segments for later use
                self.youtube_caption_segments = timed_segments
                
                # Also return full text for backward compatibility
                captions = ' '.join(text_lines)
                print(f"Extracted {len(captions)} characters of captions ({len(timed_segments)} timed segments)")
                return captions
                    
        except Exception as e:
            print(f"Error extracting captions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_voice_activity(self, audio_file: str) -> List[Tuple[float, float]]:
        """
        Detect voice activity in audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        print("Detecting voice activity...")
        
        if self.vad_pipeline is None:
            # Fallback: simple energy-based VAD
            return self._simple_vad(audio_file)
        
        try:
            # Use pyannote VAD
            vad_output = self.vad_pipeline(audio_file)
            
            speech_segments = []
            for segment in vad_output.itertracks(yield_label=True):
                start = segment[0].start
                end = segment[0].end
                label = segment[2]
                
                if label == 'SPEECH':
                    speech_segments.append((start, end))
            
            print(f"Found {len(speech_segments)} speech segments")
            return speech_segments
            
        except Exception as e:
            print(f"Error in VAD: {e}, falling back to simple VAD")
            return self._simple_vad(audio_file)
    
    def _simple_vad(self, audio_file: str) -> List[Tuple[float, float]]:
        """
        Simple energy-based VAD fallback.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of (start_time, end_time) tuples
        """
        print("Using simple energy-based VAD...")
        
        # Load audio
        audio = AudioSegment.from_wav(audio_file)
        sample_rate = audio.frame_rate
        samples = np.array(audio.get_array_of_samples())
        
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        # Calculate energy in windows
        window_size = int(sample_rate * 0.1)  # 100ms windows
        energy_threshold = np.percentile(np.abs(samples), 30)
        
        speech_segments = []
        in_speech = False
        start_time = 0
        
        for i in range(0, len(samples), window_size):
            window = samples[i:i+window_size]
            energy = np.mean(np.abs(window))
            
            if energy > energy_threshold and not in_speech:
                in_speech = True
                start_time = i / sample_rate
            elif energy <= energy_threshold and in_speech:
                in_speech = False
                end_time = i / sample_rate
                if end_time - start_time > 0.5:  # Minimum 0.5s segment
                    speech_segments.append((start_time, end_time))
        
        if in_speech:
            speech_segments.append((start_time, len(samples) / sample_rate))
        
        print(f"Found {len(speech_segments)} speech segments (simple VAD)")
        return speech_segments
    
    def _merge_speech_segments(
        self,
        speech_segments: List[Tuple[float, float]],
        gap_threshold: float = 2.0
    ) -> List[Tuple[float, float]]:
        """
        Merge nearby speech segments to create continuous speech regions.
        
        Args:
            speech_segments: List of (start, end) tuples
            gap_threshold: Maximum gap in seconds to merge segments (default: 2.0)
            
        Returns:
            List of merged (start, end) tuples
        """
        if not speech_segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(speech_segments, key=lambda x: x[0])
        
        merged = []
        current_start, current_end = sorted_segments[0]
        
        for start, end in sorted_segments[1:]:
            # If gap is small enough, merge segments
            if start - current_end <= gap_threshold:
                current_end = max(current_end, end)  # Extend to include both segments
            else:
                # Gap is too large, save current segment and start new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add the last segment
        merged.append((current_start, current_end))
        
        return merged
    
    def create_chunks(
        self,
        audio_file: str,
        speech_segments: List[Tuple[float, float]]
    ) -> List[Dict]:
        """
        Create clean 30-second chunks from speech segments, removing silence/music.
        
        Args:
            audio_file: Path to audio file
            speech_segments: List of (start, end) tuples for speech
            
        Returns:
            List of chunk dictionaries with metadata
        """
        print(f"Creating clean {self.chunk_duration}-second chunks (removing silence/music)...")
        
        if not speech_segments:
            print("No speech segments found, no chunks created")
            return []
        
        # First, merge nearby speech segments to create continuous speech regions
        print(f"Merging {len(speech_segments)} speech segments...")
        merged_segments = self._merge_speech_segments(speech_segments, gap_threshold=2.0)
        print(f"Created {len(merged_segments)} continuous speech regions")
        
        # Load audio
        audio = AudioSegment.from_wav(audio_file)
        
        chunks_dir = self.output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        chunks = []
        chunk_index = 0
        
        # Create chunks from merged segments
        for start, end in merged_segments:
            duration = end - start
            
            if duration >= self.chunk_duration:
                # Split long segments into multiple 30-second chunks
                num_chunks = int(duration / self.chunk_duration)
                for i in range(num_chunks):
                    chunk_start = start + (i * self.chunk_duration)
                    chunk_end = min(chunk_start + self.chunk_duration, end)
                    
                    # Extract clean audio chunk (no silence/music)
                    chunk_audio = audio[int(chunk_start * 1000):int(chunk_end * 1000)]
                    
                    chunk_file = chunks_dir / f"chunk_{chunk_index:04d}.wav"
                    chunk_audio.export(str(chunk_file), format="wav")
                    
                    chunks.append({
                        'index': chunk_index,
                        'start_time': chunk_start,
                        'end_time': chunk_end,
                        'duration': chunk_end - chunk_start,
                        'file_path': str(chunk_file)
                    })
                    chunk_index += 1
            elif duration >= 5:  # Keep segments at least 5 seconds
                # Keep shorter segments as-is (they're already clean, no silence)
                chunk_audio = audio[int(start * 1000):int(end * 1000)]
                
                chunk_file = chunks_dir / f"chunk_{chunk_index:04d}.wav"
                chunk_audio.export(str(chunk_file), format="wav")
                
                chunks.append({
                    'index': chunk_index,
                    'start_time': start,
                    'end_time': end,
                    'duration': duration,
                    'file_path': str(chunk_file)
                })
                chunk_index += 1
        
        print(f"Created {len(chunks)} clean chunks (silence/music removed)")
        return chunks
    
    def transcribe_chunk(self, chunk_file: str) -> Dict:
        """
        Transcribe a single chunk using Whisper.
        
        Args:
            chunk_file: Path to chunk audio file
            
        Returns:
            Dictionary with transcription results
        """
        # Lazy load Whisper model to avoid segfault during initialization
        if self.whisper_model is None:
            # Import faster_whisper here to delay loading
            from faster_whisper import WhisperModel
            
            print(f"Loading Whisper model: {self.whisper_model_name}...")
            try:
                self.whisper_model = WhisperModel(
                    self.whisper_model_name,
                    device="cpu",
                    compute_type="int8"
                )
                print("Whisper model loaded successfully")
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                # Try with float16 compute type as fallback
                print("Trying with float16 compute type...")
                try:
                    self.whisper_model = WhisperModel(
                        self.whisper_model_name,
                        device="cpu",
                        compute_type="float16"
                    )
                    print("Whisper model loaded successfully with float16")
                except Exception as e2:
                    print(f"Error loading Whisper model with float16: {e2}")
                    # Last resort: try default settings
                    print("Trying with default settings...")
                    self.whisper_model = WhisperModel(self.whisper_model_name)
                    print("Whisper model loaded successfully with default settings")
        
        print(f"Transcribing: {chunk_file}")
        
        segments, info = self.whisper_model.transcribe(
            chunk_file,
            beam_size=5,
            language="en"
        )
        
        # Collect all segments
        text_segments = []
        full_text = ""
        
        for segment in segments:
            text_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
            full_text += segment.text.strip() + " "
        
        return {
            'language': info.language,
            'language_probability': info.language_probability,
            'segments': text_segments,
            'full_text': full_text.strip()
        }
    
    def extract_caption_for_time_range(
        self,
        start_time: float,
        end_time: float,
        youtube_caption_segments: List[Dict]
    ) -> str:
        """
        Extract caption text for a specific time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            youtube_caption_segments: List of timed caption segments
            
        Returns:
            Caption text for the specified time range
        """
        if not youtube_caption_segments:
            return ""
        
        matching_segments = []
        for segment in youtube_caption_segments:
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Check if segment overlaps with the requested time range
            # Overlap occurs if: seg_start < end_time AND seg_end > start_time
            if seg_start < end_time and seg_end > start_time:
                matching_segments.append(segment['text'])
        
        return ' '.join(matching_segments)
    
    def compare_transcriptions(
        self,
        whisper_text: str,
        youtube_text: Optional[str],
        chunk_start_time: Optional[float] = None,
        chunk_end_time: Optional[float] = None
    ) -> Dict:
        """
        Compare Whisper transcription with YouTube captions.
        
        Args:
            whisper_text: Transcription from Whisper
            youtube_text: Full transcription from YouTube captions (for fallback)
            chunk_start_time: Start time of the chunk in seconds (optional)
            chunk_end_time: End time of the chunk in seconds (optional)
            
        Returns:
            Dictionary with comparison metrics
        """
        if youtube_text is None:
            return {
                'youtube_available': False,
                'message': 'YouTube captions not available for comparison'
            }
        
        # Extract caption text for the specific time range if timing is provided
        if chunk_start_time is not None and chunk_end_time is not None and self.youtube_caption_segments:
            print(f"Extracting YouTube captions for time range: {chunk_start_time:.2f}s - {chunk_end_time:.2f}s")
            youtube_text_segment = self.extract_caption_for_time_range(
                chunk_start_time,
                chunk_end_time,
                self.youtube_caption_segments
            )
            
            if youtube_text_segment:
                print(f"Found {len(youtube_text_segment)} characters of matching YouTube captions")
                youtube_text_to_compare = youtube_text_segment
            else:
                print("No matching YouTube captions found for this time range, using full caption")
                youtube_text_to_compare = youtube_text
        else:
            print("No time range provided, comparing with full YouTube caption")
            youtube_text_to_compare = youtube_text
        
        # Simple comparison metrics
        whisper_words = set(whisper_text.lower().split())
        youtube_words = set(youtube_text_to_compare.lower().split())

        print(f'Whisper words: {len(whisper_words)}')
        print(f'YouTube words (for this chunk): {len(youtube_words)}')

        # Calculate word overlap
        common_words = whisper_words.intersection(youtube_words)
        all_words = whisper_words.union(youtube_words)
        print(f'Common words: {len(common_words)}')
        print(f'Total unique words: {len(all_words)}')

        jaccard_similarity = len(common_words) / len(all_words) if all_words else 0
        
        # Character-level similarity (simple)
        whisper_chars = whisper_text.lower().replace(' ', '')
        youtube_chars = youtube_text_to_compare.lower().replace(' ', '')
        
        min_len = min(len(whisper_chars), len(youtube_chars))
        max_len = max(len(whisper_chars), len(youtube_chars))
        
        char_similarity = 0
        if max_len > 0:
            matches = sum(1 for i in range(min_len) if whisper_chars[i] == youtube_chars[i])
            char_similarity = matches / max_len
        
        return {
            'youtube_available': True,
            'chunk_start_time': chunk_start_time,
            'chunk_end_time': chunk_end_time,
            'whisper_word_count': len(whisper_words),
            'youtube_word_count': len(youtube_words),
            'common_words': len(common_words),
            'jaccard_similarity': round(jaccard_similarity, 4),
            'character_similarity': round(char_similarity, 4),
            'whisper_text': whisper_text,
            'youtube_text_segment': youtube_text_to_compare[:500] + "..." if len(youtube_text_to_compare) > 500 else youtube_text_to_compare,
            'youtube_text_full': youtube_text[:500] + "..." if len(youtube_text) > 500 else youtube_text
        }
    
    def process(self, youtube_url: str, chunk_index: int = 0) -> Dict:
        """
        Complete pipeline: download, VAD, chunking, transcription, comparison.
        
        Args:
            youtube_url: YouTube video URL
            chunk_index: Index of chunk to transcribe (default: 0)
            
        Returns:
            Dictionary with all results
        """
        print("=" * 60)
        print("YouTube Miner Pipeline Starting")
        print("=" * 60)
        
        results = {
            'youtube_url': youtube_url,
            'chunk_index': chunk_index
        }
        
        # Step 1: Download audio
        audio_file, video_info = self.download_audio(youtube_url)
        results['video_info'] = video_info
        
        # Step 2: Extract YouTube captions
        youtube_captions = self.extract_youtube_captions(youtube_url)
        results['youtube_captions'] = youtube_captions
        
        # Step 3: Voice Activity Detection
        speech_segments = self.detect_voice_activity(audio_file)
        results['speech_segments'] = speech_segments
        
        # Step 4: Create chunks
        chunks = self.create_chunks(audio_file, speech_segments)
        results['chunks'] = chunks
        
        if not chunks:
            print("Warning: No chunks created!")
            return results
        
        # Step 5: Transcribe selected chunk
        if chunk_index >= len(chunks):
            chunk_index = 0
            print(f"Warning: chunk_index out of range, using chunk 0")
        
        selected_chunk = chunks[chunk_index]
        transcription = self.transcribe_chunk(selected_chunk['file_path'])
        results['transcription'] = transcription
        
        # Step 6: Compare transcriptions (only for the specific chunk time range)
        comparison = self.compare_transcriptions(
            transcription['full_text'],
            youtube_captions,
            chunk_start_time=selected_chunk['start_time'],
            chunk_end_time=selected_chunk['end_time']
        )
        results['comparison'] = comparison
        
        # Save results to JSON
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("=" * 60)
        print("Pipeline Complete!")
        print(f"Results saved to: {results_file}")
        print("=" * 60)
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="YouTube Miner - Audio Processing Pipeline")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--chunk", type=int, default=0, help="Chunk index to transcribe (default: 0)")
    parser.add_argument("--output", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--model", default="tiny", help="Whisper model size (default: tiny)")
    
    args = parser.parse_args()
    
    miner = YouTubeMiner(
        output_dir=args.output,
        whisper_model=args.model
    )
    
    results = miner.process(args.url, args.chunk)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Video: {results['video_info']['title']}")
    print(f"Duration: {results['video_info']['duration']}s")
    print(f"Speech segments found: {len(results['speech_segments'])}")
    print(f"Chunks created: {len(results['chunks'])}")
    
    if 'transcription' in results:
        print(f"\nTranscription (Chunk {args.chunk}):")
        print(results['transcription']['full_text'])
    
    if results['comparison']['youtube_available']:
        print(f"\nComparison:")
        print(f"  Jaccard Similarity: {results['comparison']['jaccard_similarity']}")
        print(f"  Character Similarity: {results['comparison']['character_similarity']}")


if __name__ == "__main__":
    main()

