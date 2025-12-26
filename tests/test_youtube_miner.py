"""
Unit tests for YouTube Miner pipeline.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pydub import AudioSegment

# Import the module to test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from youtube_miner import YouTubeMiner


class TestYouTubeMiner:
    """Test suite for YouTubeMiner class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def miner(self, temp_dir):
        """Create a YouTubeMiner instance for testing."""
        with patch('youtube_miner.WhisperModel'), \
             patch('youtube_miner.Pipeline'):
            miner = YouTubeMiner(
                output_dir=temp_dir,
                chunk_duration=30,
                whisper_model="tiny"
            )
            return miner
    
    @pytest.fixture
    def sample_audio_file(self, temp_dir):
        """Create a sample audio file for testing."""
        # Create a simple 1-second silent audio file
        audio = AudioSegment.silent(duration=1000, frame_rate=16000)
        audio_file = Path(temp_dir) / "test_audio.wav"
        audio.export(str(audio_file), format="wav")
        return str(audio_file)
    
    def test_initialization(self, temp_dir):
        """Test YouTubeMiner initialization."""
        with patch('youtube_miner.WhisperModel') as mock_whisper, \
             patch('youtube_miner.Pipeline') as mock_pipeline:
            miner = YouTubeMiner(output_dir=temp_dir)
            
            assert miner.output_dir == Path(temp_dir)
            assert miner.chunk_duration == 30
            assert miner.whisper_model_name == "tiny"
            mock_whisper.assert_called_once()
    
    def test_initialization_custom_params(self, temp_dir):
        """Test initialization with custom parameters."""
        with patch('youtube_miner.WhisperModel'), \
             patch('youtube_miner.Pipeline'):
            miner = YouTubeMiner(
                output_dir=temp_dir,
                chunk_duration=60,
                whisper_model="base"
            )
            
            assert miner.chunk_duration == 60
            assert miner.whisper_model_name == "base"
    
    @patch('youtube_miner.YoutubeDL')
    def test_download_audio_success(self, mock_ydl_class, miner):
        """Test successful audio download."""
        # Mock yt-dlp
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        # Mock video info
        mock_info = {
            'id': 'test123',
            'title': 'Test Video',
            'duration': 120
        }
        mock_ydl.extract_info.return_value = mock_info
        
        # Create mock audio file
        audio_file = Path(miner.output_dir) / "downloads" / "test123.wav"
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        audio_file.touch()
        
        audio_path, video_info = miner.download_audio("https://youtube.com/watch?v=test123")
        
        assert video_info['id'] == 'test123'
        assert video_info['title'] == 'Test Video'
        assert video_info['duration'] == 120
        assert os.path.exists(audio_path)
    
    @patch('youtube_miner.YoutubeDL')
    def test_download_audio_failure(self, mock_ydl_class, miner):
        """Test audio download failure handling."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.side_effect = Exception("Download failed")
        
        with pytest.raises(Exception):
            miner.download_audio("https://youtube.com/watch?v=invalid")
    
    @patch('youtube_miner.YoutubeDL')
    def test_extract_youtube_captions_success(self, mock_ydl_class, miner):
        """Test successful caption extraction."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        mock_info = {
            'id': 'test123',
            'automatic_captions': {
                'en': [{'url': 'http://example.com/captions.vtt'}]
            }
        }
        mock_ydl.extract_info.return_value = mock_info
        
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello world"
            mock_response.__enter__.return_value = mock_response
            mock_urlopen.return_value = mock_response
            
            captions = miner.extract_youtube_captions("https://youtube.com/watch?v=test123")
            
            assert captions is not None
            assert "Hello world" in captions
    
    @patch('youtube_miner.YoutubeDL')
    def test_extract_youtube_captions_not_available(self, mock_ydl_class, miner):
        """Test caption extraction when captions are not available."""
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        
        mock_info = {
            'id': 'test123',
            'automatic_captions': {}
        }
        mock_ydl.extract_info.return_value = mock_info
        
        captions = miner.extract_youtube_captions("https://youtube.com/watch?v=test123")
        
        assert captions is None
    
    def test_simple_vad(self, miner, sample_audio_file):
        """Test simple VAD fallback."""
        segments = miner._simple_vad(sample_audio_file)
        
        assert isinstance(segments, list)
        # For a silent audio file, we might get no segments or some segments
        # depending on the threshold
    
    def test_create_chunks(self, miner, sample_audio_file):
        """Test chunk creation."""
        # Create longer audio file
        audio = AudioSegment.silent(duration=35000, frame_rate=16000)  # 35 seconds
        long_audio_file = Path(miner.output_dir) / "long_audio.wav"
        audio.export(str(long_audio_file), format="wav")
        
        # Mock speech segments (35 seconds of speech)
        speech_segments = [(0.0, 35.0)]
        
        chunks = miner.create_chunks(str(long_audio_file), speech_segments)
        
        assert len(chunks) > 0
        assert all('index' in chunk for chunk in chunks)
        assert all('start_time' in chunk for chunk in chunks)
        assert all('end_time' in chunk for chunk in chunks)
        assert all('file_path' in chunk for chunk in chunks)
    
    def test_create_chunks_short_segments(self, miner, sample_audio_file):
        """Test chunk creation with short segments."""
        # Short segments should be filtered out
        speech_segments = [(0.0, 2.0)]  # 2 seconds - too short
        
        chunks = miner.create_chunks(sample_audio_file, speech_segments)
        
        # Should be filtered out (less than 5 seconds)
        assert len(chunks) == 0
    
    @patch.object(YouTubeMiner, 'whisper_model')
    def test_transcribe_chunk(self, mock_whisper_model, miner, sample_audio_file):
        """Test chunk transcription."""
        # Mock Whisper transcription
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Hello world"
        
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        
        mock_whisper_model.transcribe.return_value = (
            [mock_segment],
            mock_info
        )
        
        result = miner.transcribe_chunk(sample_audio_file)
        
        assert 'full_text' in result
        assert 'segments' in result
        assert 'language' in result
        assert result['language'] == "en"
        assert "Hello world" in result['full_text']
    
    def test_compare_transcriptions(self, miner):
        """Test transcription comparison."""
        whisper_text = "Hello world this is a test"
        youtube_text = "Hello world this is a test"
        
        result = miner.compare_transcriptions(whisper_text, youtube_text)
        
        assert result['youtube_available'] is True
        assert result['jaccard_similarity'] > 0.9
        assert result['character_similarity'] > 0.9
    
    def test_compare_transcriptions_different(self, miner):
        """Test comparison with different transcriptions."""
        whisper_text = "Hello world"
        youtube_text = "Goodbye universe"
        
        result = miner.compare_transcriptions(whisper_text, youtube_text)
        
        assert result['youtube_available'] is True
        assert result['jaccard_similarity'] < 0.5
    
    def test_compare_transcriptions_no_youtube(self, miner):
        """Test comparison when YouTube captions are not available."""
        whisper_text = "Hello world"
        
        result = miner.compare_transcriptions(whisper_text, None)
        
        assert result['youtube_available'] is False
        assert 'message' in result
    
    @patch.object(YouTubeMiner, 'download_audio')
    @patch.object(YouTubeMiner, 'extract_youtube_captions')
    @patch.object(YouTubeMiner, 'detect_voice_activity')
    @patch.object(YouTubeMiner, 'create_chunks')
    @patch.object(YouTubeMiner, 'transcribe_chunk')
    @patch.object(YouTubeMiner, 'compare_transcriptions')
    def test_process_pipeline(
        self,
        mock_compare,
        mock_transcribe,
        mock_create_chunks,
        mock_vad,
        mock_extract_captions,
        mock_download,
        miner,
        temp_dir
    ):
        """Test complete processing pipeline."""
        # Setup mocks
        mock_download.return_value = (
            "test_audio.wav",
            {'id': 'test123', 'title': 'Test', 'duration': 120}
        )
        mock_extract_captions.return_value = "YouTube captions text"
        mock_vad.return_value = [(0.0, 30.0)]
        mock_create_chunks.return_value = [
            {
                'index': 0,
                'start_time': 0.0,
                'end_time': 30.0,
                'duration': 30.0,
                'file_path': 'chunk_0000.wav'
            }
        ]
        mock_transcribe.return_value = {
            'full_text': 'Whisper transcription',
            'segments': [],
            'language': 'en',
            'language_probability': 0.99
        }
        mock_compare.return_value = {
            'youtube_available': True,
            'jaccard_similarity': 0.85
        }
        
        results = miner.process("https://youtube.com/watch?v=test123", chunk_index=0)
        
        assert 'video_info' in results
        assert 'transcription' in results
        assert 'comparison' in results
        assert results['chunk_index'] == 0
        
        # Verify all methods were called
        mock_download.assert_called_once()
        mock_extract_captions.assert_called_once()
        mock_vad.assert_called_once()
        mock_create_chunks.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_compare.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

