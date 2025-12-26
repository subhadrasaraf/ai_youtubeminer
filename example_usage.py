"""
Example usage of YouTube Miner pipeline.
"""

from youtube_miner import YouTubeMiner

def main():
    # Example YouTube URL (replace with actual video)
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Initialize the miner
    print("Initializing YouTube Miner...")
    miner = YouTubeMiner(
        output_dir="outputs",
        chunk_duration=30,
        whisper_model="tiny"  # Options: tiny, base, small, medium, large
    )
    
    # Process the video
    print(f"\nProcessing: {youtube_url}")
    results = miner.process(youtube_url, chunk_index=0)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Video Title: {results['video_info']['title']}")
    print(f"Duration: {results['video_info']['duration']} seconds")
    print(f"Speech Segments: {len(results['speech_segments'])}")
    print(f"Chunks Created: {len(results['chunks'])}")
    
    if 'transcription' in results:
        print(f"\nTranscription (Chunk {results['chunk_index']}):")
        print("-" * 60)
        print(results['transcription']['full_text'])
        print("-" * 60)
        print(f"Language: {results['transcription']['language']}")
        print(f"Confidence: {results['transcription']['language_probability']:.2%}")
    
    if results['comparison']['youtube_available']:
        print(f"\nComparison with YouTube Captions:")
        print(f"  Jaccard Similarity: {results['comparison']['jaccard_similarity']:.2%}")
        print(f"  Character Similarity: {results['comparison']['character_similarity']:.2%}")
        print(f"  Common Words: {results['comparison']['common_words']}")
    else:
        print("\nYouTube captions not available for comparison")
    
    print(f"\nFull results saved to: outputs/results.json")

if __name__ == "__main__":
    main()

