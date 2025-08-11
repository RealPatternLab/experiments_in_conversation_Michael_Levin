#!/usr/bin/env python3
"""
Test the smart 5-second frame extraction on a single video.
"""

from tools.smart_10_second_extraction import Smart5SecondExtractor

def test_single_video():
    """Test smart 5-second extraction on a single video."""
    extractor = Smart5SecondExtractor()
    
    # Test on one video
    video_id = "youtube_z2IuMTY4KVg"
    
    # Get chunks for this video
    chunks = extractor.load_semantic_chunks()
    video_chunks = [c for c in chunks if c.get('video_id') == video_id]
    
    print(f"Found {len(video_chunks)} chunks for {video_id}")
    
    # Calculate needed timestamps
    timestamps = extractor.calculate_needed_timestamps(video_chunks)
    print(f"Timestamps needed: {sorted(timestamps)}")
    
    # Show some examples of the calculation
    print("\nExample calculations:")
    for i, chunk in enumerate(video_chunks[:3]):  # Show first 3 chunks
        start = chunk.get('start_time', 0)
        end = chunk.get('end_time', 0)
        midpoint = (start + end) / 2
        rounded = round(midpoint / 5) * 5
        print(f"  Chunk {i+1}: start={start:.1f}s, end={end:.1f}s, midpoint={midpoint:.1f}s, rounded={rounded}s")
    
    # Extract frames
    success = extractor.process_video(video_id, video_chunks)
    if success:
        print("Success! All frames extracted and cleaned up")
    else:
        print("Failed to extract frames")

if __name__ == "__main__":
    test_single_video() 