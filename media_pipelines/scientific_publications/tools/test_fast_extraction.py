#!/usr/bin/env python3
"""
Test the fast frame extraction on a single video.
"""

from tools.fast_frame_extraction import FastFrameExtractor

def test_single_video():
    """Test fast extraction on a single video."""
    extractor = FastFrameExtractor()
    
    # Test on one video
    video_id = "youtube_z2IuMTY4KVg"
    
    # Get chunks for this video
    chunks = extractor.load_semantic_chunks()
    video_chunks = [c for c in chunks if c.get('video_id') == video_id]
    
    print(f"Found {len(video_chunks)} chunks for {video_id}")
    
    # Collect timestamps needed
    timestamps = extractor.collect_timestamps_for_video(video_chunks)
    print(f"Timestamps needed: {sorted(timestamps)}")
    
    # Extract frames
    success = extractor.extract_frames_10_second_method(video_id)
    if success:
        # Clean up unused frames
        deleted_count = extractor.cleanup_unused_frames(video_id, timestamps)
        print(f"Success! Deleted {deleted_count} unused frames")
    else:
        print("Failed to extract frames")

if __name__ == "__main__":
    test_single_video() 