#!/usr/bin/env python3
"""
Test the smart frame extraction on a single video.
"""

from tools.smart_frame_extraction import SmartFrameExtractor

def test_single_video():
    """Test smart extraction on a single video."""
    extractor = SmartFrameExtractor()
    
    # Test on one video
    video_id = "youtube_z2IuMTY4KVg"
    
    # Get chunks for this video
    chunks = extractor.load_semantic_chunks()
    video_chunks = [c for c in chunks if c.get('video_id') == video_id]
    
    print(f"Found {len(video_chunks)} chunks for {video_id}")
    
    # Collect exact timestamps from frame_path
    timestamps = extractor.collect_exact_timestamps_for_video(video_chunks)
    print(f"Timestamps needed: {sorted(timestamps)}")
    
    # Extract frames
    success = extractor.extract_frames_for_video(video_id, timestamps)
    if success:
        print("Success! All frames extracted")
    else:
        print("Failed to extract frames")

if __name__ == "__main__":
    test_single_video() 