#!/usr/bin/env python3
"""
Test the efficient frame extraction on a single video.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Set
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_video_extraction(video_id: str = "youtube_z2IuMTY4KVg"):
    """Test frame extraction on a single video."""
    podcasts_dir = Path("data/ingested/youtube/podcasts - 1 - on -1")
    chunks_file = podcasts_dir / "rag_chunks" / "conversation_chunks.json"
    videos_dir = podcasts_dir / "videos"
    frames_dir = podcasts_dir / "frames"
    
    logger.info(f"🧪 Testing efficient frame extraction on {video_id}")
    
    # Load chunks for this video
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Filter chunks for this video
    video_chunks = [c for c in chunks if c.get('video_id') == video_id]
    logger.info(f"📊 Found {len(video_chunks)} chunks for {video_id}")
    
    # Collect timestamps
    timestamps = set()
    for chunk in video_chunks:
        start_time = chunk.get('start_time', 0)
        if start_time is not None:
            rounded_time = math.floor(start_time / 10) * 10
            timestamps.add(rounded_time)
    
    logger.info(f"⏱️  Timestamps needed: {sorted(list(timestamps))}")
    
    # Check video file exists
    video_path = videos_dir / video_id / "video.mp4"
    if not video_path.exists():
        logger.error(f"❌ Video file not found: {video_path}")
        return False
    
    # Create frames directory
    frames_dir_video = frames_dir / video_id
    frames_dir_video.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    sorted_timestamps = sorted(list(timestamps))
    logger.info(f"🎞️  Extracting {len(sorted_timestamps)} frames...")
    
    successful_extractions = 0
    
    for timestamp in sorted_timestamps:
        frame_path = frames_dir_video / f"frame_{timestamp}.jpg"
        
        # Skip if frame already exists
        if frame_path.exists():
            logger.info(f"   Frame {timestamp}s already exists, skipping")
            successful_extractions += 1
            continue
        
        try:
            # Extract individual frame using -ss (seek)
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-ss', str(timestamp),
                '-vframes', '1',
                '-q:v', '2',  # High quality
                '-y',  # Overwrite existing files
                str(frame_path)
            ]
            
            logger.info(f"   🔧 Extracting frame_{timestamp}.jpg...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and frame_path.exists():
                successful_extractions += 1
                logger.info(f"   ✅ Extracted frame_{timestamp}.jpg")
            else:
                logger.warning(f"   ⚠️  Failed to extract frame_{timestamp}.jpg")
                
        except subprocess.TimeoutExpired:
            logger.warning(f"   ⏰ Timeout extracting frame_{timestamp}.jpg")
        except Exception as e:
            logger.warning(f"   ❌ Error extracting frame_{timestamp}.jpg: {e}")
    
    logger.info(f"📊 {video_id}: {successful_extractions}/{len(sorted_timestamps)} frames extracted")
    
    if successful_extractions == len(sorted_timestamps):
        logger.info("✅ Test extraction successful!")
        return True
    else:
        logger.warning("⚠️  Test extraction partially successful")
        return False

if __name__ == "__main__":
    test_single_video_extraction() 