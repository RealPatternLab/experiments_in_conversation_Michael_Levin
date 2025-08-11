#!/usr/bin/env python3
"""
Test the original 10-second frame extraction method for speed comparison.
"""

import subprocess
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_10_second_extraction(video_id: str = "youtube_z2IuMTY4KVg"):
    """Test the original 10-second frame extraction method."""
    podcasts_dir = Path("data/ingested/youtube/podcasts - 1 - on -1")
    video_path = podcasts_dir / "videos" / video_id / "video.mp4"
    frames_dir = podcasts_dir / "frames" / video_id
    
    logger.info(f"🧪 Testing 10-second frame extraction on {video_id}")
    
    if not video_path.exists():
        logger.error(f"❌ Video file not found: {video_path}")
        return False
    
    # Create frames directory
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Use the original fast method: extract frames every 10 seconds
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", "fps=1/10",  # 1 frame every 10 seconds
            "-y",  # Overwrite existing files
            str(frames_dir / "frame_%d.jpg")  # %d will be the frame number
        ]
        
        logger.info("🚀 Starting 10-second frame extraction...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            # Rename frames to use actual timestamps (like the original script)
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            for i, frame_file in enumerate(frame_files):
                # Calculate actual timestamp: frame number * 10 seconds
                timestamp = i * 10
                new_name = f"frame_{timestamp}.jpg"
                frame_file.rename(frames_dir / new_name)
            
            elapsed_time = time.time() - start_time
            frame_count = len(frame_files)
            
            logger.info(f"✅ Successfully extracted {frame_count} frames in {elapsed_time:.1f} seconds")
            logger.info(f"📊 Speed: {frame_count/elapsed_time:.1f} frames/second")
            
            # Show first few frames
            logger.info(f"📸 First frames: {[f.name for f in frame_files[:5]]}")
            
            return True
        else:
            logger.error(f"❌ ffmpeg failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ Extraction timed out after 2 minutes")
        return False
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        return False

if __name__ == "__main__":
    test_10_second_extraction() 