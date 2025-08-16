#!/usr/bin/env python3
"""
Step 05: Frame Extraction
Extracts video frames at regular intervals for visual content analysis.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self):
        self.input_dir = Path("step_02_extracted_playlist_content")
        self.output_dir = Path("step_05_frames")
        self.output_dir.mkdir(exist_ok=True)
        
        # Frame extraction parameters
        # Target ~15-20 second intervals (roughly half the expected chunk size of 30-60 seconds)
        self.frame_interval = 18  # Extract frame every 18 seconds for better chunk coverage
        self.frame_quality = 2    # FFmpeg quality (1-31, lower is better)
        self.max_frames = 200     # Increased from 100 to cover full video duration
    
    def process_all_videos(self):
        """Process all videos in the input directory"""
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Load video metadata
        metadata_file = self.input_dir / "download_summary.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file {metadata_file} not found")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Use the new results structure from updated step 2
        videos = metadata.get('results', [])
        logger.info(f"Found {len(videos)} videos to process")
        
        # Track processing statistics
        total_videos = len(videos)
        new_frames = 0
        existing_frames = 0
        failed_frames = 0
        
        for video in videos:
            try:
                result = self.process_single_video(video)
                if result == 'new':
                    new_frames += 1
                elif result == 'existing':
                    existing_frames += 1
                elif result == 'failed':
                    failed_frames += 1
            except Exception as e:
                logger.error(f"Failed to process video {video.get('video_id', 'unknown')}: {e}")
                failed_frames += 1
        
        # Log summary
        logger.info(f"Frame Extraction Summary:")
        logger.info(f"  Total videos: {total_videos}")
        logger.info(f"  New frames: {new_frames}")
        logger.info(f"  Existing frames: {existing_frames}")
        logger.info(f"  Failed: {failed_frames}")
        if total_videos > 0:
            success_rate = ((new_frames + existing_frames) / total_videos * 100)
            logger.info(f"  Success rate: {success_rate:.1f}%")
        else:
            logger.info(f"  Success rate: N/A (no videos to process)")
    
    def process_single_video(self, video_info: Dict[str, Any]):
        """Process a single video"""
        video_id = video_info['video_id']
        video_path = video_info.get('local_path')
        duration = video_info.get('duration', 0)
        
        if not video_path or not Path(video_path).exists():
            logger.warning(f"Video file not found for {video_id}")
            return 'failed'
        
        # Check if frames already exist
        video_frames_dir = self.output_dir / video_id
        frames_summary_file = video_frames_dir / f"{video_id}_frames_summary.json"
        
        if video_frames_dir.exists() and frames_summary_file.exists():
            logger.info(f"Frames already exist for {video_id}, skipping")
            return 'existing'
        
        logger.info(f"Processing video: {video_id} (duration: {duration}s)")
        
        # Create output directory for this video
        video_frames_dir.mkdir(exist_ok=True)
        
        # Extract frames
        frames_info = self.extract_frames(video_path, video_id, video_frames_dir, duration)
        
        # Save frame metadata
        self.save_frame_metadata(frames_info, video_id, video_info)
        
        logger.info(f"Successfully processed video: {video_id}")
        return 'new'
    
    def extract_frames(self, video_path: str, video_id: str, 
                      output_dir: Path, duration: float) -> List[Dict[str, Any]]:
        """Extract frames from video at regular intervals"""
        frames_info = []
        
        try:
            # Calculate frame extraction points
            frame_times = self.calculate_frame_times(duration)
            
            for i, timestamp in enumerate(frame_times):
                if i >= self.max_frames:
                    logger.info(f"Reached maximum frames limit for {video_id}")
                    break
                
                frame_filename = f"{video_id}_frame_{i:03d}_{timestamp:.1f}s.jpg"
                frame_path = output_dir / frame_filename
                
                # Extract frame using FFmpeg
                success = self.extract_single_frame(video_path, timestamp, frame_path)
                
                if success:
                    frame_info = {
                        'frame_id': f"{video_id}_frame_{i:03d}",
                        'filename': frame_filename,
                        'timestamp': timestamp,
                        'file_path': str(frame_path),
                        'file_size': frame_path.stat().st_size if frame_path.exists() else 0,
                        'extraction_order': i
                    }
                    frames_info.append(frame_info)
                    logger.debug(f"Extracted frame: {frame_filename}")
                else:
                    logger.warning(f"Failed to extract frame at {timestamp}s")
            
            logger.info(f"Extracted {len(frames_info)} frames for {video_id}")
            return frames_info
            
        except Exception as e:
            logger.error(f"Failed to extract frames for {video_id}: {e}")
            return []
    
    def calculate_frame_times(self, duration: float) -> List[float]:
        """Calculate timestamps for frame extraction"""
        if duration <= 0:
            # Default to 60 seconds if duration unknown
            duration = 60
        
        frame_times = []
        current_time = 0
        
        while current_time < duration and len(frame_times) < self.max_frames:
            frame_times.append(current_time)
            current_time += self.frame_interval
        
        return frame_times
    
    def extract_single_frame(self, video_path: str, timestamp: float, 
                           output_path: Path) -> bool:
        """Extract a single frame at specified timestamp"""
        try:
            # Use FFmpeg to extract frame
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(timestamp),  # Seek to timestamp
                '-vframes', '1',        # Extract 1 frame
                '-q:v', str(self.frame_quality),  # Quality setting
                '-y',                   # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg failed for timestamp {timestamp}: {result.stderr}")
                return False
            
            # Verify frame was created
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            else:
                logger.warning(f"Frame file not created or empty: {output_path}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"Frame extraction timed out for timestamp {timestamp}")
            return False
        except Exception as e:
            logger.warning(f"Frame extraction failed for timestamp {timestamp}: {e}")
            return False
    
    def save_frame_metadata(self, frames_info: List[Dict[str, Any]], 
                           video_id: str, video_info: Dict[str, Any]):
        """Save frame metadata to file"""
        try:
            # Create frame extraction summary
            extraction_summary = {
                'video_id': video_id,
                'video_metadata': video_info,
                'extraction_parameters': {
                    'frame_interval': self.frame_interval,
                    'frame_quality': self.frame_quality,
                    'max_frames': self.max_frames
                },
                'total_frames_extracted': len(frames_info),
                'frames': frames_info,
                'processing_timestamp': video_info.get('upload_date', '')
            }
            
            # Save to file
            output_file = self.output_dir / f"{video_id}_frames_summary.json"
            with open(output_file, 'w') as f:
                json.dump(extraction_summary, f, indent=2)
            
            logger.info(f"Frame metadata saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save frame metadata: {e}")
    
    def get_video_duration(self, video_path: str) -> float:
        """Get video duration using FFprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                return duration
            else:
                logger.warning(f"Failed to get duration for {video_path}")
                return 0
                
        except Exception as e:
            logger.warning(f"Failed to get video duration: {e}")
            return 0

def main():
    """Main execution function"""
    try:
        extractor = FrameExtractor()
        extractor.process_all_videos()
        logger.info("Frame extraction step completed successfully")
    except Exception as e:
        logger.error(f"Frame extraction step failed: {e}")
        raise

if __name__ == "__main__":
    main()
