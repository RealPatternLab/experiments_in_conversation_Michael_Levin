#!/usr/bin/env python3
"""
Step 05: Frame Extraction
Extracts video frames at regular intervals for visual content analysis.
"""

import os
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from pipeline_progress_queue import get_progress_queue

# Load environment variables
load_dotenv()

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('frame_extraction')

class FrameExtractor:
    def __init__(self, progress_queue=None):
        self.input_dir = Path("step_02_extracted_playlist_content")
        self.output_dir = Path("step_05_frames")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = progress_queue
        
        # Frame extraction parameters
        # Target ~15-20 second intervals (roughly half the expected chunk size of 30-60 seconds)
        self.frame_interval = 18  # Extract frame every 18 seconds for better chunk coverage
        self.frame_quality = 2    # FFmpeg quality (1-31, lower is better)
        self.max_frames = 500     # Increased from 100 to cover full video duration
    
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
        
        # Handle both old and new metadata structures
        videos = metadata.get('results', [])  # New structure
        if not videos:
            videos = metadata.get('videos', [])  # Fallback to old structure
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
        
        # Handle different metadata structures
        video_path = None
        if 'local_path' in video_info:
            video_path = video_info['local_path']
        elif 'download_result' in video_info and 'video_file' in video_info['download_result']:
            video_path = video_info['download_result']['video_file']
        
        # Get duration from different possible locations
        duration = 0
        
        # First try to get duration from the current video_info
        if 'duration' in video_info:
            duration = video_info['duration']
        elif 'download_result' in video_info and 'additional_metadata' in video_info['download_result']:
            duration = video_info['download_result']['additional_metadata'].get('duration', 0)
        
        # If duration is still 0, try to get it from the playlist metadata
        if duration == 0:
            try:
                playlist_metadata_file = Path("step_01_raw/playlist_and_video_metadata.json")
                if playlist_metadata_file.exists():
                    with open(playlist_metadata_file, 'r') as f:
                        playlist_data = json.load(f)
                    
                    # Find this video in the playlist data
                    for playlist in playlist_data.get('playlists', []):
                        for video in playlist.get('videos', []):
                            if video.get('video_id') == video_id:
                                duration = video.get('duration', 0)
                                logger.info(f"Found duration {duration}s for {video_id} in playlist metadata")
                                break
                        if duration > 0:
                            break
            except Exception as e:
                logger.warning(f"Failed to read duration from playlist metadata: {e}")
        
        # If still no duration, try to extract it from the video file using FFmpeg
        if duration == 0:
            try:
                import subprocess
                cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    duration = float(result.stdout.strip())
                    logger.info(f"Extracted duration {duration}s for {video_id} from video file")
            except Exception as e:
                logger.warning(f"Failed to extract duration from video file: {e}")
        
        if not video_path or not Path(video_path).exists():
            logger.warning(f"Video file not found for {video_id}")
            return 'failed'
        
        # Check if frames already exist using progress queue
        if self.progress_queue:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_05_frame_extraction') == 'completed':
                logger.info(f"Frames already completed for {video_id} (progress queue), skipping")
                return 'existing'
        
        # Fallback: Check if frames already exist and are complete (for backward compatibility)
        video_frames_dir = self.output_dir / video_id
        frames_summary_file = self.output_dir / f"{video_id}_frames_summary.json"
        
        logger.info(f"Checking skip logic for {video_id}:")
        logger.info(f"  Current working directory: {Path.cwd()}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Video frames dir: {video_frames_dir}")
        logger.info(f"  Frames summary file: {frames_summary_file}")
        logger.info(f"  Video frames dir exists: {video_frames_dir.exists()}")
        logger.info(f"  Frames summary file exists: {frames_summary_file.exists()}")
        logger.info(f"  Video frames dir absolute: {video_frames_dir.absolute()}")
        logger.info(f"  Frames summary file absolute: {frames_summary_file.absolute()}")
        
        if video_frames_dir.exists() and frames_summary_file.exists():
            # Check if summary file is complete and frames actually exist
            try:
                with open(frames_summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                expected_frame_count = summary_data.get('total_frames_extracted', 0)
                actual_frame_files = list(video_frames_dir.glob(f"{video_id}_frame_*.jpg"))
                
                logger.info(f"  Expected frame count: {expected_frame_count}")
                logger.info(f"  Actual frame files found: {len(actual_frame_files)}")
                
                if expected_frame_count > 0 and len(actual_frame_files) >= expected_frame_count:
                    logger.info(f"Frames already exist for {video_id} ({expected_frame_count} frames), skipping")
                    return 'existing'
                else:
                    logger.info(f"Frames directory exists but incomplete for {video_id} (expected: {expected_frame_count}, found: {len(actual_frame_files)}), will re-process")
            except Exception as e:
                logger.warning(f"Error reading frames summary for {video_id}: {e}, will re-process")
        else:
            logger.info(f"  Skip conditions not met, will process video")
        
        logger.info(f"Processing video: {video_id} (duration: {duration}s)")
        
        # Create output directory for this video
        video_frames_dir.mkdir(exist_ok=True)
        
        # Extract frames
        frames_info = self.extract_frames(video_path, video_id, video_frames_dir, duration)
        
        # Save frame metadata only if we actually processed frames
        if frames_info:
            self.save_frame_metadata(frames_info, video_id, video_info)
            
            # Update progress queue
            if self.progress_queue:
                self.progress_queue.update_video_step_status(
                    video_id,
                    'step_05_frame_extraction',
                    'completed',
                    metadata={
                        'frames_summary_file': str(frames_summary_file),
                        'video_frames_dir': str(video_frames_dir),
                        'total_frames_extracted': len(frames_info),
                        'completed_at': datetime.now().isoformat()
                    }
                )
                logger.info(f"📊 Progress queue updated: step 5 completed for {video_id}")
            
            logger.info(f"Successfully processed video: {video_id}")
            return 'new'
        else:
            logger.warning(f"No frames were extracted for {video_id}")
            return 'failed'
    
    def extract_frames(self, video_path: str, video_id: str, 
                      output_dir: Path, duration: float) -> List[Dict[str, Any]]:
        """Extract frames from video at regular intervals"""
        frames_info = []
        
        try:
            # Calculate frame extraction points
            frame_times = self.calculate_frame_times(duration)
            
            logger.info(f"Calculated {len(frame_times)} frame extraction points for {video_id} (duration: {duration}s)")
            
            for i, timestamp in enumerate(frame_times):
                if i >= self.max_frames:
                    logger.info(f"Reached maximum frames limit for {video_id}")
                    break
                
                frame_filename = f"{video_id}_frame_{i:03d}_{timestamp:.1f}s.jpg"
                frame_path = output_dir / frame_filename
                
                # Skip if frame already exists
                if frame_path.exists():
                    logger.debug(f"Frame already exists: {frame_filename}, skipping extraction")
                    frame_info = {
                        'frame_id': f"{video_id}_frame_{i:03d}",
                        'filename': frame_filename,
                        'timestamp': timestamp,
                        'file_path': str(frame_path),
                        'file_size': frame_path.stat().st_size,
                        'extraction_order': i,
                        'status': 'existing'
                    }
                    frames_info.append(frame_info)
                    continue
                
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
        # Initialize progress queue
        progress_queue = get_progress_queue()
        logger.info("✅ Progress queue initialized")
        
        extractor = FrameExtractor(progress_queue)
        extractor.process_all_videos()
        logger.info("Frame extraction step completed successfully")
    except Exception as e:
        logger.error(f"Frame extraction step failed: {e}")
        raise

if __name__ == "__main__":
    main()
