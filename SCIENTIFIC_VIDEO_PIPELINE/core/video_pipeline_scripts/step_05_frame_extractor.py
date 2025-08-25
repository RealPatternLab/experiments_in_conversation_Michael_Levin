#!/usr/bin/env python3
"""
Step 05: Frame Extractor (Unified)
Extracts frames from videos at configurable intervals.
Supports both formal presentations and multi-speaker conversations.
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

# Load environment variables
load_dotenv()

class UnifiedFrameExtractor:
    """Unified frame extractor supporting multiple pipeline types"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the unified frame extractor
        
        Args:
            pipeline_config: Pipeline-specific configuration
            progress_queue: Progress tracking queue
        """
        self.config = pipeline_config
        self.progress_queue = progress_queue
        
        # Pipeline metadata
        self.pipeline_type = pipeline_config.get('pipeline_type', 'unknown')
        self.pipeline_name = pipeline_config.get('pipeline_name', 'Unknown Pipeline')
        
        # Frame extraction configuration
        self.frame_config = pipeline_config.get('frame_extraction', {})
        self.enabled = self.frame_config.get('enabled', True)
        self.interval_seconds = self.frame_config.get('interval_seconds', 15)
        self.quality = self.frame_config.get('quality', 'high')
        self.max_frames_per_video = self.frame_config.get('max_frames_per_video', 1000)
        self.frame_format = self.frame_config.get('frame_format', 'jpg')
        self.frame_size = self.frame_config.get('frame_size', '1280x720')
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Input/output directories
        self.input_dir = Path("step_02_extracted_playlist_content")
        self.output_dir = Path("step_05_frames")
        self.output_dir.mkdir(exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'total_videos': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'skipped_extractions': 0,
            'total_frames_extracted': 0,
            'total_size_mb': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        try:
            # Try to import centralized logging
            from logging_config import setup_logging
            return setup_logging(f'{self.pipeline_type}_frame_extractor')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_frame_extractor')
    
    def process_all_videos(self):
        """Process all videos in the input directory"""
        if not self.enabled:
            self.logger.info("Frame extraction is disabled in configuration, skipping")
            return
        
        if not self.input_dir.exists():
            self.logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find video directories
        video_dirs = [d for d in self.input_dir.iterdir() if d.is_dir() and d.name.startswith('video_')]
        self.logger.info(f"Found {len(video_dirs)} video directories")
        
        self.stats['total_videos'] = len(video_dirs)
        
        for video_dir in video_dirs:
            try:
                video_id = video_dir.name.replace('video_', '')
                self.process_single_video(video_id, video_dir)
            except Exception as e:
                self.logger.error(f"Failed to process {video_dir.name}: {e}")
        
        self.log_processing_summary()
    
    def process_single_video(self, video_id: str, video_dir: Path):
        """Process a single video for frame extraction"""
        self.logger.info(f"Processing video: {video_id}")
        
        # Check progress
        if self.progress_queue:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_05_frame_extractor') == 'completed':
                self.logger.info(f"Frame extraction already completed for {video_id}, skipping")
                self.stats['skipped_extractions'] += 1
                return
        
        # Check if frames already exist
        frames_dir = self.output_dir / video_id
        if frames_dir.exists() and any(frames_dir.glob(f"*.{self.frame_format}")):
            self.logger.info(f"Frames already exist for {video_id}, skipping")
            self.stats['skipped_extractions'] += 1
            return
        
        try:
            # Find video file
            video_file = self._find_video_file(video_dir)
            if not video_file:
                self.logger.warning(f"No video file found for {video_id}")
                return
            
            # Get video duration
            duration = self._get_video_duration(video_file)
            if duration <= 0:
                self.logger.warning(f"Could not determine duration for {video_id}")
                return
            
            # Extract frames
            success = self._extract_frames(video_id, video_file, duration)
            
            if success:
                self.stats['successful_extractions'] += 1
                
                # Update progress
                if self.progress_queue:
                    self.progress_queue.update_video_step(video_id, 'step_05_frame_extractor', 'completed', {
                        'frames_extracted': self._count_frames(video_id),
                        'interval_seconds': self.interval_seconds,
                        'frame_format': self.frame_format
                    })
                
                self.logger.info(f"✅ Successfully extracted frames for {video_id}")
            else:
                self.stats['failed_extractions'] += 1
                self.logger.error(f"❌ Failed to extract frames for {video_id}")
                
        except Exception as e:
            self.stats['failed_extractions'] += 1
            self.logger.error(f"Error processing {video_id}: {e}")
    
    def _find_video_file(self, video_dir: Path) -> Optional[Path]:
        """Find the video file in the video directory"""
        # Look for common video formats
        video_extensions = ['*.mp4', '*.webm', '*.mkv', '*.avi']
        
        for extension in video_extensions:
            video_files = list(video_dir.glob(extension))
            if video_files:
                return video_files[0]  # Return first match
        
        return None
    
    def _get_video_duration(self, video_file: Path) -> float:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                str(video_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                self.logger.debug(f"Video duration: {duration:.2f} seconds")
                return duration
            else:
                self.logger.warning(f"Failed to get duration: {result.stderr}")
                return 0.0
                
        except subprocess.TimeoutExpired:
            self.logger.warning("ffprobe timed out")
            return 0.0
        except Exception as e:
            self.logger.warning(f"Error getting video duration: {e}")
            return 0.0
    
    def _extract_frames(self, video_id: str, video_file: Path, duration: float) -> bool:
        """Extract frames using ffmpeg"""
        try:
            # Create frames directory
            frames_dir = self.output_dir / video_id
            frames_dir.mkdir(exist_ok=True)
            
            # Calculate frame extraction parameters
            total_frames = int(duration / self.interval_seconds)
            if total_frames > self.max_frames_per_video:
                total_frames = self.max_frames_per_video
                self.logger.info(f"Limiting frames to {total_frames} (max allowed)")
            
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', str(video_file),
                '-vf', f'fps=1/{self.interval_seconds}',
                '-frame_pts', '1',
                '-vsync', '0',
                '-f', 'image2',
                '-q:v', self._get_quality_setting(),
                '-vf', f'scale={self.frame_size}',
                str(frames_dir / f'frame_%04d_{self.interval_seconds}s.{self.frame_format}'),
                '-y'  # Overwrite existing files
            ]
            
            self.logger.info(f"Extracting {total_frames} frames from {video_id} (every {self.interval_seconds}s)")
            
            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._calculate_timeout(duration)
            )
            
            if result.returncode == 0:
                # Count extracted frames
                frame_count = self._count_frames(video_id)
                self.stats['total_frames_extracted'] += frame_count
                
                # Create frame summary
                self._create_frame_summary(video_id, frames_dir, frame_count, duration)
                
                self.logger.info(f"✅ Extracted {frame_count} frames for {video_id}")
                return True
            else:
                self.logger.error(f"ffmpeg failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Frame extraction timed out for {video_id}")
            return False
        except Exception as e:
            self.logger.error(f"Frame extraction error: {e}")
            return False
    
    def _get_quality_setting(self) -> str:
        """Get ffmpeg quality setting based on configuration"""
        quality_map = {
            'low': '5',
            'medium': '3',
            'high': '1',
            'very_high': '0'
        }
        return quality_map.get(self.quality, '3')
    
    def _calculate_timeout(self, duration: float) -> int:
        """Calculate timeout for frame extraction"""
        # Base timeout: 30 seconds + 2 seconds per minute of video
        base_timeout = 30
        duration_minutes = duration / 60
        timeout = base_timeout + int(duration_minutes * 2)
        
        # Cap at 10 minutes
        return min(timeout, 600)
    
    def _count_frames(self, video_id: str) -> int:
        """Count extracted frames for a video"""
        frames_dir = self.output_dir / video_id
        if not frames_dir.exists():
            return 0
        
        frame_files = list(frames_dir.glob(f"*.{self.frame_format}"))
        return len(frame_files)
    
    def _create_frame_summary(self, video_id: str, frames_dir: Path, frame_count: int, duration: float):
        """Create summary of extracted frames"""
        try:
            # Calculate total size
            total_size = 0.0
            for frame_file in frames_dir.glob(f"*.{self.frame_format}"):
                total_size += self.get_file_size_mb(frame_file)
            
            # Create summary data
            summary = {
                'video_id': video_id,
                'pipeline_type': self.pipeline_type,
                'extraction_timestamp': datetime.now().isoformat(),
                'frame_count': frame_count,
                'interval_seconds': self.interval_seconds,
                'frame_format': self.frame_format,
                'frame_size': self.frame_size,
                'quality': self.quality,
                'video_duration': duration,
                'total_size_mb': total_size,
                'frames': []
            }
            
            # Add frame details
            for frame_file in sorted(frames_dir.glob(f"*.{self.frame_format}")):
                frame_info = {
                    'filename': frame_file.name,
                    'size_mb': self.get_file_size_mb(frame_file),
                    'timestamp': self._extract_timestamp_from_filename(frame_file.name)
                }
                summary['frames'].append(frame_info)
            
            # Save summary
            summary_file = self.output_dir / f"{video_id}_frames_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Update statistics
            self.stats['total_size_mb'] += total_size
            
        except Exception as e:
            self.logger.warning(f"Failed to create frame summary for {video_id}: {e}")
    
    def _extract_timestamp_from_filename(self, filename: str) -> float:
        """Extract timestamp from frame filename"""
        try:
            # Expected format: frame_0001_15s.jpg
            import re
            match = re.search(r'_(\d+)s\.', filename)
            if match:
                return float(match.group(1))
        except:
            pass
        return 0.0
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def log_processing_summary(self):
        """Log a summary of processing statistics"""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info(f"🎉 {self.pipeline_type.upper()} Frame Extraction Summary:")
        self.logger.info(f"  Total videos: {self.stats['total_videos']}")
        self.logger.info(f"  Successful: {self.stats['successful_extractions']}")
        self.logger.info(f"  Failed: {self.stats['failed_extractions']}")
        self.logger.info(f"  Skipped: {self.stats['skipped_extractions']}")
        self.logger.info(f"  Total frames extracted: {self.stats['total_frames_extracted']}")
        self.logger.info(f"  Total size: {self.stats['total_size_mb']:.1f} MB")
        self.logger.info(f"  Frame interval: {self.interval_seconds}s")
        self.logger.info(f"  Frame format: {self.frame_format}")
        self.logger.info(f"  Frame size: {self.frame_size}")
        self.logger.info(f"  Quality: {self.quality}")
        self.logger.info(f"  Duration: {duration}")
        
        if self.stats['total_videos'] > 0:
            success_rate = (self.stats['successful_extractions'] / self.stats['total_videos']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        if self.stats['total_frames_extracted'] > 0:
            avg_frames_per_video = self.stats['total_frames_extracted'] / self.stats['successful_extractions']
            self.logger.info(f"  Average frames per video: {avg_frames_per_video:.1f}")
        
        self.logger.info(f"  Output directory: {self.output_dir}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Frame Extractor")
    parser.add_argument("--pipeline-type", required=True, 
                       choices=["formal_presentations", "conversations_1_on_1", "conversations_1_on_2"],
                       help="Pipeline type to process")
    parser.add_argument("--config-dir", default="core/pipeline_configs",
                       help="Directory containing pipeline configurations")
    
    args = parser.parse_args()
    
    try:
        # Load pipeline configuration
        from core.config_loader import load_pipeline_config
        config = load_pipeline_config(args.pipeline_type, args.config_dir)
        
        # Initialize frame extractor
        extractor = UnifiedFrameExtractor(config)
        
        # Process videos
        extractor.process_all_videos()
        
        print(f"\n✅ Frame extraction completed for {args.pipeline_type}")
        
    except Exception as e:
        print(f"❌ Frame extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
