#!/usr/bin/env python3
"""
Step 02: Video Downloader (Unified)
Downloads videos from YouTube using yt-dlp with fallback strategies.
Supports both formal presentations and multi-speaker conversations.
"""

import os
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UnifiedVideoDownloader:
    """Unified video downloader supporting multiple pipeline types"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the unified video downloader
        
        Args:
            pipeline_config: Pipeline-specific configuration
            progress_queue: Progress tracking queue
        """
        self.config = pipeline_config
        self.progress_queue = progress_queue
        
        # Pipeline metadata
        self.pipeline_type = pipeline_config.get('pipeline_type', 'unknown')
        self.pipeline_name = pipeline_config.get('pipeline_name', 'Unknown Pipeline')
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Input/output directories
        self.input_dir = Path("step_01_raw")
        self.output_dir = Path("step_02_extracted_playlist_content")
        self.output_dir.mkdir(exist_ok=True)
        
        # Download configuration
        self.download_config = pipeline_config.get('download_config', {})
        self.retry_attempts = self.download_config.get('retry_attempts', 3)
        self.retry_delay = self.download_config.get('retry_delay', 5)
        self.timeout_seconds = self.download_config.get('timeout_seconds', 300)
        
        # Processing statistics
        self.stats = {
            'total_videos': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'skipped_downloads': 0,
            'total_size_mb': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        try:
            # Try to import centralized logging
            from logging_config import setup_logging
            return setup_logging(f'{self.pipeline_type}_video_downloader')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_video_downloader')
    
    def process_all_videos(self):
        """Process all videos in the input directory"""
        if not self.input_dir.exists():
            self.logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find video metadata files
        metadata_files = list(self.input_dir.glob("*_video_metadata.json"))
        self.logger.info(f"Found {len(metadata_files)} video metadata files")
        
        for metadata_file in metadata_files:
            try:
                self.process_video_metadata(metadata_file)
            except Exception as e:
                self.logger.error(f"Failed to process {metadata_file.name}: {e}")
        
        self.log_processing_summary()
    
    def process_video_metadata(self, metadata_file: Path):
        """Process a single video metadata file"""
        try:
            with open(metadata_file, 'r') as f:
                video_metadata = json.load(f)
            
            playlist_name = metadata_file.stem.replace('_video_metadata', '')
            self.logger.info(f"Processing videos for playlist: {playlist_name}")
            
            for video_entry in video_metadata:
                video_id = video_entry.get('video_id')
                if video_id:
                    self.process_single_video(video_id, playlist_name)
            
        except Exception as e:
            self.logger.error(f"Error processing metadata file {metadata_file.name}: {e}")
    
    def process_single_video(self, video_id: str, playlist_name: str):
        """Process a single video download"""
        self.logger.info(f"Processing video: {video_id}")
        
        # Check progress
        if self.progress_queue:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_02_video_downloader') == 'completed':
                self.logger.info(f"Video already downloaded for {video_id}, skipping")
                self.stats['skipped_downloads'] += 1
                return
        
        # Check if video already exists
        video_dir = self.output_dir / f"video_{video_id}"
        if video_dir.exists() and any(video_dir.glob("*.mp4")):
            self.logger.info(f"Video directory already exists for {video_id}, skipping")
            self.stats['skipped_downloads'] += 1
            return
        
        self.stats['total_videos'] += 1
        
        try:
            # Attempt download with fallback strategies
            success = self._download_video_with_fallbacks(video_id, playlist_name)
            
            if success:
                self.stats['successful_downloads'] += 1
                
                # Update progress
                if self.progress_queue:
                    self.progress_queue.update_video_step(video_id, 'step_02_video_downloader', 'completed', {
                        'download_size_mb': self._get_video_size_mb(video_id),
                        'playlist_name': playlist_name
                    })
                
                self.logger.info(f"✅ Successfully downloaded {video_id}")
            else:
                self.stats['failed_downloads'] += 1
                self.logger.error(f"❌ Failed to download {video_id}")
                
        except Exception as e:
            self.stats['failed_downloads'] += 1
            self.logger.error(f"Error processing {video_id}: {e}")
    
    def _download_video_with_fallbacks(self, video_id: str, playlist_name: str) -> bool:
        """Download video using multiple fallback strategies"""
        strategies = [
            self._download_with_yt_dlp_standard,
            self._download_with_yt_dlp_mobile_client,
            self._download_with_yt_dlp_cookies,
            self._download_with_yt_dlp_legacy
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                self.logger.info(f"  Trying download strategy {i+1}/{len(strategies)} for {video_id}")
                success = strategy(video_id, playlist_name)
                
                if success:
                    self.logger.info(f"  ✅ Strategy {i+1} succeeded for {video_id}")
                    return True
                else:
                    self.logger.warning(f"  ⚠️ Strategy {i+1} failed for {video_id}")
                    
            except Exception as e:
                self.logger.warning(f"  ⚠️ Strategy {i+1} failed with error: {e}")
            
            # Wait before trying next strategy
            if i < len(strategies) - 1:
                self.logger.info(f"  Waiting {self.retry_delay}s before next strategy...")
                time.sleep(self.retry_delay)
        
        self.logger.error(f"  ❌ All download strategies failed for {video_id}")
        return False
    
    def _download_with_yt_dlp_standard(self, video_id: str, playlist_name: str) -> bool:
        """Download using standard yt-dlp configuration"""
        try:
            video_dir = self.output_dir / f"video_{video_id}"
            video_dir.mkdir(exist_ok=True)
            
            # Standard yt-dlp command
            cmd = [
                'yt-dlp',
                '--format', 'best[ext=mp4]/best',
                '--output', str(video_dir / '%(id)s.%(ext)s'),
                '--write-info-json',
                '--write-description',
                '--write-thumbnail',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            if result.returncode == 0:
                self._save_download_metadata(video_id, playlist_name, video_dir, 'standard')
                return True
            else:
                self.logger.warning(f"Standard download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Standard download timed out for {video_id}")
            return False
        except Exception as e:
            self.logger.warning(f"Standard download error: {e}")
            return False
    
    def _download_with_yt_dlp_mobile_client(self, video_id: str, playlist_name: str) -> bool:
        """Download using mobile client configuration (often has access to more formats)"""
        try:
            video_dir = self.output_dir / f"video_{video_id}"
            video_dir.mkdir(exist_ok=True)
            
            # Mobile client configuration
            cmd = [
                'yt-dlp',
                '--format', 'best[ext=mp4]/best',
                '--output', str(video_dir / '%(id)s.%(ext)s'),
                '--write-info-json',
                '--write-description',
                '--write-thumbnail',
                '--user-agent', 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            if result.returncode == 0:
                self._save_download_metadata(video_id, playlist_name, video_dir, 'mobile_client')
                return True
            else:
                self.logger.warning(f"Mobile client download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Mobile client download timed out for {video_id}")
            return False
        except Exception as e:
            self.logger.warning(f"Mobile client download error: {e}")
            return False
    
    def _download_with_yt_dlp_cookies(self, video_id: str, playlist_name: str) -> bool:
        """Download using cookies for authenticated access"""
        try:
            # Check if cookies file exists
            cookies_file = Path.home() / ".config" / "youtube-dl" / "cookies.txt"
            if not cookies_file.exists():
                self.logger.info("No cookies file found, skipping cookie-based download")
                return False
            
            video_dir = self.output_dir / f"video_{video_id}"
            video_dir.mkdir(exist_ok=True)
            
            # Cookie-based download
            cmd = [
                'yt-dlp',
                '--format', 'best[ext=mp4]/best',
                '--output', str(video_dir / '%(id)s.%(ext)s'),
                '--write-info-json',
                '--write-description',
                '--write-thumbnail',
                '--cookies', str(cookies_file),
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            if result.returncode == 0:
                self._save_download_metadata(video_id, playlist_name, video_dir, 'cookies')
                return True
            else:
                self.logger.warning(f"Cookie-based download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Cookie-based download timed out for {video_id}")
            return False
        except Exception as e:
            self.logger.warning(f"Cookie-based download error: {e}")
            return False
    
    def _download_with_yt_dlp_legacy(self, video_id: str, playlist_name: str) -> bool:
        """Download using legacy youtube-dl as final fallback"""
        try:
            video_dir = self.output_dir / f"video_{video_id}"
            video_dir.mkdir(exist_ok=True)
            
            # Try youtube-dl as fallback
            cmd = [
                'youtube-dl',
                '--format', 'best[ext=mp4]/best',
                '--output', str(video_dir / '%(id)s.%(ext)s'),
                '--write-info-json',
                '--write-description',
                '--write-thumbnail',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            
            if result.returncode == 0:
                self._save_download_metadata(video_id, playlist_name, video_dir, 'legacy')
                return True
            else:
                self.logger.warning(f"Legacy download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Legacy download timed out for {video_id}")
            return False
        except Exception as e:
            self.logger.warning(f"Legacy download error: {e}")
            return False
    
    def _save_download_metadata(self, video_id: str, playlist_name: str, video_dir: Path, strategy: str):
        """Save metadata about the successful download"""
        try:
            # Find downloaded files
            video_files = list(video_dir.glob("*.mp4"))
            info_files = list(video_dir.glob("*.info.json"))
            desc_files = list(video_dir.glob("*.description"))
            thumb_files = list(video_dir.glob("*.webp"))
            
            # Create download metadata
            download_metadata = {
                'video_id': video_id,
                'playlist_name': playlist_name,
                'pipeline_type': self.pipeline_type,
                'download_strategy': strategy,
                'download_timestamp': datetime.now().isoformat(),
                'files': {
                    'video': [f.name for f in video_files],
                    'info': [f.name for f in info_files],
                    'description': [f.name for f in desc_files],
                    'thumbnail': [f.name for f in thumb_files]
                },
                'total_size_mb': sum(self.get_file_size_mb(f) for f in video_files)
            }
            
            # Save metadata
            metadata_file = video_dir / f"{video_id}_download_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(download_metadata, f, indent=2)
            
            # Update statistics
            self.stats['total_size_mb'] += download_metadata['total_size_mb']
            
        except Exception as e:
            self.logger.warning(f"Failed to save download metadata for {video_id}: {e}")
    
    def _get_video_size_mb(self, video_id: str) -> float:
        """Get the size of downloaded video in MB"""
        video_dir = self.output_dir / f"video_{video_id}"
        if not video_dir.exists():
            return 0.0
        
        total_size = 0.0
        for video_file in video_dir.glob("*.mp4"):
            total_size += self.get_file_size_mb(video_file)
        
        return total_size
    
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
        
        self.logger.info(f"🎉 {self.pipeline_type.upper()} Video Download Summary:")
        self.logger.info(f"  Total videos: {self.stats['total_videos']}")
        self.logger.info(f"  Successful: {self.stats['successful_downloads']}")
        self.logger.info(f"  Failed: {self.stats['failed_downloads']}")
        self.logger.info(f"  Skipped: {self.stats['skipped_downloads']}")
        self.logger.info(f"  Total size: {self.stats['total_size_mb']:.1f} MB")
        self.logger.info(f"  Duration: {duration}")
        
        if self.stats['total_videos'] > 0:
            success_rate = (self.stats['successful_downloads'] / self.stats['total_videos']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        self.logger.info(f"  Output directory: {self.output_dir}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Video Downloader")
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
        
        # Initialize downloader
        downloader = UnifiedVideoDownloader(config)
        
        # Process videos
        downloader.process_all_videos()
        
        print(f"\n✅ Video downloading completed for {args.pipeline_type}")
        
    except Exception as e:
        print(f"❌ Video downloading failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
