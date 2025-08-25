#!/usr/bin/env python3
"""
Step 01: Playlist Processor (Unified)
Processes YouTube playlists to extract video metadata and prepare for processing.
Supports both formal presentations and multi-speaker conversations.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UnifiedPlaylistProcessor:
    """Unified playlist processor supporting multiple pipeline types"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the unified playlist processor
        
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
        self.output_dir = Path("step_01_raw")
        self.output_dir.mkdir(exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'total_playlists': 0,
            'total_videos': 0,
            'videos_processed': 0,
            'failed_videos': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        try:
            # Try to import centralized logging
            from logging_config import setup_logging
            return setup_logging(f'{self.pipeline_type}_playlist_processor')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_playlist_processor')
    
    def process_playlists(self):
        """Process all playlists in the input directory"""
        if not self.input_dir.exists():
            self.logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find playlist files
        playlist_files = list(self.input_dir.glob("*.txt")) + list(self.input_dir.glob("*.txt"))
        self.logger.info(f"Found {len(playlist_files)} playlist files")
        
        self.stats['total_playlists'] = len(playlist_files)
        
        for playlist_file in playlist_files:
            try:
                self.process_single_playlist(playlist_file)
            except Exception as e:
                self.logger.error(f"Failed to process {playlist_file.name}: {e}")
        
        self.log_processing_summary()
    
    def process_single_playlist(self, playlist_file: Path):
        """Process a single playlist file"""
        playlist_name = playlist_file.stem
        self.logger.info(f"Processing playlist: {playlist_name}")
        
        try:
            # Read playlist content
            with open(playlist_file, 'r') as f:
                playlist_content = f.read().strip()
            
            # Extract video IDs
            video_ids = self._extract_video_ids(playlist_content)
            
            if not video_ids:
                self.logger.warning(f"No video IDs found in {playlist_name}")
                return
            
            self.logger.info(f"Found {len(video_ids)} videos in playlist")
            self.stats['total_videos'] += len(video_ids)
            
            # Create playlist metadata
            playlist_metadata = {
                'playlist_name': playlist_name,
                'pipeline_type': self.pipeline_type,
                'pipeline_name': self.pipeline_name,
                'total_videos': len(video_ids),
                'video_ids': video_ids,
                'created_at': datetime.now().isoformat(),
                'source_file': str(playlist_file)
            }
            
            # Save playlist metadata
            metadata_file = self.output_dir / f"{playlist_name}_playlist_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(playlist_metadata, f, indent=2)
            
            # Create video metadata entries
            video_metadata = []
            for video_id in video_ids:
                video_entry = self._create_video_metadata(video_id, playlist_name)
                video_metadata.append(video_entry)
            
            # Save video metadata
            videos_file = self.output_dir / f"{playlist_name}_video_metadata.json"
            with open(videos_file, 'w') as f:
                json.dump(video_metadata, f, indent=2)
            
            # Create playlist summary
            summary = {
                'playlist_name': playlist_name,
                'pipeline_type': self.pipeline_type,
                'total_videos': len(video_ids),
                'processing_status': 'ready',
                'created_at': datetime.now().isoformat()
            }
            
            summary_file = self.output_dir / f"{playlist_name}_playlist_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"✅ Successfully processed playlist {playlist_name}")
            self.stats['videos_processed'] += len(video_ids)
            
        except Exception as e:
            self.logger.error(f"Error processing playlist {playlist_name}: {e}")
    
    def _extract_video_ids(self, playlist_content: str) -> List[str]:
        """Extract video IDs from playlist content"""
        video_ids = []
        
        # Common YouTube URL patterns
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'([a-zA-Z0-9_-]{11})',  # Just the ID
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, playlist_content)
            for match in matches:
                if match not in video_ids:
                    video_ids.append(match)
        
        # Remove duplicates while preserving order
        unique_ids = []
        for video_id in video_ids:
            if video_id not in unique_ids:
                unique_ids.append(video_id)
        
        return unique_ids
    
    def _create_video_metadata(self, video_id: str, playlist_name: str) -> Dict[str, Any]:
        """Create metadata entry for a single video"""
        return {
            'video_id': video_id,
            'playlist_name': playlist_name,
            'pipeline_type': self.pipeline_type,
            'processing_status': 'pending',
            'steps_completed': [],
            'created_at': datetime.now().isoformat(),
            'metadata': {
                'title': None,
                'duration': None,
                'upload_date': None,
                'channel': None,
                'description': None
            }
        }
    
    def log_processing_summary(self):
        """Log a summary of processing statistics"""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info(f"🎉 {self.pipeline_type.upper()} Playlist Processing Summary:")
        self.logger.info(f"  Total playlists: {self.stats['total_playlists']}")
        self.logger.info(f"  Total videos: {self.stats['total_videos']}")
        self.logger.info(f"  Videos processed: {self.stats['videos_processed']}")
        self.logger.info(f"  Failed: {self.stats['failed_videos']}")
        self.logger.info(f"  Duration: {duration}")
        
        if self.stats['total_videos'] > 0:
            success_rate = (self.stats['videos_processed'] / self.stats['total_videos']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        self.logger.info(f"  Output directory: {self.output_dir}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Playlist Processor")
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
        
        # Initialize processor
        processor = UnifiedPlaylistProcessor(config)
        
        # Process playlists
        processor.process_playlists()
        
        print(f"\n✅ Playlist processing completed for {args.pipeline_type}")
        
    except Exception as e:
        print(f"❌ Playlist processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
