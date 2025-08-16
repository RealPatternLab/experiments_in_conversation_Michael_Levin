#!/usr/bin/env python3
"""
Step 01: Playlist Processing
Processes YouTube playlist URLs and extracts metadata for all videos within the playlist.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('playlist_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlaylistProcessor:
    def __init__(self):
        self.input_dir = Path("step_01_raw")
        self.output_dir = Path("step_01_raw")  # Same directory for now
        
        # YouTube API configuration (optional, for enhanced metadata)
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        
        # Playlist processing parameters
        self.max_videos_per_playlist = 100  # Limit for processing
        
    def process_playlists(self):
        """Process all playlist files in the input directory"""
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find playlist files
        playlist_files = list(self.input_dir.glob("*playlist*.txt"))
        if not playlist_files:
            logger.error("No playlist files found")
            return
        
        for playlist_file in playlist_files:
            try:
                self.process_single_playlist(playlist_file)
            except Exception as e:
                logger.error(f"Failed to process {playlist_file.name}: {e}")
    
    def process_single_playlist(self, playlist_file: Path):
        """Process a single playlist file"""
        logger.info(f"Processing playlist file: {playlist_file.name}")
        
        # Read playlist URLs
        with open(playlist_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Found {len(urls)} URLs to process")
        
        # Process each URL
        processed_playlists = []
        total_videos = 0
        
        for url in urls:
            try:
                playlist_data = self.process_playlist_url(url)
                if playlist_data:
                    processed_playlists.append(playlist_data)
                    total_videos += playlist_data.get('total_videos_found', 0)
            except Exception as e:
                logger.error(f"Failed to process URL {url}: {e}")
        
        # Create summary
        summary = {
            'total_urls': len(urls),
            'valid_urls': len(processed_playlists),
            'invalid_urls': len(urls) - len(processed_playlists),
            'total_videos_found': total_videos,
            'playlists': processed_playlists,
            'processing_timestamp': self.get_timestamp()
        }
        
        # Save results
        self.save_results(summary, playlist_file.stem)
        
        logger.info(f"Processed {len(processed_playlists)} playlists with {total_videos} total videos")
    
    def process_playlist_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Process a single playlist URL"""
        try:
            # Parse URL
            parsed_url = urlparse(url)
            
            if 'youtube.com' not in parsed_url.netloc:
                logger.warning(f"Not a YouTube URL: {url}")
                return None
            
            # Extract playlist ID
            playlist_id = self.extract_playlist_id(url)
            if not playlist_id:
                logger.warning(f"Could not extract playlist ID from: {url}")
                return None
            
            logger.info(f"Processing playlist: {playlist_id}")
            
            # Get playlist metadata
            playlist_metadata = self.get_playlist_metadata(playlist_id)
            if not playlist_metadata:
                return None
            
            # Get videos in playlist
            videos = self.get_playlist_videos(playlist_id)
            
            # Create playlist data
            playlist_data = {
                'url': url,
                'is_valid': True,
                'error': None,
                'playlist_id': playlist_id,
                'domain': parsed_url.netloc,
                'scheme': parsed_url.scheme,
                'query_params': parse_qs(parsed_url.query),
                'playlist_metadata': playlist_metadata,
                'videos': videos,
                'total_videos_found': len(videos)
            }
            
            return playlist_data
            
        except Exception as e:
            logger.error(f"Failed to process playlist URL {url}: {e}")
            return {
                'url': url,
                'is_valid': False,
                'error': str(e),
                'playlist_id': None,
                'videos': [],
                'total_videos_found': 0
            }
    
    def extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from YouTube URL"""
        # Handle different YouTube URL formats
        patterns = [
            r'list=([a-zA-Z0-9_-]+)',  # Standard playlist parameter
            r'/playlist\?list=([a-zA-Z0-9_-]+)',  # Playlist URL
            r'/watch\?.*list=([a-zA-Z0-9_-]+)',  # Video with playlist
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_playlist_metadata(self, playlist_id: str) -> Dict[str, Any]:
        """Get basic playlist metadata"""
        # For now, return basic metadata
        # In a full implementation, you could use YouTube API for enhanced data
        return {
            'playlist_id': playlist_id,
            'title': f"Playlist {playlist_id}",
            'description': "Playlist description",
            'channel_title': "Channel name",
            'published_at': "",
            'video_count': 0,
            'privacy_status': "public"
        }
    
    def get_playlist_videos(self, playlist_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get videos in the playlist"""
        # This is a simplified version
        # In a full implementation, you would use YouTube API or yt-dlp
        
        # For now, return placeholder data
        # You would typically use yt-dlp to get actual video information
        videos = []
        
        # Example of what yt-dlp would return:
        # yt-dlp --flat-playlist --get-id --get-title --get-duration "https://www.youtube.com/playlist?list=" + playlist_id
        
        # Placeholder video entry
        videos.append({
            'video_id': 'placeholder_id',
            'title': 'Video Title',
            'description': 'Video description',
            'channel_title': 'Channel name',
            'published_at': '',
            'playlist_position': 1,
            'duration': 0,
            'view_count': 0,
            'like_count': 0,
            'category_id': [],
            'default_language': None,
            'default_audio_language': None
        })
        
        return videos
    
    def save_results(self, summary: Dict[str, Any], filename_prefix: str):
        """Save processing results"""
        try:
            # Save detailed results
            output_file = self.output_dir / f"{filename_prefix}_and_video_metadata.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Results saved: {output_file}")
            
            # Save summary
            summary_file = self.output_dir / f"{filename_prefix}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump({
                    'total_playlists': summary['valid_urls'],
                    'total_videos': summary['total_videos_found'],
                    'processing_timestamp': summary['processing_timestamp']
                }, f, indent=2)
            
            logger.info(f"Summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()

def main():
    """Main execution function"""
    try:
        processor = PlaylistProcessor()
        processor.process_playlists()
        logger.info("Playlist processing step completed successfully")
    except Exception as e:
        logger.error(f"Playlist processing step failed: {e}")
        raise

if __name__ == "__main__":
    main()
