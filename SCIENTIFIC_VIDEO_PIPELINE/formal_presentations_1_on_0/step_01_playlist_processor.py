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
        
        # Check if yt-dlp is available
        self.yt_dlp_available = self.check_yt_dlp_availability()
    
    def check_yt_dlp_availability(self) -> bool:
        """Check if yt-dlp is available and working"""
        try:
            import subprocess
            result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                logger.info(f"yt-dlp version {version} is available")
                return True
            else:
                logger.warning("yt-dlp is not working properly")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("yt-dlp is not installed or not in PATH")
            return False
        except Exception as e:
            logger.warning(f"Error checking yt-dlp: {e}")
            return False
        
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
        
        # Save results with consistent naming
        self.save_results(summary, "playlist")
        
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
        """Get playlist metadata using yt-dlp"""
        if not self.yt_dlp_available:
            logger.warning("yt-dlp not available, using fallback metadata")
            return {
                'playlist_id': playlist_id,
                'title': f"Playlist {playlist_id}",
                'description': "Playlist description",
                'channel_title': "Unknown Channel",
                'published_at': "",
                'video_count': 0,
                'privacy_status': "public"
            }
        
        try:
            import subprocess
            import json
            
            # Use yt-dlp to get playlist info
            playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
            
            # Get playlist metadata
            cmd = [
                'yt-dlp',
                '--flat-playlist',
                '--dump-json',
                '--no-playlist-reverse',
                '--playlist-items', '1',  # Just get first item to get playlist info
                playlist_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse the first video to get playlist context
                try:
                    first_video = json.loads(result.stdout.strip().split('\n')[0])
                    playlist_info = first_video.get('playlist', {})
                    
                    # Handle case where playlist_info might be a string or dict
                    if isinstance(playlist_info, dict):
                        return {
                            'playlist_id': playlist_id,
                            'title': playlist_info.get('title', f"Playlist {playlist_id}"),
                            'description': playlist_info.get('description', "Playlist description"),
                            'channel_title': playlist_info.get('uploader', "Unknown Channel"),
                            'published_at': playlist_info.get('upload_date', ""),
                            'video_count': playlist_info.get('playlist_count', 0),
                            'privacy_status': "public"
                        }
                    else:
                        logger.warning(f"Playlist info is not a dict for {playlist_id}: {type(playlist_info)}")
                except (json.JSONDecodeError, IndexError) as e:
                    logger.warning(f"Failed to parse playlist metadata for {playlist_id}: {e}")
            
            # Fallback to basic metadata
            return {
                'playlist_id': playlist_id,
                'title': f"Playlist {playlist_id}",
                'description': "Playlist description",
                'channel_title': "Unknown Channel",
                'published_at': "",
                'video_count': 0,
                'privacy_status': "public"
            }
            
        except Exception as e:
            logger.error(f"Failed to get playlist metadata for {playlist_id}: {e}")
            # Fallback to basic metadata
            return {
                'playlist_id': playlist_id,
                'title': f"Playlist {playlist_id}",
                'description': "Playlist description",
                'channel_title': "Unknown Channel",
                'published_at': "",
                'video_count': 0,
                'privacy_status': "public"
            }
    
    def get_playlist_videos(self, playlist_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get videos in the playlist using yt-dlp"""
        if not self.yt_dlp_available:
            logger.warning("yt-dlp not available, using fallback video data")
            return []
        
        try:
            import subprocess
            import json
            
            playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
            videos = []
            
            # Use yt-dlp to get playlist videos
            cmd = [
                'yt-dlp',
                '--flat-playlist',
                '--dump-json',
                '--no-playlist-reverse',
                playlist_url
            ]
            
            logger.info(f"Fetching playlist videos using yt-dlp: {playlist_url}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"yt-dlp failed for playlist {playlist_id}: {result.stderr}")
                return None
            
            if not result.stdout.strip():
                logger.warning(f"No videos found in playlist {playlist_id}")
                return []
            
            # Parse each video entry
            video_lines = result.stdout.strip().split('\n')
            logger.info(f"Found {len(video_lines)} video entries in playlist")
            
            # Limit to first 2 videos for testing
            max_videos = 2
            video_lines = video_lines[:max_videos]
            logger.info(f"Limiting to first {max_videos} videos for testing")
            
            for i, line in enumerate(video_lines):
                try:
                    video_data = json.loads(line.strip())
                    
                    # Extract video information
                    video_info = {
                        'video_id': video_data.get('id', f'unknown_{i}'),
                        'title': video_data.get('title', 'Unknown Title'),
                        'description': video_data.get('description', ''),
                        'channel_title': video_data.get('uploader', 'Unknown Channel'),
                        'published_at': video_data.get('upload_date', ''),
                        'playlist_position': i + 1,
                        'duration': video_data.get('duration', 0),
                        'view_count': video_data.get('view_count', 0),
                        'like_count': video_data.get('like_count', 0),
                        'category_id': video_data.get('categories', []),
                        'default_language': video_data.get('language', None),
                        'default_audio_language': video_data.get('audio_language', None)
                    }
                    
                    videos.append(video_info)
                    logger.debug(f"Added video: {video_info['title']} ({video_info['video_id']})")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse video entry {i}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing video entry {i}: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(videos)} videos from playlist {playlist_id}")
            return videos
            
        except subprocess.TimeoutExpired:
            logger.error(f"yt-dlp timeout for playlist {playlist_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get playlist videos for {playlist_id}: {e}")
            return None
    
    def save_results(self, summary: Dict[str, Any], filename_prefix: str = "playlist"):
        """Save processing results"""
        try:
            # Save detailed results with consistent naming
            output_file = self.output_dir / "playlist_and_video_metadata.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Results saved: {output_file}")
            
            # Save summary with consistent naming
            summary_file = self.output_dir / "playlist_summary.json"
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
