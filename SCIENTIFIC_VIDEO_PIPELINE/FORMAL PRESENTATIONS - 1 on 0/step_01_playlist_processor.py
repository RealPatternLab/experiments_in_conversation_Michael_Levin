#!/usr/bin/env python3
"""
Step 01: Playlist URL Processor

This script processes YouTube playlist URLs from text files and validates them.
It extracts playlist information and video metadata, preparing for video downloading in step 2.
"""

import re
import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# YouTube Data API imports
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    logging.warning("YouTube Data API not available. Video metadata extraction will be limited.")

# Progress queue integration (similar to publications pipeline)
try:
    from pipeline_progress_queue import PipelineProgressQueue
    PROGRESS_QUEUE_AVAILABLE = True
except ImportError:
    PROGRESS_QUEUE_AVAILABLE = False
    logging.warning("Progress queue not available. Pipeline tracking will be limited.")


class PlaylistProcessor:
    """Processes YouTube playlist URLs and extracts video metadata."""
    
    def __init__(self, raw_dir: str = "step_01_raw"):
        self.raw_dir = Path(raw_dir)
        self.logger = self._setup_logging()
        
        # YouTube API setup
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        if self.youtube_api_key and YOUTUBE_API_AVAILABLE:
            try:
                self.youtube_service = build('youtube', 'v3', developerKey=self.youtube_api_key)
                self.logger.info("YouTube Data API service initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize YouTube API service: {e}")
                self.youtube_service = None
        else:
            self.youtube_service = None
            if not self.youtube_api_key:
                self.logger.warning("YOUTUBE_API_KEY environment variable not set")
            if not YOUTUBE_API_AVAILABLE:
                self.logger.warning("YouTube Data API not available")
        
        # Initialize progress queue if available
        self.progress_queue = None
        if PROGRESS_QUEUE_AVAILABLE:
            try:
                self.progress_queue = PipelineProgressQueue()
                self.logger.info("Progress queue initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize progress queue: {e}")
                self.progress_queue = None
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        file_handler = logging.FileHandler('playlist_processing.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def validate_youtube_playlist_url(self, url: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate a YouTube playlist URL and extract playlist ID.
        
        Args:
            url: YouTube playlist URL to validate
            
        Returns:
            Tuple of (is_valid, error_message, playlist_id)
        """
        try:
            # Basic URL validation
            parsed = urlparse(url.strip())
            if parsed.netloc not in ['www.youtube.com', 'youtube.com', 'm.youtube.com']:
                return False, f"Invalid domain: {parsed.netloc}", None
            
            # Check if it's a playlist URL
            if '/playlist' not in parsed.path:
                return False, "URL does not contain '/playlist' path", None
            
            # Extract playlist ID from query parameters
            query_params = parse_qs(parsed.query)
            playlist_id = query_params.get('list', [None])[0]
            
            if not playlist_id:
                return False, "No playlist ID found in URL", None
            
            # Validate playlist ID format (YouTube playlist IDs are typically 34 characters)
            if len(playlist_id) < 10 or len(playlist_id) > 50:
                return False, f"Invalid playlist ID length: {len(playlist_id)}", None
            
            return True, "", playlist_id
            
        except Exception as e:
            return False, f"URL parsing error: {str(e)}", None
    
    def extract_playlist_metadata(self, playlist_id: str) -> Dict[str, Any]:
        """
        Extract metadata about the playlist itself.
        
        Args:
            playlist_id: YouTube playlist ID
            
        Returns:
            Dictionary with playlist metadata
        """
        if not self.youtube_service:
            self.logger.warning("YouTube API service not available, using minimal metadata")
            return {
                'playlist_id': playlist_id,
                'title': f"Playlist {playlist_id}",
                'description': "Metadata not available without YouTube API key",
                'channel_title': "Unknown",
                'published_at': None,
                'video_count': 0,
                'api_available': False
            }
        
        try:
            # Get playlist details
            playlist_response = self.youtube_service.playlists().list(
                part='snippet,contentDetails',
                id=playlist_id
            ).execute()
            
            if not playlist_response.get('items'):
                self.logger.warning(f"No playlist found with ID: {playlist_id}")
                return {
                    'playlist_id': playlist_id,
                    'title': f"Unknown Playlist {playlist_id}",
                    'description': "Playlist not found",
                    'channel_title': "Unknown",
                    'published_at': None,
                    'video_count': 0,
                    'api_available': True
                }
            
            playlist_info = playlist_response['items'][0]
            snippet = playlist_info.get('snippet', {})
            content_details = playlist_info.get('contentDetails', {})
            
            metadata = {
                'playlist_id': playlist_id,
                'title': snippet.get('title', 'Untitled Playlist'),
                'description': snippet.get('description', ''),
                'channel_title': snippet.get('channelTitle', 'Unknown Channel'),
                'published_at': snippet.get('publishedAt'),
                'video_count': content_details.get('itemCount', 0),
                'api_available': True,
                'thumbnails': snippet.get('thumbnails', {}),
                'tags': snippet.get('tags', []),
                'default_language': snippet.get('defaultLanguage'),
                'privacy_status': snippet.get('privacyStatus')
            }
            
            self.logger.info(f"Extracted playlist metadata: {metadata['title']} ({metadata['video_count']} videos)")
            return metadata
            
        except HttpError as e:
            self.logger.error(f"YouTube API error: {e}")
            return {
                'playlist_id': playlist_id,
                'title': f"Playlist {playlist_id}",
                'description': f"API Error: {str(e)}",
                'channel_title': "Unknown",
                'published_at': None,
                'video_count': 0,
                'api_available': True,
                'api_error': str(e)
            }
        except Exception as e:
            self.logger.error(f"Error extracting playlist metadata: {e}")
            return {
                'playlist_id': playlist_id,
                'title': f"Playlist {playlist_id}",
                'description': f"Error: {str(e)}",
                'channel_title': "Unknown",
                'published_at': None,
                'video_count': 0,
                'api_available': True,
                'api_error': str(e)
            }
    
    def extract_playlist_videos(self, playlist_id: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Extract metadata for all videos in the playlist.
        
        Args:
            playlist_id: YouTube playlist ID
            max_results: Maximum number of videos to extract (default 50)
            
        Returns:
            List of video metadata dictionaries
        """
        if not self.youtube_service:
            self.logger.warning("YouTube API service not available, cannot extract video metadata")
            return []
        
        videos = []
        next_page_token = None
        
        try:
            while len(videos) < max_results:
                # Get playlist items
                playlist_items_response = self.youtube_service.playlistItems().list(
                    part='snippet,contentDetails',
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()
                
                items = playlist_items_response.get('items', [])
                if not items:
                    break
                
                for item in items:
                    snippet = item.get('snippet', {})
                    content_details = item.get('contentDetails', {})
                    
                    video_metadata = {
                        'video_id': content_details.get('videoId'),
                        'title': snippet.get('title', 'Untitled Video'),
                        'description': snippet.get('description', ''),
                        'channel_title': snippet.get('channelTitle', 'Unknown Channel'),
                        'published_at': snippet.get('publishedAt'),
                        'playlist_position': snippet.get('position', 0),
                        'duration': None,  # Will be filled in next step
                        'view_count': None,  # Will be filled in next step
                        'like_count': None,  # Will be filled in next step
                        'thumbnails': snippet.get('thumbnails', {}),
                        'tags': snippet.get('tags', []),
                        'category_id': snippet.get('categoryId'),
                        'default_language': snippet.get('defaultLanguage'),
                        'default_audio_language': snippet.get('defaultAudioLanguage')
                    }
                    
                    videos.append(video_metadata)
                
                # Check for next page
                next_page_token = playlist_items_response.get('nextPageToken')
                if not next_page_token:
                    break
            
            self.logger.info(f"Extracted metadata for {len(videos)} videos from playlist")
            return videos
            
        except HttpError as e:
            self.logger.error(f"YouTube API error extracting videos: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error extracting video metadata: {e}")
            return []
    
    def read_playlist_file(self, filename: str = "youtube_playlist.txt") -> List[str]:
        """
        Read playlist URLs from a text file.
        
        Args:
            filename: Name of the playlist file
            
        Returns:
            List of playlist URLs
        """
        playlist_file = self.raw_dir / filename
        
        if not playlist_file.exists():
            self.logger.error(f"Playlist file not found: {playlist_file}")
            return []
        
        try:
            with open(playlist_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            self.logger.info(f"Read {len(urls)} URLs from {filename}")
            return urls
            
        except Exception as e:
            self.logger.error(f"Error reading playlist file: {e}")
            return []
    
    def process_playlist_url(self, url: str) -> Dict[str, Any]:
        """
        Process a single playlist URL and extract comprehensive information.
        
        Args:
            url: YouTube playlist URL to process
            
        Returns:
            Dictionary with playlist and video information
        """
        is_valid, error_msg, playlist_id = self.validate_youtube_playlist_url(url)
        
        if not is_valid:
            self.logger.error(f"Invalid playlist URL: {error_msg}")
            return {
                'url': url,
                'is_valid': False,
                'error': error_msg,
                'playlist_id': None,
                'processed_at': datetime.now().isoformat()
            }
        
        # Extract additional information from URL
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Extract playlist metadata
        playlist_metadata = self.extract_playlist_metadata(playlist_id)
        
        # Extract video metadata
        video_metadata = self.extract_playlist_videos(playlist_id)
        
        playlist_info = {
            'url': url,
            'is_valid': True,
            'error': None,
            'playlist_id': playlist_id,
            'domain': parsed.netloc,
            'scheme': parsed.scheme,
            'query_params': query_params,
            'playlist_metadata': playlist_metadata,
            'videos': video_metadata,
            'total_videos_found': len(video_metadata),
            'processed_at': datetime.now().isoformat()
        }
        
        self.logger.info(f"Successfully processed playlist: {playlist_id} with {len(video_metadata)} videos")
        return playlist_info
    
    def process_all_playlists(self) -> Dict[str, Any]:
        """
        Process all playlist URLs found in the raw directory.
        
        Returns:
            Dictionary with processing results
        """
        self.logger.info("Starting playlist processing...")
        
        # Read playlist URLs
        urls = self.read_playlist_file()
        if not urls:
            self.logger.warning("No playlist URLs found to process")
            return {
                'total_urls': 0,
                'valid_urls': 0,
                'invalid_urls': 0,
                'total_videos_found': 0,
                'playlists': [],
                'processing_timestamp': datetime.now().isoformat()
            }
        
        # Process each URL
        results = []
        valid_count = 0
        invalid_count = 0
        total_videos = 0
        
        for url in urls:
            result = self.process_playlist_url(url)
            results.append(result)
            
            if result['is_valid']:
                valid_count += 1
                total_videos += result.get('total_videos_found', 0)
            else:
                invalid_count += 1
        
        # Create summary
        summary = {
            'total_urls': len(urls),
            'valid_urls': valid_count,
            'invalid_urls': invalid_count,
            'total_videos_found': total_videos,
            'playlists': results,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_processing_results(summary)
        
        self.logger.info(f"Processing complete: {valid_count} valid, {invalid_count} invalid, {total_videos} total videos")
        return summary
    
    def _save_processing_results(self, results: Dict[str, Any]) -> None:
        """Save processing results to a JSON file."""
        output_file = self.raw_dir / "playlist_processing_results.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def get_valid_playlists(self) -> List[Dict[str, Any]]:
        """Get list of valid playlists for next pipeline step."""
        results_file = self.raw_dir / "playlist_processing_results.json"
        
        if not results_file.exists():
            self.logger.warning("No processing results found. Run process_all_playlists() first.")
            return []
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            valid_playlists = [p for p in data['playlists'] if p['is_valid']]
            return valid_playlists
            
        except Exception as e:
            self.logger.error(f"Error reading results file: {e}")
            return []


def main():
    """Main execution function."""
    processor = PlaylistProcessor()
    
    print("🎬 YouTube Playlist Processor - Step 01")
    print("=" * 50)
    
    # Check API availability
    if not processor.youtube_service:
        print("⚠️  YouTube Data API not available")
        print("   Set YOUTUBE_API_KEY environment variable for full metadata extraction")
        print("   Continuing with basic URL validation only...")
        print()
    
    # Process all playlists
    results = processor.process_all_playlists()
    
    # Display results
    print(f"\n📊 Processing Results:")
    print(f"Total URLs: {results['total_urls']}")
    print(f"Valid Playlists: {results['valid_urls']}")
    print(f"Invalid URLs: {results['invalid_urls']}")
    print(f"Total Videos Found: {results['total_videos_found']}")
    
    if results['valid_urls'] > 0:
        print(f"\n✅ Valid Playlists:")
        for playlist in results['playlists']:
            if playlist['is_valid']:
                playlist_meta = playlist.get('playlist_metadata', {})
                video_count = playlist.get('total_videos_found', 0)
                title = playlist_meta.get('title', f"Playlist {playlist['playlist_id']}")
                print(f"  • {title} ({video_count} videos)")
                
                # Show first few videos
                videos = playlist.get('videos', [])
                if videos:
                    print(f"    Videos:")
                    for i, video in enumerate(videos[:3]):  # Show first 3
                        print(f"      {i+1}. {video.get('title', 'Unknown')}")
                    if len(videos) > 3:
                        print(f"      ... and {len(videos) - 3} more")
    
    if results['invalid_urls'] > 0:
        print(f"\n❌ Invalid URLs:")
        for playlist in results['playlists']:
            if not playlist['is_valid']:
                print(f"  • {playlist['url']} - {playlist['error']}")
    
    print(f"\n📁 Results saved to: step_01_raw/playlist_processing_results.json")
    
    # Show next steps
    if results['valid_urls'] > 0:
        print(f"\n🚀 Ready for Step 02: Video Download and Metadata Extraction")
        print(f"Run: python step_02_video_downloader.py")


if __name__ == "__main__":
    main()
