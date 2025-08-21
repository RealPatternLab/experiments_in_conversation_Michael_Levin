#!/usr/bin/env python3
"""
Step 01: Playlist Processing for Conversations
Processes YouTube playlist URLs and extracts metadata for all videos within the playlist.
Specialized for conversations between Michael Levin and other researchers.
"""

import os
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs
import requests
from dotenv import load_dotenv
from pipeline_progress_queue import get_progress_queue

# Load environment variables
load_dotenv()

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('playlist_processing')

class ConversationsPlaylistProcessor:
    def __init__(self):
        self.input_dir = Path("step_01_raw")
        self.output_dir = Path("step_01_raw")  # Same directory for now
        
        # YouTube API configuration (optional, for enhanced metadata)
        self.youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        
        # Playlist processing parameters
        self.max_videos_per_playlist = 100  # Limit for processing
        
        # Check if yt-dlp is available
        self.yt_dlp_available = self.check_yt_dlp_availability()
        
        # Initialize progress queue
        self.progress_queue = get_progress_queue()
        
        # File paths for change detection
        self.metadata_file = self.output_dir / "playlist_and_video_metadata.json"
        self.previous_videos = self.load_previous_videos()
    
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
                playlist_info = self.process_playlist_url(url)
                if playlist_info:
                    processed_playlists.append(playlist_info)
                    total_videos += playlist_info['video_count']
                    logger.info(f"✅ Processed playlist: {playlist_info['title']} ({playlist_info['video_count']} videos)")
                else:
                    logger.error(f"❌ Failed to process playlist URL: {url}")
            except Exception as e:
                logger.error(f"❌ Error processing playlist URL {url}: {e}")
        
        # Save comprehensive metadata
        self.save_playlist_metadata(processed_playlists, total_videos)
        
        logger.info(f"🎉 Playlist processing complete!")
        logger.info(f"   Processed: {len(processed_playlists)} playlists")
        logger.info(f"   Total videos: {total_videos}")
    
    def process_playlist_url(self, playlist_url: str) -> Optional[Dict[str, Any]]:
        """Process a single playlist URL and extract metadata"""
        logger.info(f"Processing playlist URL: {playlist_url}")
        
        try:
            # Extract playlist ID
            playlist_id = self.extract_playlist_id(playlist_url)
            if not playlist_id:
                logger.error(f"Could not extract playlist ID from: {playlist_url}")
                return None
            
            # Get playlist info using yt-dlp
            playlist_info = self.get_playlist_info_with_ytdlp(playlist_url, playlist_id)
            if not playlist_info:
                logger.error(f"Failed to get playlist info for: {playlist_url}")
                return None
            
            # Add to progress queue
            self.progress_queue.add_playlist(
                playlist_id=playlist_id,
                playlist_title=playlist_info['title'],
                video_count=playlist_info['video_count']
            )
            
            return playlist_info
            
        except Exception as e:
            logger.error(f"Error processing playlist URL {playlist_url}: {e}")
            return None
    
    def extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from YouTube URL"""
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                if parsed_url.path == '/playlist':
                    return parse_qs(parsed_url.query).get('list', [None])[0]
            return None
        except Exception as e:
            logger.error(f"Error extracting playlist ID from {url}: {e}")
            return None
    
    def get_playlist_info_with_ytdlp(self, playlist_url: str, playlist_id: str) -> Optional[Dict[str, Any]]:
        """Get playlist information using yt-dlp"""
        try:
            import subprocess
            
            # Get playlist info
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--flat-playlist',
                '--playlist-items', '1',  # Just get first video to get playlist info
                playlist_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            lines = result.stdout.strip().split('\n')
            
            if not lines:
                logger.error("No playlist info returned from yt-dlp")
                return None
            
            # Parse first video to get playlist info
            first_video = json.loads(lines[0])
            
            # Get total video count
            count_cmd = [
                'yt-dlp',
                '--flat-playlist',
                '--get-id',
                playlist_url
            ]
            
            count_result = subprocess.run(count_cmd, capture_output=True, text=True, check=True, timeout=30)
            video_ids = count_result.stdout.strip().split('\n')
            video_count = len([vid for vid in video_ids if vid])
            
            playlist_info = {
                'playlist_id': playlist_id,
                'title': first_video.get('playlist_title', f'Playlist {playlist_id}'),
                'uploader': first_video.get('playlist_uploader', 'Unknown'),
                'video_count': video_count,
                'playlist_url': playlist_url,
                'first_video_id': first_video.get('id'),
                'first_video_title': first_video.get('title', 'Unknown'),
                'first_video_uploader': first_video.get('uploader', 'Unknown'),
                'first_video_upload_date': first_video.get('upload_date'),
                'first_video_duration': first_video.get('duration'),
                'first_video_view_count': first_video.get('view_count'),
                'first_video_like_count': first_video.get('like_count'),
                'first_video_description': first_video.get('description', ''),
                'first_video_tags': first_video.get('tags', []),
                'first_video_categories': first_video.get('categories', []),
                'processing_type': 'conversations_1_on_1',
                'expected_speakers': 2,  # Michael Levin + 1 other researcher
                'content_type': 'conversation',
                'extraction_timestamp': subprocess.run(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], 
                                                    capture_output=True, text=True, check=True).stdout.strip()
            }
            
            # Get all video IDs and basic info
            all_videos_cmd = [
                'yt-dlp',
                '--dump-json',
                '--flat-playlist',
                playlist_url
            ]
            
            all_videos_result = subprocess.run(all_videos_cmd, capture_output=True, text=True, check=True, timeout=60)
            all_video_lines = all_videos_result.stdout.strip().split('\n')
            
            videos = []
            for line in all_video_lines:
                if line.strip():
                    try:
                        video_data = json.loads(line)
                        video_info = {
                            'video_id': video_data.get('id'),
                            'title': video_data.get('title', 'Unknown'),
                            'uploader': video_data.get('uploader', 'Unknown'),
                            'upload_date': video_data.get('upload_date'),
                            'duration': video_data.get('duration'),
                            'view_count': video_data.get('view_count'),
                            'like_count': video_data.get('like_count'),
                            'description': video_data.get('description', ''),
                            'tags': video_data.get('tags', []),
                            'categories': video_data.get('categories', []),
                            'playlist_index': video_data.get('playlist_index'),
                            'playlist_id': playlist_id
                        }
                        videos.append(video_info)
                        
                        # Add to progress queue
                        if video_info['video_id']:
                            self.progress_queue.add_video(
                                video_id=video_info['video_id'],
                                video_title=video_info['title'],
                                playlist_id=playlist_id
                            )
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse video line: {e}")
                        continue
            
            playlist_info['videos'] = videos
            
            return playlist_info
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout getting playlist info for: {playlist_url}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp command failed for {playlist_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting playlist info for {playlist_url}: {e}")
            return None
    
    def load_previous_videos(self) -> set:
        """Load previously processed videos for change detection"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    videos = set()
                    for playlist in data.get('playlists', []):
                        for video in playlist.get('videos', []):
                            if video.get('video_id'):
                                videos.add(video['video_id'])
                    return videos
            except Exception as e:
                logger.warning(f"Failed to load previous videos: {e}")
        return set()
    
    def save_playlist_metadata(self, playlists: List[Dict[str, Any]], total_videos: int):
        """Save comprehensive playlist and video metadata"""
        metadata = {
            'pipeline_type': 'conversations_1_on_1',
            'description': 'Metadata for conversations between Michael Levin and other researchers',
            'extraction_timestamp': subprocess.run(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], 
                                                capture_output=True, text=True, check=True).stdout.strip(),
            'total_playlists': len(playlists),
            'total_videos': total_videos,
            'playlists': playlists,
            'processing_notes': [
                'This pipeline is designed for conversations between Michael Levin and other researchers',
                'Speaker diarization will be used to identify different speakers',
                'Semantic chunking will focus on Levin\'s views and knowledge',
                'Q&A pairs will be extracted for fine-tuning purposes'
            ]
        }
        
        # Save to file
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved playlist metadata to: {self.metadata_file}")
        
        # Also save a summary file
        summary_file = self.output_dir / "playlist_summary.json"
        summary = {
            'pipeline_type': 'conversations_1_on_1',
            'total_playlists': len(playlists),
            'total_videos': total_videos,
            'playlists': [
                {
                    'playlist_id': p['playlist_id'],
                    'title': p['title'],
                    'video_count': p['video_count'],
                    'uploader': p['uploader']
                }
                for p in playlists
            ],
            'extraction_timestamp': metadata['extraction_timestamp']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved playlist summary to: {summary_file}")

def main():
    """Main function"""
    try:
        processor = ConversationsPlaylistProcessor()
        processor.process_playlists()
        
        logger.info("🎉 Playlist processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Playlist processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
