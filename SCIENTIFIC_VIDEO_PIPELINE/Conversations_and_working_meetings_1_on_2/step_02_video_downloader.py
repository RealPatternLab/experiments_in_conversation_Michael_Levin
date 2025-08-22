#!/usr/bin/env python3
"""
Step 02: Video Download for 1-on-2 Conversations
Downloads videos using yt-dlp with comprehensive metadata extraction.
Specialized for conversations between Michael Levin and 2 other researchers.
"""

import os
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs
import yt_dlp
from dotenv import load_dotenv
from pipeline_progress_queue import get_progress_queue

# Load environment variables
load_dotenv()

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('video_download')

class ConversationsVideoDownloader:
    def __init__(self):
        self.input_dir = Path("step_01_raw")
        self.output_dir = Path("step_02_extracted_playlist_content")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = get_progress_queue()
        
        # Check if yt-dlp is available
        self.yt_dlp_available = self.check_yt_dlp_availability()
        
        # yt-dlp configuration for 1-on-2 conversations
        self.yt_dlp_config = {
            'format': 'best[height<=720][ext=mp4]/best[height<=720]/best',  # Flexible format selection
            'write_info_json': True,
            'write_auto_subs': True,
            'sub_langs': 'en',
            'write_description': True,
            'write_thumbnail': True,
            'extract_flat': False,
            'ignoreerrors': False,
            'no_warnings': False,
            'quiet': False,
            'verbose': True
        }
        
        # File paths
        self.metadata_file = self.input_dir / "playlist_and_video_metadata.json"
        self.download_summary_file = self.output_dir / "download_summary.json"
        
        # Load existing metadata
        self.playlist_metadata = self.load_playlist_metadata()
    
    def check_yt_dlp_availability(self) -> bool:
        """Check if yt-dlp is available and working"""
        try:
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
    
    def load_playlist_metadata(self) -> Dict[str, Any]:
        """Load playlist metadata from step 1"""
        if not self.metadata_file.exists():
            logger.error(f"Metadata file {self.metadata_file} not found")
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load playlist metadata: {e}")
            return {}
    
    def get_pending_videos(self) -> List[Dict[str, Any]]:
        """Get videos that need to be downloaded"""
        if not self.playlist_metadata:
            return []
        
        pending_videos = []
        
        for playlist in self.playlist_metadata.get('playlists', []):
            for video in playlist.get('videos', []):
                video_id = video.get('video_id')
                if not video_id:
                    continue
                
                # Check if video is already downloaded
                video_dir = self.output_dir / f"video_{video_id}"
                if not video_dir.exists() or not list(video_dir.glob("video.*")):
                    # Construct the complete video info with URL
                    video_info = {
                        'video_id': video_id,
                        'playlist_id': playlist.get('playlist_id'),
                        'playlist_title': playlist.get('title', 'Unknown'),
                        'video_title': video.get('title', 'Unknown'),
                        'video_url': f"https://www.youtube.com/watch?v={video_id}",
                        'uploader': video.get('uploader', 'Unknown'),
                        'duration': video.get('duration', 0),
                        'view_count': video.get('view_count', 0),
                        'upload_date': video.get('upload_date', ''),
                        'description': video.get('description', ''),
                        'tags': video.get('tags', []),
                        'categories': video.get('categories', [])
                    }
                    pending_videos.append(video_info)
        
        return pending_videos
    
    def get_available_formats(self, video_url: str) -> List[Dict[str, Any]]:
        """Get available formats for a video to help with format selection"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                formats = info.get('formats', [])
                
                # Filter and sort formats
                available_formats = []
                for fmt in formats:
                    if fmt.get('height') and fmt.get('ext'):
                        available_formats.append({
                            'format_id': fmt.get('format_id'),
                            'ext': fmt.get('ext'),
                            'height': fmt.get('height'),
                            'filesize': fmt.get('filesize'),
                            'vcodec': fmt.get('vcodec'),
                            'acodec': fmt.get('acodec')
                        })
                
                # Sort by height (quality) and prefer mp4
                available_formats.sort(key=lambda x: (x['height'] or 0, x['ext'] != 'mp4'), reverse=True)
                
                return available_formats
                
        except Exception as e:
            logger.warning(f"Could not get available formats: {e}")
            return []
    
    def select_best_format(self, video_url: str) -> str:
        """Select the best available format for download"""
        available_formats = self.get_available_formats(video_url)
        
        if not available_formats:
            # Fallback to flexible format selection that handles separate streams
            # Use mp4 for both video and audio since m4a might not be available
            return 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=mp4]/bestvideo[height<=720]+bestaudio/best'
        
        # Look for good quality mp4 format with both video and audio
        for fmt in available_formats:
            if (fmt['ext'] == 'mp4' and 
                fmt['height'] and 
                fmt['height'] <= 720 and 
                fmt['vcodec'] != 'none' and 
                fmt['acodec'] != 'none'):
                return fmt['format_id']
        
        # If no combined format found, try separate video+audio streams
        # This handles cases where only video-only and audio-only streams are available
        return 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=mp4]/bestvideo[height<=720]+bestaudio/best'
    
    def download_video(self, video_info: Dict[str, Any]) -> bool:
        """Download a single video using multiple fallback strategies"""
        video_id = video_info['video_id']
        video_url = video_info['video_url']
        
        logger.info(f"⬇️ Downloading video: {video_info['video_title']} ({video_id})")
        
        # Create video directory
        video_dir = self.output_dir / f"video_{video_id}"
        video_dir.mkdir(exist_ok=True)
        
        # Define download strategies in order of reliability
        download_strategies = [
            {
                'name': 'Format Diagnostics',
                'description': 'Lists available formats to diagnose the issue',
                'ydl_opts': {
                    'listformats': True,
                    'outtmpl': str(video_dir / 'video.%(ext)s'),
                    'writeinfojson': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'writedescription': True,
                    'writethumbnail': True,
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'quiet': False,
                    'verbose': True
                }
            },
            {
                'name': 'Aggressive Android Client',
                'description': 'Most aggressive - bypasses all restrictions',
                'ydl_opts': {
                    'format': 'best',
                    'extractor_args': {'youtube': {'player_client': 'android'}},
                    'outtmpl': str(video_dir / 'video.%(ext)s'),
                    'writeinfojson': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'writedescription': True,
                    'writethumbnail': True,
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'quiet': False,
                    'verbose': True
                }
            },
            {
                'name': 'Android Client Extractor',
                'description': 'Reliable - bypasses format restrictions',
                'ydl_opts': {
                    'format': 'best[height<=720]',
                    'extractor_args': {'youtube': {'player_client': 'android'}},
                    'outtmpl': str(video_dir / 'video.%(ext)s'),
                    'writeinfojson': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'writedescription': True,
                    'writethumbnail': True,
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'quiet': False,
                    'verbose': True
                }
            },
            {
                'name': 'Mobile Client Extractor',
                'description': 'Uses mobile client to access different formats',
                'ydl_opts': {
                    'format': 'best[height<=720]',
                    'extractor_args': {'youtube': {'player_client': 'mweb'}},
                    'outtmpl': str(video_dir / 'video.%(ext)s'),
                    'writeinfojson': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'writedescription': True,
                    'writethumbnail': True,
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'quiet': False,
                    'verbose': True
                }
            },
            {
                'name': 'TV Client Extractor',
                'description': 'Uses TV client for alternative format access',
                'ydl_opts': {
                    'format': 'best[height<=720]',
                    'extractor_args': {'youtube': {'player_client': 'tv'}},
                    'outtmpl': str(video_dir / 'video.%(ext)s'),
                    'writeinfojson': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'writedescription': True,
                    'writethumbnail': True,
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'quiet': False,
                    'verbose': True
                }
            },
            {
                'name': 'Flexible Format Selection',
                'description': 'Handles separate video+audio streams',
                'ydl_opts': {
                    'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=mp4]/bestvideo[height<=720]+bestaudio/best',
                    'outtmpl': str(video_dir / 'video.%(ext)s'),
                    'writeinfojson': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'writedescription': True,
                    'writethumbnail': True,
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'quiet': False,
                    'verbose': True
                }
            },
            {
                'name': 'Basic Best Quality',
                'description': 'Simple fallback - any available format',
                'ydl_opts': {
                    'format': 'best',
                    'outtmpl': str(video_dir / 'video.%(ext)s'),
                    'writeinfojson': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'writedescription': True,
                    'writethumbnail': True,
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'quiet': False,
                    'verbose': True
                }
            },
            {
                'name': 'Specific Format IDs',
                'description': 'Uses exact format IDs we discovered are available',
                'ydl_opts': {
                    'format': '609+234/232+234/231+234/230+234/229+234',  # 720p+audio, 720p+audio, 480p+audio, etc.
                    'outtmpl': str(video_dir / 'video.%(ext)s'),
                    'writeinfojson': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'writedescription': True,
                    'writethumbnail': True,
                    'ignoreerrors': False,
                    'no_warnings': False,
                    'quiet': False,
                    'verbose': True
                }
            }
        ]
        
        # Try each strategy until one succeeds
        for i, strategy in enumerate(download_strategies, 1):
            logger.info(f"   🎯 Strategy {i}/{len(download_strategies)}: {strategy['name']}")
            logger.info(f"      {strategy['description']}")
            
            try:
                # Clear any partial downloads from previous attempts
                self.cleanup_partial_downloads(video_dir)
                
                # Attempt download with current strategy
                with yt_dlp.YoutubeDL(strategy['ydl_opts']) as ydl:
                    logger.info(f"      Starting download for {video_id}")
                    ydl.download([video_url])
                
                # Verify download
                if self.verify_download_success(video_dir):
                    video_file = self.get_video_file(video_dir)
                    logger.info(f"✅ Success with {strategy['name']}: {video_file.name} ({video_file.stat().st_size / (1024*1024):.1f} MB)")
                    
                    # Update progress queue
                    self.progress_queue.update_video_step_status(
                        video_id,
                        'step_02_video_download',
                        'completed',
                        {
                            'download_path': str(video_file),
                            'file_size_mb': video_file.stat().st_size / (1024*1024),
                            'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'successful_strategy': strategy['name']
                        }
                    )
                    
                    # Save enhanced metadata
                    self.save_enhanced_metadata(video_info, video_file, video_dir)
                    
                    return True
                else:
                    logger.warning(f"      ❌ Strategy {strategy['name']} failed verification")
                    
            except Exception as e:
                logger.warning(f"      ❌ Strategy {strategy['name']} failed: {e}")
                continue
        
        # If all strategies failed
        logger.error(f"❌ All download strategies failed for {video_id}")
        
        # Update progress queue with failure
        self.progress_queue.update_video_step_status(
            video_id,
            'step_02_video_download',
            'failed',
            {
                'error': 'All download strategies failed',
                'failed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'strategies_tried': [s['name'] for s in download_strategies]
            }
        )
        
        return False
    
    def cleanup_partial_downloads(self, video_dir: Path):
        """Clean up partial downloads from previous attempts"""
        try:
            # Remove partial files
            for partial_file in video_dir.glob("*.part"):
                partial_file.unlink()
                logger.debug(f"      Removed partial file: {partial_file.name}")
            
            # Remove ytdl temporary files
            for ytdl_file in video_dir.glob("*.ytdl"):
                ytdl_file.unlink()
                logger.debug(f"      Removed ytdl file: {ytdl_file.name}")
                
        except Exception as e:
            logger.debug(f"      Cleanup warning: {e}")
    
    def verify_download_success(self, video_dir: Path) -> bool:
        """Verify that a download was successful"""
        try:
            # Look for actual video files (not metadata)
            video_files = list(video_dir.glob("video.*"))
            video_files = [f for f in video_files if f.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']]
            
            if not video_files:
                return False
            
            # Check if the video file has actual content (not 0 bytes)
            video_file = video_files[0]
            if video_file.stat().st_size < 1000:  # Less than 1KB
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"      Verification error: {e}")
            return False
    
    def get_video_file(self, video_dir: Path) -> Path:
        """Get the main video file from the directory"""
        video_files = list(video_dir.glob("video.*"))
        video_files = [f for f in video_files if f.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']]
        
        if not video_files:
            raise FileNotFoundError("No video file found")
        
        # Return the largest video file (usually the main one)
        return max(video_files, key=lambda f: f.stat().st_size)
    
    def save_enhanced_metadata(self, video_info: Dict[str, Any], video_file: Path, video_dir: Path):
        """Save enhanced metadata for the downloaded video"""
        video_id = video_info['video_id']
        
        # Load yt-dlp info file if available
        info_file = video_dir / "video.info.json"
        enhanced_info = {}
        
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    enhanced_info = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load yt-dlp info for {video_id}: {e}")
        
        # Create comprehensive metadata for 1-on-2 conversations
        metadata = {
            'video_id': video_id,
            'playlist_id': video_info['playlist_id'],
            'playlist_title': video_info['playlist_title'],
            'title': video_info['video_title'],
            'uploader': video_info['uploader'],
            'upload_date': video_info['upload_date'],
            'duration': video_info['duration'],
            'view_count': video_info['view_count'],
            'description': video_info['description'],
            'tags': video_info['tags'],
            'categories': video_info['categories'],
            'video_url': video_info['video_url'],
            'download_path': str(video_file),
            'file_size_mb': video_file.stat().st_size / (1024*1024),
            'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_type': 'conversations_1_on_2',  # Updated for 1-on-2
            'expected_speakers': 3,  # Michael Levin + 2 other researchers
            'content_type': 'multi_speaker_conversation',
            'conversation_metadata': {
                'speaker_diarization_expected': True,
                'qa_extraction_target': True,
                'levin_knowledge_focus': True,
                'conversation_context_preservation': True,
                'multi_speaker_analysis': True,
                'collaboration_pattern_detection': True
            },
            'yt_dlp_info': enhanced_info,
            'files_generated': {
                'video_file': str(video_file),
                'info_json': str(info_file) if info_file.exists() else None,
                'subtitles': [str(f) for f in video_dir.glob("*.vtt")],
                'thumbnail': [str(f) for f in video_dir.glob("*.webp")],
                'description': [str(f) for f in video_dir.glob("*.description")]
            }
        }
        
        # Save metadata
        metadata_file = video_dir / f"{video_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved enhanced metadata: {metadata_file}")
    
    def process_all_videos(self):
        """Process all pending videos"""
        if not self.yt_dlp_available:
            logger.error("yt-dlp is not available. Cannot proceed with video downloads.")
            return
        
        # Get pending videos
        pending_videos = self.get_pending_videos()
        
        if not pending_videos:
            logger.info("🎉 No videos pending for download. All videos are already processed.")
            return
        
        logger.info(f"📹 Found {len(pending_videos)} videos pending for download")
        
        # Process each video
        successful_downloads = 0
        failed_downloads = 0
        
        for i, video_info in enumerate(pending_videos, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📹 Processing video {i}/{len(pending_videos)}")
            logger.info(f"{'='*60}")
            
            try:
                success = self.download_video(video_info)
                if success:
                    successful_downloads += 1
                    logger.info(f"✅ Completed video {i}/{len(pending_videos)}")
                else:
                    failed_downloads += 1
                    logger.error(f"❌ Failed to process video {i}/{len(pending_videos)}")
                
                # Add delay between downloads to be respectful
                if i < len(pending_videos):
                    logger.info("   Waiting 2 seconds before next download...")
                    time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing video {i}: {e}")
                failed_downloads += 1
                continue
        
        # Generate download summary
        self.generate_download_summary(successful_downloads, failed_downloads)
        
        logger.info(f"\n🎉 Video download processing complete!")
        logger.info(f"   Successful: {successful_downloads}")
        logger.info(f"   Failed: {failed_downloads}")
        logger.info(f"   Total: {len(pending_videos)}")
    
    def generate_download_summary(self, successful_downloads: int, failed_downloads: int):
        """Generate a summary of the download process"""
        summary = {
            'pipeline_type': 'conversations_1_on_2',
            'step': 'step_02_video_download',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_videos_processed': successful_downloads + failed_downloads,
                'successful_downloads': successful_downloads,
                'failed_downloads': failed_downloads,
                'success_rate': f"{(successful_downloads/(successful_downloads + failed_downloads)*100):.1f}%" if (successful_downloads + failed_downloads) > 0 else "0%"
            },
            'download_details': []
        }
        
        # Add details for each video
        for playlist in self.playlist_metadata.get('playlists', []):
            for video in playlist.get('videos', []):
                video_id = video['video_id']
                step_status = self.progress_queue.get_video_step_status(video_id, 'step_02_video_download')
                
                video_dir = self.output_dir / f"video_{video_id}"
                video_file = video_dir / "video.mp4"
                
                video_detail = {
                    'video_id': video_id,
                    'title': video.get('title', 'Unknown'),
                    'playlist_id': playlist['playlist_id'],
                    'playlist_title': playlist.get('title', 'Unknown'),
                    'status': step_status,
                    'download_path': str(video_file) if video_file.exists() else None,
                    'file_size_mb': round(video_file.stat().st_size / (1024*1024), 2) if video_file.exists() else None
                }
                
                summary['download_details'].append(video_detail)
        
        # Save summary
        with open(self.download_summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved download summary: {self.download_summary_file}")

def main():
    """Main function"""
    try:
        downloader = ConversationsVideoDownloader()
        downloader.process_all_videos()
        
        logger.info("🎉 Video download step completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Video download step failed: {e}")
        raise

if __name__ == "__main__":
    main()
