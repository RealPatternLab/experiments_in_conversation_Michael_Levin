#!/usr/bin/env python3
"""
Step 02: Video Download for Conversations
Downloads videos using yt-dlp with comprehensive metadata extraction.
Specialized for conversations between Michael Levin and other researchers.
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
        
        # yt-dlp configuration for conversations
        self.yt_dlp_config = {
            'format': '232+233',  # 1152x720 video + audio (format IDs from available formats)
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
            playlist_id = playlist['playlist_id']
            
            for video in playlist.get('videos', []):
                video_id = video['video_id']
                
                # Check if video is pending for download
                step_status = self.progress_queue.get_video_step_status(video_id, 'step_02_video_download')
                
                # Treat None (no entry) as pending, or if status is explicitly pending
                if step_status is None or step_status == 'pending':
                    # Check if video files already exist
                    video_dir = self.output_dir / f"video_{video_id}"
                    video_file = video_dir / "video.mp4"
                    
                    if not video_file.exists():
                        pending_videos.append({
                            'video_id': video_id,
                            'playlist_id': playlist_id,
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
                        })
                    else:
                        logger.info(f"Video {video_id} already exists, marking as completed")
                        self.progress_queue.update_video_step_status(
                            video_id, 
                            'step_02_video_download', 
                            'completed',
                            {'download_path': str(video_file)}
                        )
        
        return pending_videos
    
    def download_video(self, video_info: Dict[str, Any]) -> bool:
        """Download a single video"""
        video_id = video_info['video_id']
        video_url = video_info['video_url']
        
        logger.info(f"⬇️ Downloading video: {video_info['video_title']} ({video_id})")
        
        # Create video directory
        video_dir = self.output_dir / f"video_{video_id}"
        video_dir.mkdir(exist_ok=True)
        
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': '232+233',  # 1152x720 video + audio (format IDs from available formats)
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
            
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"   Starting download for {video_id}")
                ydl.download([video_url])
            
            # Verify download
            video_files = list(video_dir.glob("video.*"))
            video_files = [f for f in video_files if f.suffix not in ['.json', '.vtt', '.description', '.webp']]
            
            if not video_files:
                logger.error(f"❌ No video file found after download for {video_id}")
                return False
            
            video_file = video_files[0]
            logger.info(f"✅ Downloaded: {video_file.name} ({video_file.stat().st_size / (1024*1024):.1f} MB)")
            
            # Update progress queue
            self.progress_queue.update_video_step_status(
                video_id,
                'step_02_video_download',
                'completed',
                {
                    'download_path': str(video_file),
                    'file_size_mb': video_file.stat().st_size / (1024*1024),
                    'download_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            )
            
            # Save enhanced metadata
            self.save_enhanced_metadata(video_info, video_file, video_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download video {video_id}: {e}")
            
            # Update progress queue with failure
            self.progress_queue.update_video_step_status(
                video_id,
                'step_02_video_download',
                'failed',
                {'error': str(e), 'failed_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            )
            
            return False
    
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
        
        # Create comprehensive metadata
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
            'processing_type': 'conversations_1_on_1',
            'expected_speakers': 2,  # Michael Levin + 1 other researcher
            'content_type': 'conversation',
            'conversation_metadata': {
                'speaker_diarization_expected': True,
                'qa_extraction_target': True,
                'levin_knowledge_focus': True,
                'conversation_context_preservation': True
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
            'pipeline_type': 'conversations_1_on_1',
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
