#!/usr/bin/env python3
"""
Step 02: Video Download & Processing

This script downloads videos from the playlist metadata extracted in Step 01.
It downloads each video, extracts additional metadata, and organizes files
in the step_02_extracted_playlist_content directory.
"""

import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
from pipeline_progress_queue import get_progress_queue

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('video_download.log')
        ]
    )
    return logging.getLogger(__name__)

def check_yt_dlp_available() -> bool:
    """Check if yt-dlp is available on the system."""
    try:
        result = subprocess.run(['yt-dlp', '--version'], 
                              capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def load_playlist_metadata(metadata_file: Path) -> Dict[str, Any]:
    """Load the playlist and video metadata from Step 01."""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded metadata for {data.get('total_videos_found', 0)} videos")
        return data
        
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return {}

def create_video_directory(output_dir: Path, video_id: str) -> Path:
    """Create a directory for a specific video."""
    video_dir = output_dir / f"video_{video_id}"
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir

def download_video(video_id: str, video_title: str, output_dir: Path) -> Dict[str, Any]:
    """
    Download a single video using yt-dlp.
    
    Args:
        video_id: YouTube video ID
        video_title: Video title for logging
        output_dir: Directory to save the video
        
    Returns:
        Dictionary with download results
    """
    video_dir = create_video_directory(output_dir, video_id)
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    print(f"⬇️  Downloading: {video_title[:60]}...")
    
    try:
        # Download video with metadata
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",  # Limit quality for processing
            "--write-info-json",
            "--write-auto-subs",
            "--sub-langs", "en",
            "--write-description",
            "--write-thumbnail",
            "-o", str(video_dir / "video.%(ext)s"),
            video_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the downloaded video file
        video_files = list(video_dir.glob("video.*"))
        # Filter to only video files (exclude metadata files)
        video_files = [f for f in video_files if f.suffix.lower() in ['.mp4', '.webm', '.mkv', '.avi', '.mov']]
        
        if video_files:
            video_file = video_files[0]
            file_size_mb = video_file.stat().st_size / (1024*1024)
            print(f"✅ Downloaded: {video_file.name} ({file_size_mb:.1f} MB)")
            
            # Extract additional metadata
            additional_metadata = extract_additional_metadata(video_dir, video_id)
            
            return {
                'success': True,
                'video_file': str(video_file),
                'file_size_mb': video_file.stat().st_size / (1024*1024),
                'additional_metadata': additional_metadata,
                'download_time': datetime.now().isoformat()
            }
        else:
            print(f"❌ No video file found after download")
            return {
                'success': False,
                'error': 'No video file found after download',
                'download_time': datetime.now().isoformat()
            }
            
    except subprocess.CalledProcessError as e:
        error_msg = f"yt-dlp error: {e.stderr}"
        print(f"❌ Download failed: {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'download_time': datetime.now().isoformat()
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ Download failed: {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'download_time': datetime.now().isoformat()
        }

def extract_additional_metadata(video_dir: Path, video_id: str) -> Dict[str, Any]:
    """Extract additional metadata from downloaded files."""
    metadata = {}
    
    # Load video info JSON
    info_file = video_dir / "video.info.json"
    if info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                video_info = json.load(f)
            
            metadata.update({
                'upload_date': video_info.get('upload_date'),
                'duration': video_info.get('duration'),
                'view_count': video_info.get('view_count'),
                'like_count': video_info.get('like_count'),
                'tags': video_info.get('tags', []),
                'categories': video_info.get('categories', []),
                'language': video_info.get('language'),
                'age_limit': video_info.get('age_limit'),
                'is_live': video_info.get('is_live', False),
                'was_live': video_info.get('was_live', False),
                'live_status': video_info.get('live_status'),
                'availability': video_info.get('availability'),
                'format': video_info.get('format'),
                'resolution': video_info.get('resolution'),
                'fps': video_info.get('fps'),
                'vcodec': video_info.get('vcodec'),
                'acodec': video_info.get('acodec'),
                'filesize': video_info.get('filesize'),
                'filesize_approx': video_info.get('filesize_approx')
            })
        except Exception as e:
            print(f"⚠️  Failed to parse video info: {e}")
    
    # Check for subtitle files
    subtitle_files = list(video_dir.glob("*.vtt"))
    metadata['subtitle_files'] = [f.name for f in subtitle_files]
    
    # Check for thumbnail
    thumbnail_files = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.webp"))
    metadata['thumbnail_files'] = [f.name for f in thumbnail_files]
    
    # Check for description
    description_file = video_dir / "video.description.txt"
    if description_file.exists():
        try:
            with open(description_file, 'r', encoding='utf-8') as f:
                metadata['description'] = f.read()
        except Exception as e:
            print(f"⚠️  Failed to read description: {e}")
    
    return metadata

def save_video_metadata(video_id: str, original_metadata: Dict, download_result: Dict, output_dir: Path):
    """Save comprehensive metadata for a video."""
    video_dir = create_video_directory(output_dir, video_id)
    
    # Combine original and new metadata
    comprehensive_metadata = {
        'video_id': video_id,
        'original_metadata': original_metadata,
        'download_result': download_result,
        'processing_timestamp': datetime.now().isoformat()
    }
    
    metadata_file = video_dir / "comprehensive_metadata.json"
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved metadata: {metadata_file}")
        
    except Exception as e:
        print(f"❌ Failed to save metadata: {e}")

def get_video_file_info(video_dir: Path) -> Dict[str, Any]:
    """
    Get information about an existing video file.
    
    Args:
        video_dir: Directory containing the video
        
    Returns:
        Dictionary with file information
    """
    video_files = list(video_dir.glob("video.*"))
    # Filter to only video files
    video_files = [f for f in video_files if f.suffix.lower() in ['.mp4', '.webm', '.mkv', '.avi', '.mov']]
    
    if not video_files:
        return {'exists': False, 'file_path': None, 'file_size_mb': 0}
    
    video_file = video_files[0]
    file_size_mb = video_file.stat().st_size / (1024*1024)
    
    return {
        'exists': True,
        'file_path': str(video_file),
        'file_size_mb': file_size_mb,
        'file_extension': video_file.suffix
    }

def update_existing_video_metadata(video_id: str, new_playlist_metadata: Dict, output_dir: Path):
    """
    Update metadata for an existing video to show it belongs to this playlist.
    
    Args:
        video_id: YouTube video ID
        new_playlist_metadata: Metadata from the current playlist
        output_dir: Base output directory
    """
    video_dir = output_dir / f"video_{video_id}"
    metadata_file = video_dir / "comprehensive_metadata.json"
    
    # Get actual video file information
    video_info = get_video_file_info(video_dir)
    
    if not metadata_file.exists():
        print(f"⚠️  No existing metadata found for {video_id}, creating new entry")
        # Create new metadata if none exists
        save_video_metadata(video_id, new_playlist_metadata, {
            'success': True,
            'video_file': video_info.get('file_path', str(video_dir / "video.mp4")),
            'file_size_mb': video_info.get('file_size_mb', 0),
            'additional_metadata': {},
            'download_time': 'already_processed',
            'status': 'existing_video'
        }, output_dir)
        return
    
    try:
        # Load existing metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            existing_metadata = json.load(f)
        
        # Add playlist information
        if 'playlists' not in existing_metadata:
            existing_metadata['playlists'] = []
        
        # Check if this playlist is already listed
        current_playlist_id = new_playlist_metadata.get('playlist_id', 'unknown')
        playlist_already_exists = any(
            p.get('playlist_id') == current_playlist_id 
            for p in existing_metadata['playlists']
        )
        
        if not playlist_already_exists:
            # Add this playlist to the list
            playlist_info = {
                'playlist_id': current_playlist_id,
                'playlist_title': new_playlist_metadata.get('playlist_title', 'Unknown'),
                'added_to_playlist': datetime.now().isoformat(),
                'video_position': new_playlist_metadata.get('position', 0)
            }
            existing_metadata['playlists'].append(playlist_info)
            
            # Update the processing timestamp
            existing_metadata['last_updated'] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Updated metadata for {video_id} - added to playlist {current_playlist_id}")
        else:
            print(f"ℹ️  Video {video_id} already belongs to playlist {current_playlist_id}")
            
    except Exception as e:
        print(f"❌ Failed to update metadata for {video_id}: {e}")
        # Fallback: create new metadata entry
        save_video_metadata(video_id, new_playlist_metadata, {
            'success': True,
            'video_file': video_info.get('file_path', str(video_dir / "video.mp4")),
            'file_size_mb': video_info.get('file_size_mb', 0),
            'additional_metadata': {},
            'download_time': 'already_processed',
            'status': 'existing_video_metadata_updated'
        }, output_dir)

def process_playlist_videos(playlist_data: Dict, output_dir: Path, max_videos: Optional[int] = None, progress_queue = None) -> Dict[str, Any]:
    """
    Process all videos in the playlist.
    
    Args:
        playlist_data: Data from Step 01
        output_dir: Directory to save videos
        max_videos: Maximum number of videos to process (for testing)
        
    Returns:
        Dictionary with processing results
    """
    if not playlist_data.get('playlists'):
        print("❌ No playlist data found")
        return {}
    
    playlist = playlist_data['playlists'][0]
    videos = playlist.get('videos', [])
    
    if max_videos:
        videos = videos[:max_videos]
        print(f"🔢 Limited to {max_videos} videos for testing")
    
    print(f"🎬 Processing {len(videos)} videos...")
    
    results = []
    successful_downloads = 0
    failed_downloads = 0
    skipped_videos = 0
    existing_videos = 0
    
    for i, video in enumerate(videos, 1):
        video_id = video.get('video_id')
        video_title = video.get('title', 'Unknown Title')
        
        if not video_id:
            print(f"⚠️  Skipping video {i}: No video ID")
            continue
        
        print(f"\n{'='*60}")
        print(f"📹 Processing video {i}/{len(videos)}: {video_id}")
        print(f"{'='*60}")
        
        # Check if video has already been processed using progress queue
        if progress_queue:
            video_status = progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_02_video_download') == 'completed':
                print(f"🔄 Video already processed in step 2: {video_id}")
                print(f"   Status: {video_status.get('step_02_video_download')}")
                
                # Track as existing video
                existing_videos += 1
                results.append({
                    'video_id': video_id,
                    'title': video_title,
                    'status': 'already_processed',
                    'local_path': str(output_dir / f"video_{video_id}/video.mp4"),  # Full path
                    'file_size_mb': 0,  # Will be updated if file exists
                    'playlist_added': datetime.now().isoformat()
                })
                
                print(f"✅ Video {i} already processed (skipping)")
                
                # Update progress queue to mark as completed
                if progress_queue:
                    progress_queue.update_video_step_status(
                        video_id,
                        'step_02_video_download',
                        'completed',
                        metadata={
                            'status': 'already_processed',
                            'playlist_added': datetime.now().isoformat()
                        }
                    )
                    print(f"   📊 Progress queue updated: step 2 marked as completed")
                continue
        
        # Fallback: Check if video directory exists (for backward compatibility)
        video_dir = output_dir / f"video_{video_id}"
        video_info = get_video_file_info(video_dir)
        
        if video_dir.exists() and video_info['exists']:
            print(f"🔄 Video already exists: {video_id}")
            print(f"   Directory: {video_dir}")
            print(f"   Video file: {video_info['file_path']}")
            print(f"   File size: {video_info['file_size_mb']:.1f} MB")
            
            # Update metadata to show this video belongs to this playlist
            update_existing_video_metadata(video_id, video, output_dir)
            
            # Track as existing video
            existing_videos += 1
            results.append({
                'video_id': video_id,
                'title': video_title,
                'status': 'already_processed',
                'local_path': video_info['file_path'],
                'file_size_mb': video_info['file_size_mb'],
                'playlist_added': datetime.now().isoformat()
            })
            
            print(f"✅ Video {i} metadata updated (already processed)")
            
            # Update progress queue to mark as completed
            if progress_queue:
                progress_queue.update_video_step_status(
                    video_id,
                    'step_02_video_download',
                    'completed',
                    metadata={
                        'status': 'already_processed',
                        'file_size_mb': video_info['file_size_mb'],
                        'local_path': str(video_info['file_path']),
                        'playlist_added': datetime.now().isoformat()
                    }
                )
                print(f"   📊 Progress queue updated: step 2 marked as completed")
            continue
        
        # Download video if it doesn't exist
        print(f"⬇️  Downloading new video: {video_id}")
        download_result = download_video(video_id, video_title, output_dir)
        
        # Save comprehensive metadata
        save_video_metadata(video_id, video, download_result, output_dir)
        
        # Track results
        if download_result['success']:
            successful_downloads += 1
            print(f"✅ Video {i} downloaded successfully")
            
            # Update progress queue
            if progress_queue:
                progress_queue.update_video_step_status(
                    video_id,
                    'step_02_video_download',
                    'completed',
                    metadata={
                        'file_size_mb': download_result.get('file_size_mb', 0),
                        'local_path': str(download_result.get('local_path', '')),
                        'download_timestamp': datetime.now().isoformat()
                    }
                )
                print(f"   📊 Progress queue updated: step 2 completed")
        else:
            failed_downloads += 1
            print(f"❌ Video {i} download failed")
            
            # Update progress queue with error
            if progress_queue:
                progress_queue.update_video_step_status(
                    video_id,
                    'step_02_video_download',
                    'failed',
                    error=f"Download failed: {download_result.get('error', 'Unknown error')}"
                )
                print(f"   📊 Progress queue updated: step 2 failed")
        
        results.append({
            'video_id': video_id,
            'title': video_title,
            'download_result': download_result,
            'status': 'newly_downloaded'
        })
        
        # Small delay to be respectful to YouTube
        time.sleep(1)
    
    # Create summary
    summary = {
        'total_videos': len(videos),
        'successful_downloads': successful_downloads,
        'failed_downloads': failed_downloads,
        'existing_videos': existing_videos,
        'skipped_videos': skipped_videos,
        'success_rate': (successful_downloads / len(videos)) * 100 if videos else 0,
        'processing_timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    # Save summary
    summary_file = output_dir / "download_summary.json"
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Download summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"❌ Failed to save summary: {e}")
    
    return summary

def main():
    """Main function to download videos from playlist metadata."""
    print("🎬 Video Download & Processing - Step 02")
    print("=" * 50)
    
    # Setup
    logger = setup_logging()
    
    # Initialize progress queue
    progress_queue = get_progress_queue()
    print("✅ Progress queue initialized")
    
    # Check if yt-dlp is available
    if not check_yt_dlp_available():
        print("❌ yt-dlp is not available on your system")
        print("   Install it with: uv pip install yt-dlp")
        return
    
    print("✅ yt-dlp is available")
    
    # Setup directories
    step_01_dir = Path("step_01_raw")
    step_02_dir = Path("step_02_extracted_playlist_content")
    
    # Create output directory
    step_02_dir.mkdir(exist_ok=True)
    
    # Load metadata from Step 01
    metadata_file = step_01_dir / "playlist_and_video_metadata.json"
    if not metadata_file.exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        print("   Run Step 01 first: python test_step_01_integration.py")
        return
    
    playlist_data = load_playlist_metadata(metadata_file)
    if not playlist_data:
        return
    
    # Check if we have videos to process
    total_videos = playlist_data.get('total_videos_found', 0)
    if total_videos == 0:
        print("❌ No videos found in playlist metadata")
        return
    
    print(f"📊 Found {total_videos} videos to process")
    
    # Ask user if they want to limit videos for testing
    if total_videos > 10:
        print(f"\n⚠️  This playlist has {total_videos} videos")
        print("   For testing, you may want to limit the number of videos")
        print("   You can modify the max_videos parameter in the script")
    
    # Process videos (limit to 5 for testing - remove this limit for production)
    max_videos = 5  # Change this to None for all videos
    if max_videos:
        print(f"🔢 Limiting to {max_videos} videos for testing")
    
    summary = process_playlist_videos(playlist_data, step_02_dir, max_videos, progress_queue)
    
    # Display results
    print(f"\n📊 Download Summary:")
    print(f"Total Videos: {summary['total_videos']}")
    print(f"New Downloads: {summary['successful_downloads']}")
    print(f"Existing Videos: {summary['existing_videos']}")
    print(f"Failed Downloads: {summary['failed_downloads']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    total_processed = summary['successful_downloads'] + summary['existing_videos']
    if total_processed > 0:
        print(f"\n🚀 Ready for Step 03: Transcription")
        print(f"Total videos ready: {total_processed}")
        print(f"Videos located at: {step_02_dir}")
        print(f"Run: python step_03_transcription_webhook.py")
    else:
        print(f"\n❌ No videos were processed successfully")
        print(f"Check the logs for error details")

if __name__ == "__main__":
    main()
