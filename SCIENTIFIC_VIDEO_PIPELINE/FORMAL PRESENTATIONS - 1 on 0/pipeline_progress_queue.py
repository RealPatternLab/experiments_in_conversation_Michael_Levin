#!/usr/bin/env python3
"""
Pipeline Progress Queue Manager - Video Pipeline

Tracks the progress of video content through each step of the scientific video pipeline.
Provides thread-safe status updates and querying capabilities.
"""

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PipelineProgressQueue:
    """Manages pipeline progress tracking for video content."""
    
    def __init__(self, queue_file: Path = Path("pipeline_progress_queue.json")):
        self.queue_file = queue_file
        self.lock = threading.Lock()
        
        # Initialize the queue file if it doesn't exist
        self._initialize_queue_file()
    
    def _initialize_queue_file(self):
        """Initialize the progress queue file with default structure."""
        if not self.queue_file.exists():
            initial_data = {
                "pipeline_progress": {},
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "pipeline_type": "video_pipeline"
            }
            self._save_queue_data(initial_data)
            logger.info(f"Created new video pipeline progress queue: {self.queue_file}")
    
    def _load_queue_data(self) -> Dict[str, Any]:
        """Load the current queue data from file."""
        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file is corrupted, reinitialize
            logger.warning("Queue file corrupted, reinitializing...")
            self._initialize_queue_file()
            return self._load_queue_data()
    
    def _save_queue_data(self, data: Dict[str, Any]):
        """Save queue data to file."""
        try:
            # Ensure directory exists
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save queue data: {e}")
            raise
    
    def add_playlist(self, playlist_id: str, playlist_url: str) -> bool:
        """Add a new playlist to the progress queue."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                # Check if playlist already exists
                if playlist_id in data["pipeline_progress"]:
                    logger.warning(f"Playlist {playlist_id} already exists in progress queue")
                    return False
                
                # Initialize playlist with default status
                data["pipeline_progress"][playlist_id] = {
                    "playlist_url": playlist_url,
                    "step_01_playlist_processing": "complete",
                    "step_02_video_download": "pending",
                    "step_03_transcription": "pending",
                    "step_04_metadata_enhancement": "pending",
                    "step_05_frame_extraction": "pending",
                    "step_06_frame_chunk_alignment": "pending",
                    "step_07_consolidated_embedding": "pending",
                    "step_08_archive": "pending",
                    "current_status": "processing",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "video_count": 0,
                    "total_duration": 0,
                    "error_count": 0,
                    "errors": []
                }
                
                data["last_updated"] = datetime.now().isoformat()
                self._save_queue_data(data)
                
                logger.info(f"Added playlist {playlist_id} to progress queue")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add playlist {playlist_id}: {e}")
                return False
    
    def add_video(self, playlist_id: str, video_id: str, video_title: str) -> bool:
        """Add a video to a playlist's progress tracking."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if playlist_id not in data["pipeline_progress"]:
                    logger.error(f"Playlist {playlist_id} not found in progress queue")
                    return False
                
                playlist = data["pipeline_progress"][playlist_id]
                
                # Initialize videos list if it doesn't exist
                if "videos" not in playlist:
                    playlist["videos"] = {}
                
                # Add video with default status
                playlist["videos"][video_id] = {
                    "title": video_title,
                    "step_02_video_download": "pending",
                    "step_03_transcription": "pending",
                    "step_04_metadata_enhancement": "pending",
                    "step_05_frame_extraction": "pending",
                    "step_06_frame_chunk_alignment": "pending",
                    "step_07_consolidated_embedding": "pending",
                    "step_08_archive": "pending",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "duration": 0,
                    "frame_count": 0,
                    "chunk_count": 0,
                    "error_count": 0,
                    "errors": []
                }
                
                playlist["video_count"] = len(playlist["videos"])
                playlist["last_updated"] = datetime.now().isoformat()
                data["last_updated"] = datetime.now().isoformat()
                
                self._save_queue_data(data)
                
                logger.info(f"Added video {video_id} to playlist {playlist_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add video {video_id} to playlist {playlist_id}: {e}")
                return False
    
    def update_step_status(self, playlist_id: str, step_name: str, status: str, 
                          video_id: Optional[str] = None, details: Optional[Dict] = None) -> bool:
        """
        Update the status of a pipeline step.
        
        Args:
            playlist_id: ID of the playlist
            step_name: Name of the step to update
            status: New status (pending, processing, complete, failed)
            video_id: Optional video ID if updating video-specific step
            details: Optional additional details about the step
        """
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if playlist_id not in data["pipeline_progress"]:
                    logger.error(f"Playlist {playlist_id} not found in progress queue")
                    return False
                
                playlist = data["pipeline_progress"][playlist_id]
                
                if video_id and video_id in playlist.get("videos", {}):
                    # Update video-specific step
                    video = playlist["videos"][video_id]
                    video[step_name] = status
                    video["last_updated"] = datetime.now().isoformat()
                    
                    if details:
                        for key, value in details.items():
                            video[key] = value
                    
                    logger.info(f"Updated {step_name} for video {video_id} in playlist {playlist_id}: {status}")
                else:
                    # Update playlist-level step
                    playlist[step_name] = status
                    logger.info(f"Updated {step_name} for playlist {playlist_id}: {status}")
                
                playlist["last_updated"] = datetime.now().isoformat()
                data["last_updated"] = datetime.now().isoformat()
                
                self._save_queue_data(data)
                return True
                
            except Exception as e:
                logger.error(f"Failed to update step status: {e}")
                return False
    
    def get_playlist_status(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a playlist."""
        try:
            data = self._load_queue_data()
            return data["pipeline_progress"].get(playlist_id)
        except Exception as e:
            logger.error(f"Failed to get playlist status: {e}")
            return None
    
    def get_all_playlists(self) -> Dict[str, Any]:
        """Get all playlists and their status."""
        try:
            data = self._load_queue_data()
            return data["pipeline_progress"]
        except Exception as e:
            logger.error(f"Failed to get all playlists: {e}")
            return {}
    
    def get_pending_playlists(self, step_name: str) -> List[str]:
        """Get list of playlist IDs pending a specific step."""
        try:
            data = self._load_queue_data()
            pending = []
            
            for playlist_id, playlist in data["pipeline_progress"].items():
                if playlist.get(step_name) == "pending":
                    pending.append(playlist_id)
            
            return pending
            
        except Exception as e:
            logger.error(f"Failed to get pending playlists: {e}")
            return []
    
    def get_pending_videos(self, playlist_id: str, step_name: str) -> List[str]:
        """Get list of video IDs pending a specific step within a playlist."""
        try:
            data = self._load_queue_data()
            
            if playlist_id not in data["pipeline_progress"]:
                return []
            
            playlist = data["pipeline_progress"][playlist_id]
            pending = []
            
            for video_id, video in playlist.get("videos", {}).items():
                if video.get(step_name) == "pending":
                    pending.append(video_id)
            
            return pending
            
        except Exception as e:
            logger.error(f"Failed to get pending videos: {e}")
            return []
    
    def mark_step_complete(self, playlist_id: str, step_name: str, 
                          video_id: Optional[str] = None, details: Optional[Dict] = None) -> bool:
        """Mark a step as complete with optional details."""
        return self.update_step_status(playlist_id, step_name, "complete", video_id, details)
    
    def mark_step_failed(self, playlist_id: str, step_name: str, error_message: str,
                         video_id: Optional[str] = None) -> bool:
        """Mark a step as failed with error message."""
        details = {
            "error_message": error_message,
            "failed_at": datetime.now().isoformat()
        }
        
        if video_id:
            # Add error to video's error list
            try:
                data = self._load_queue_data()
                if playlist_id in data["pipeline_progress"]:
                    playlist = data["pipeline_progress"][playlist_id]
                    if video_id in playlist.get("videos", {}):
                        video = playlist["videos"][video_id]
                        if "errors" not in video:
                            video["errors"] = []
                        video["errors"].append(error_message)
                        video["error_count"] = len(video["errors"])
            except Exception as e:
                logger.error(f"Failed to add error to video: {e}")
        
        return self.update_step_status(playlist_id, step_name, "failed", video_id, details)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire pipeline status."""
        try:
            data = self._load_queue_data()
            playlists = data["pipeline_progress"]
            
            total_playlists = len(playlists)
            total_videos = sum(len(p.get("videos", {})) for p in playlists.values())
            
            # Count statuses for each step
            step_statuses = {}
            for step in ["step_01_playlist_processing", "step_02_video_download", 
                        "step_03_transcription", "step_04_metadata_enhancement",
                        "step_05_frame_extraction", "step_06_frame_chunk_alignment",
                        "step_07_consolidated_embedding", "step_08_archive"]:
                step_statuses[step] = {"pending": 0, "processing": 0, "complete": 0, "failed": 0}
            
            for playlist in playlists.values():
                for step in step_statuses:
                    status = playlist.get(step, "pending")
                    step_statuses[step][status] += 1
            
            summary = {
                "total_playlists": total_playlists,
                "total_videos": total_videos,
                "step_statuses": step_statuses,
                "last_updated": data.get("last_updated"),
                "pipeline_type": data.get("pipeline_type", "video_pipeline")
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get pipeline summary: {e}")
            return {}
    
    def cleanup_completed_playlists(self) -> int:
        """Remove playlists that have completed all steps."""
        with self.lock:
            try:
                data = self._load_queue_data()
                playlists = data["pipeline_progress"]
                
                completed_playlists = []
                for playlist_id, playlist in playlists.items():
                    # Check if all steps are complete
                    all_complete = all(
                        playlist.get(step) == "complete" 
                        for step in ["step_01_playlist_processing", "step_02_video_download",
                                    "step_03_transcription", "step_04_metadata_enhancement",
                                    "step_05_frame_extraction", "step_06_frame_chunk_alignment",
                                    "step_07_consolidated_embedding", "step_08_archive"]
                    )
                    
                    if all_complete:
                        completed_playlists.append(playlist_id)
                
                # Remove completed playlists
                for playlist_id in completed_playlists:
                    del playlists[playlist_id]
                
                if completed_playlists:
                    data["last_updated"] = datetime.now().isoformat()
                    self._save_queue_data(data)
                    logger.info(f"Cleaned up {len(completed_playlists)} completed playlists")
                
                return len(completed_playlists)
                
            except Exception as e:
                logger.error(f"Failed to cleanup completed playlists: {e}")
                return 0


def main():
    """Test the progress queue functionality."""
    queue = PipelineProgressQueue()
    
    print("🎬 Video Pipeline Progress Queue Test")
    print("=" * 40)
    
    # Test adding a playlist
    playlist_id = "PL6SlweOjqYXzKV6C63v17Y_hm8EXKW7wv"
    playlist_url = "https://www.youtube.com/playlist?list=PL6SlweOjqYXzKV6C63v17Y_hm8EXKW7wv"
    
    success = queue.add_playlist(playlist_id, playlist_url)
    print(f"Added playlist: {success}")
    
    # Test adding a video
    video_id = "test_video_123"
    video_title = "Test Video Title"
    success = queue.add_video(playlist_id, video_id, video_title)
    print(f"Added video: {success}")
    
    # Test updating step status
    success = queue.mark_step_complete(playlist_id, "step_01_playlist_processing")
    print(f"Marked step 01 complete: {success}")
    
    # Get summary
    summary = queue.get_pipeline_summary()
    print(f"\nPipeline Summary:")
    print(f"Total Playlists: {summary['total_playlists']}")
    print(f"Total Videos: {summary['total_videos']}")
    
    print(f"\n✅ Progress queue test completed successfully!")


if __name__ == "__main__":
    main()

