#!/usr/bin/env python3
"""
Video Pipeline Progress Queue Manager

Tracks the progress of videos and playlists through each step of the scientific video pipeline.
Provides thread-safe status updates and querying capabilities for the 7-step video processing workflow.
"""

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VideoPipelineProgressQueue:
    """Manages pipeline progress tracking for videos and playlists."""
    
    def __init__(self, queue_file: Path = Path("logs/pipeline_progress_queue.json")):
        self.queue_file = queue_file
        self.lock = threading.Lock()
        
        # Initialize the queue file if it doesn't exist
        self._initialize_queue_file()
    
    def _initialize_queue_file(self):
        """Initialize the progress queue file with default structure."""
        if not self.queue_file.exists():
            initial_data = {
                "pipeline_progress": {},
                "playlist_progress": {},
                "pipeline_info": {
                    "total_steps": 8,
                    "steps": [
                        "step_01_playlist_processing",
                        "step_02_video_download",
                        "step_03_transcription",
                        "step_04_semantic_chunking",
                        "step_05_frame_extraction",
                        "step_06_frame_chunk_alignment",
                        "step_07_consolidated_embedding",
                        "step_08_cleanup"
                    ]
                },
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            self._save_queue_data(initial_data)
            logger.info(f"Created new video pipeline progress queue: {self.queue_file}")
    
    def _load_queue_data(self) -> Dict[str, Any]:
        """Load the current queue data from file."""
        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Migrate old format to new format if needed
                self._migrate_old_format(data)
                
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            # If file is corrupted, reinitialize
            logger.warning("Queue file corrupted, reinitializing...")
            self._initialize_queue_file()
            return self._load_queue_data()
    
    def _migrate_old_format(self, data: Dict[str, Any]):
        """Migrate old format data to new format with change tracking."""
        if "pipeline_progress" not in data:
            return
        
        migrated = False
        for video_id, video_data in data["pipeline_progress"].items():
            # Check if this video needs migration
            if isinstance(video_data.get("video_title"), str):
                # Migrate old string format to new structure
                old_title = video_data["video_title"]
                old_playlist_id = video_data.get("playlist_id")
                
                # Create new structure
                video_data["video_title"] = {
                    "current": old_title,
                    "history": [
                        {
                            "timestamp": video_data.get("created_at", datetime.now().isoformat()),
                            "value": old_title,
                            "action": "migrated"
                        }
                    ]
                }
                
                video_data["playlist_ids"] = {
                    "current": [old_playlist_id] if old_playlist_id else [],
                    "history": [
                        {
                            "timestamp": video_data.get("created_at", datetime.now().isoformat()),
                            "playlist_id": old_playlist_id,
                            "action": "migrated"
                        }
                    ] if old_playlist_id else []
                }
                
                video_data["upload_date"] = {
                    "current": None,
                    "history": []
                }
                
                # Remove old fields
                if "playlist_id" in video_data:
                    del video_data["playlist_id"]
                
                migrated = True
                logger.info(f"Migrated video {video_id} to new format")
        
        if migrated:
            data["last_updated"] = datetime.now().isoformat()
            self._save_queue_data(data)
            logger.info("Migration completed")
    
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
    
    def add_playlist(self, playlist_id: str, playlist_title: str, video_count: int) -> bool:
        """Add a new playlist to the progress queue."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                # Check if playlist already exists
                if playlist_id in data["playlist_progress"]:
                    logger.warning(f"Playlist {playlist_id} already exists in progress queue")
                    return False
                
                # Initialize playlist with default status
                data["playlist_progress"][playlist_id] = {
                    "playlist_title": playlist_title,
                    "video_count": video_count,
                    "videos_processed": 0,
                    "videos_failed": 0,
                    "current_status": "processing",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "errors": []
                }
                
                data["last_updated"] = datetime.now().isoformat()
                self._save_queue_data(data)
                
                logger.info(f"Added playlist {playlist_id} to progress queue")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add playlist {playlist_id}: {e}")
                return False
    
    def add_video(self, video_id: str, video_title: str, playlist_id: str = None) -> bool:
        """Add a new video to the progress queue."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                # Check if video already exists
                if video_id in data["pipeline_progress"]:
                    logger.warning(f"Video {video_id} already exists in progress queue")
                    return False
                
                # Initialize video with default status for all 7 steps
                data["pipeline_progress"][video_id] = {
                    "video_title": {
                        "current": video_title,
                        "history": [
                            {
                                "timestamp": datetime.now().isoformat(),
                                "value": video_title,
                                "action": "initial"
                            }
                        ]
                    },
                    "playlist_ids": {
                        "current": [playlist_id] if playlist_id else [],
                        "history": [
                            {
                                "timestamp": datetime.now().isoformat(),
                                "playlist_id": playlist_id,
                                "action": "added"
                            }
                        ] if playlist_id else []
                    },
                    "upload_date": {
                        "current": None,
                        "history": []
                    },
                    "step_01_playlist_processing": "pending",
                    "step_02_video_download": "pending",
                    "step_03_transcription": "pending",
                    "step_04_semantic_chunking": "pending",
                    "step_05_frame_extraction": "pending",
                    "step_06_frame_chunk_alignment": "pending",
                    "step_07_consolidated_embedding": "pending",
                    "current_status": "pending",
                    "current_step": "step_01_playlist_processing",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "chunk_count": 0,
                    "frame_count": 0,
                    "error_count": 0,
                    "errors": [],
                    "processing_metadata": {
                        "duration": None,
                        "file_size_mb": None,
                        "transcript_length": None,
                        "chunk_count": None,
                        "frame_count": None
                    }
                }
                
                data["last_updated"] = datetime.now().isoformat()
                self._save_queue_data(data)
                
                logger.info(f"Added video {video_id} to progress queue")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add video {video_id}: {e}")
                return False
    
    def update_video_step_status(self, video_id: str, step: str, status: str, 
                                metadata: Dict[str, Any] = None, error: str = None) -> bool:
        """Update the status of a specific step for a video."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                video_data = data["pipeline_progress"][video_id]
                
                # Update step status
                if step in video_data:
                    video_data[step] = status
                    video_data["last_updated"] = datetime.now().isoformat()
                    
                    # Update current step if this step is being processed
                    if status == "processing":
                        video_data["current_step"] = step
                    
                    # Update metadata if provided
                    if metadata:
                        video_data["processing_metadata"].update(metadata)
                    
                    # Handle errors
                    if error:
                        video_data["error_count"] += 1
                        error_entry = {
                            "step": step,
                            "error": error,
                            "timestamp": datetime.now().isoformat()
                        }
                        video_data["errors"].append(error_entry)
                    
                    # Update current status based on step statuses
                    self._update_video_current_status(video_data)
                    
                    # Update playlist progress for all playlists this video belongs to
                    playlist_ids = video_data.get("playlist_ids", {}).get("current", [])
                    for playlist_id in playlist_ids:
                        self._update_playlist_progress(playlist_id, data)
                    
                    data["last_updated"] = datetime.now().isoformat()
                    self._save_queue_data(data)
                    
                    logger.info(f"Updated {video_id} {step}: {status}")
                    return True
                else:
                    logger.error(f"Invalid step {step} for video {video_id}")
                    return False
                
            except Exception as e:
                logger.error(f"Failed to update video step status: {e}")
                return False
    
    def update_video_title(self, video_id: str, new_title: str, reason: str = "updated") -> bool:
        """Update the video title and track the change."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                video_data = data["pipeline_progress"][video_id]
                current_title = video_data["video_title"]["current"]
                
                if current_title != new_title:
                    # Add to history
                    video_data["video_title"]["history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "value": new_title,
                        "action": reason,
                        "previous_value": current_title
                    })
                    
                    # Update current value
                    video_data["video_title"]["current"] = new_title
                    video_data["last_updated"] = datetime.now().isoformat()
                    
                    data["last_updated"] = datetime.now().isoformat()
                    self._save_queue_data(data)
                    
                    logger.info(f"Updated video {video_id} title: '{current_title}' → '{new_title}'")
                    return True
                else:
                    logger.info(f"Video {video_id} title unchanged: '{new_title}'")
                    return True
                
            except Exception as e:
                logger.error(f"Failed to update video title: {e}")
                return False
    
    def add_video_to_playlist(self, video_id: str, playlist_id: str, reason: str = "added") -> bool:
        """Add a video to a playlist and track the change."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                video_data = data["pipeline_progress"][video_id]
                current_playlists = video_data["playlist_ids"]["current"]
                
                if playlist_id not in current_playlists:
                    # Add to history
                    video_data["playlist_ids"]["history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "playlist_id": playlist_id,
                        "action": reason
                    })
                    
                    # Update current list
                    current_playlists.append(playlist_id)
                    video_data["last_updated"] = datetime.now().isoformat()
                    
                    # Update playlist progress
                    self._update_playlist_progress(playlist_id, data)
                    
                    data["last_updated"] = datetime.now().isoformat()
                    self._save_queue_data(data)
                    
                    logger.info(f"Added video {video_id} to playlist {playlist_id}")
                    return True
                else:
                    logger.info(f"Video {video_id} already in playlist {playlist_id}")
                    return True
                
            except Exception as e:
                logger.error(f"Failed to add video to playlist: {e}")
                return False
    
    def remove_video_from_playlist(self, video_id: str, playlist_id: str, reason: str = "removed") -> bool:
        """Remove a video from a playlist and track the change."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                video_data = data["pipeline_progress"][video_id]
                current_playlists = video_data["playlist_ids"]["current"]
                
                if playlist_id in current_playlists:
                    # Add to history
                    video_data["playlist_ids"]["history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "playlist_id": playlist_id,
                        "action": reason
                    })
                    
                    # Update current list
                    current_playlists.remove(playlist_id)
                    video_data["last_updated"] = datetime.now().isoformat()
                    
                    # Update playlist progress
                    self._update_playlist_progress(playlist_id, data)
                    
                    data["last_updated"] = datetime.now().isoformat()
                    self._save_queue_data(data)
                    
                    logger.info(f"Removed video {video_id} from playlist {playlist_id}")
                    return True
                else:
                    logger.info(f"Video {video_id} not in playlist {playlist_id}")
                    return True
                
            except Exception as e:
                logger.error(f"Failed to remove video from playlist: {e}")
                return False
    
    def update_video_upload_date(self, video_id: str, upload_date: str, reason: str = "discovered") -> bool:
        """Update the video upload date and track the change."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                video_data = data["pipeline_progress"][video_id]
                current_date = video_data["upload_date"]["current"]
                
                if current_date != upload_date:
                    # Add to history
                    video_data["upload_date"]["history"].append({
                        "timestamp": datetime.now().isoformat(),
                        "value": upload_date,
                        "action": reason,
                        "previous_value": current_date
                    })
                    
                    # Update current value
                    video_data["upload_date"]["current"] = upload_date
                    video_data["last_updated"] = datetime.now().isoformat()
                    
                    data["last_updated"] = datetime.now().isoformat()
                    self._save_queue_data(data)
                    
                    logger.info(f"Updated video {video_id} upload date: '{current_date}' → '{upload_date}'")
                    return True
                else:
                    logger.info(f"Video {video_id} upload date unchanged: '{upload_date}'")
                    return True
                
            except Exception as e:
                logger.error(f"Failed to update video upload date: {e}")
                return False
    
    def _update_video_current_status(self, video_data: Dict[str, Any]):
        """Update the current status of a video based on step statuses."""
        steps = [
            "step_01_playlist_processing",
            "step_02_video_download", 
            "step_03_transcription",
            "step_04_semantic_chunking",
            "step_05_frame_extraction",
            "step_06_frame_chunk_alignment",
            "step_07_consolidated_embedding"
        ]
        
        # Count statuses
        pending_count = sum(1 for step in steps if video_data.get(step) == "pending")
        processing_count = sum(1 for step in steps if video_data.get(step) == "processing")
        completed_count = sum(1 for step in steps if video_data.get(step) == "completed")
        failed_count = sum(1 for step in steps if video_data.get(step) == "failed")
        skipped_count = sum(1 for step in steps if video_data.get(step) == "skipped")
        
        # Determine current status
        if failed_count > 0:
            video_data["current_status"] = "failed"
        elif processing_count > 0:
            video_data["current_status"] = "processing"
        elif completed_count == len(steps):
            video_data["current_status"] = "completed"
        elif completed_count + skipped_count == len(steps):
            video_data["current_status"] = "completed"
        elif pending_count == len(steps):
            video_data["current_status"] = "pending"
        else:
            video_data["current_status"] = "partially_completed"
    
    def _update_playlist_progress(self, playlist_id: str, data: Dict[str, Any]):
        """Update playlist progress based on video statuses."""
        if playlist_id not in data["playlist_progress"]:
            return
        
        playlist_data = data["playlist_progress"][playlist_id]
        videos_in_playlist = [
            video_id for video_id, video_data in data["pipeline_progress"].items()
            if video_data.get("playlist_id") == playlist_id
        ]
        
        if not videos_in_playlist:
            return
        
        # Count video statuses
        completed_count = sum(1 for video_id in videos_in_playlist 
                            if data["pipeline_progress"][video_id]["current_status"] == "completed")
        failed_count = sum(1 for video_id in videos_in_playlist 
                          if data["pipeline_progress"][video_id]["current_status"] == "failed")
        
        playlist_data["videos_processed"] = completed_count
        playlist_data["videos_failed"] = failed_count
        playlist_data["last_updated"] = datetime.now().isoformat()
        
        # Update playlist status
        if failed_count == len(videos_in_playlist):
            playlist_data["current_status"] = "failed"
        elif completed_count == len(videos_in_playlist):
            playlist_data["current_status"] = "completed"
        elif completed_count + failed_count == len(videos_in_playlist):
            playlist_data["current_status"] = "completed"
        else:
            playlist_data["current_status"] = "processing"
    
    def get_video_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a video."""
        with self.lock:
            try:
                data = self._load_queue_data()
                return data["pipeline_progress"].get(video_id)
            except Exception as e:
                logger.error(f"Failed to get video status: {e}")
                return None
    
    def get_playlist_status(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a playlist."""
        with self.lock:
            try:
                data = self._load_queue_data()
                return data["playlist_progress"].get(playlist_id)
            except Exception as e:
                logger.error(f"Failed to get playlist status: {e}")
                return None
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire pipeline status."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                videos = data["pipeline_progress"]
                playlists = data["playlist_progress"]
                
                # Count video statuses
                total_videos = len(videos)
                completed_videos = sum(1 for v in videos.values() if v["current_status"] == "completed")
                processing_videos = sum(1 for v in videos.values() if v["current_status"] == "processing")
                failed_videos = sum(1 for v in videos.values() if v["current_status"] == "failed")
                pending_videos = sum(1 for v in videos.values() if v["current_status"] == "pending")
                
                # Count playlist statuses
                total_playlists = len(playlists)
                completed_playlists = sum(1 for p in playlists.values() if p["current_status"] == "completed")
                processing_playlists = sum(1 for p in playlists.values() if p["current_status"] == "processing")
                failed_playlists = sum(1 for p in playlists.values() if p["current_status"] == "failed")
                
                # Step completion counts
                step_counts = {}
                for step in data["pipeline_info"]["steps"]:
                    step_counts[step] = {
                        "completed": sum(1 for v in videos.values() if v.get(step) == "completed"),
                        "processing": sum(1 for v in videos.values() if v.get(step) == "processing"),
                        "failed": sum(1 for v in videos.values() if v.get(step) == "failed"),
                        "pending": sum(1 for v in videos.values() if v.get(step) == "pending"),
                        "skipped": sum(1 for v in videos.values() if v.get(step) == "skipped")
                    }
                
                summary = {
                    "total_videos": total_videos,
                    "video_statuses": {
                        "completed": completed_videos,
                        "processing": processing_videos,
                        "failed": failed_videos,
                        "pending": pending_videos
                    },
                    "total_playlists": total_playlists,
                    "playlist_statuses": {
                        "completed": completed_playlists,
                        "processing": processing_playlists,
                        "failed": failed_playlists
                    },
                    "step_progress": step_counts,
                    "last_updated": data["last_updated"]
                }
                
                return summary
                
            except Exception as e:
                logger.error(f"Failed to get pipeline summary: {e}")
                return {}
    
    def get_failed_videos(self) -> List[Dict[str, Any]]:
        """Get a list of videos that have failed processing."""
        with self.lock:
            try:
                data = self._load_queue_data()
                failed_videos = [
                    {"video_id": video_id, **video_data}
                    for video_id, video_data in data["pipeline_progress"].items()
                    if video_data["current_status"] == "failed"
                ]
                return failed_videos
            except Exception as e:
                logger.error(f"Failed to get failed videos: {e}")
                return []
    
    def get_stuck_videos(self, timeout_minutes: int = 30) -> List[Dict[str, Any]]:
        """Get videos that have been stuck in processing for too long."""
        with self.lock:
            try:
                data = self._load_queue_data()
                current_time = datetime.now()
                stuck_videos = []
                
                for video_id, video_data in data["pipeline_progress"].items():
                    if video_data["current_status"] == "processing":
                        last_updated = datetime.fromisoformat(video_data["last_updated"])
                        time_diff = current_time - last_updated
                        
                        if time_diff.total_seconds() > timeout_minutes * 60:
                            stuck_videos.append({
                                "video_id": video_id,
                                "stuck_for_minutes": time_diff.total_seconds() / 60,
                                **video_data
                            })
                
                return stuck_videos
            except Exception as e:
                logger.error(f"Failed to get stuck videos: {e}")
                return []
    
    def reset_video(self, video_id: str, step: str = None) -> bool:
        """Reset a video's progress, optionally for a specific step."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                video_data = data["pipeline_progress"][video_id]
                
                if step:
                    # Reset specific step
                    if step in video_data:
                        video_data[step] = "pending"
                        logger.info(f"Reset {video_id} {step} to pending")
                else:
                    # Reset all steps
                    steps = [
                        "step_01_playlist_processing",
                        "step_02_video_download",
                        "step_03_transcription",
                        "step_04_semantic_chunking",
                        "step_05_frame_extraction",
                        "step_06_frame_chunk_alignment",
                        "step_07_consolidated_embedding"
                    ]
                    
                    for step_name in steps:
                        video_data[step_name] = "pending"
                    
                    video_data["current_status"] = "pending"
                    video_data["current_step"] = "step_01_playlist_processing"
                    video_data["error_count"] = 0
                    video_data["errors"] = []
                    
                    logger.info(f"Reset {video_id} all steps to pending")
                
                video_data["last_updated"] = datetime.now().isoformat()
                data["last_updated"] = datetime.now().isoformat()
                self._save_queue_data(data)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to reset video: {e}")
                return False
    
    def cleanup_old_entries(self, days_old: int = 30) -> int:
        """Remove old completed entries from the queue."""
        with self.lock:
            try:
                data = self._load_queue_data()
                current_time = datetime.now()
                removed_count = 0
                
                # Remove old completed videos
                old_videos = []
                for video_id, video_data in data["pipeline_progress"].items():
                    if video_data["current_status"] == "completed":
                        created_at = datetime.fromisoformat(video_data["created_at"])
                        time_diff = current_time - created_at
                        
                        if time_diff.days > days_old:
                            old_videos.append(video_id)
                
                for video_id in old_videos:
                    del data["pipeline_progress"][video_id]
                    removed_count += 1
                
                # Remove old completed playlists
                old_playlists = []
                for playlist_id, playlist_data in data["playlist_progress"].items():
                    if playlist_data["current_status"] == "completed":
                        created_at = datetime.fromisoformat(playlist_data["created_at"])
                        time_diff = current_time - created_at
                        
                        if time_diff.days > days_old:
                            old_playlists.append(playlist_id)
                
                for playlist_id in old_playlists:
                    del data["playlist_progress"][playlist_id]
                
                if removed_count > 0:
                    data["last_updated"] = datetime.now().isoformat()
                    self._save_queue_data(data)
                    logger.info(f"Cleaned up {removed_count} old entries")
                
                return removed_count
                
            except Exception as e:
                logger.error(f"Failed to cleanup old entries: {e}")
                return 0


# Convenience functions for easy integration
def get_progress_queue() -> VideoPipelineProgressQueue:
    """Get a global progress queue instance."""
    return VideoPipelineProgressQueue()

def update_video_step(video_id: str, step: str, status: str, 
                     metadata: Dict[str, Any] = None, error: str = None) -> bool:
    """Convenience function to update video step status."""
    queue = get_progress_queue()
    return queue.update_video_step_status(video_id, step, status, metadata, error)

def get_pipeline_status() -> Dict[str, Any]:
    """Convenience function to get pipeline summary."""
    queue = get_progress_queue()
    return queue.get_pipeline_summary()

def update_video_title(video_id: str, new_title: str, reason: str = "updated") -> bool:
    """Convenience function to update video title."""
    queue = get_progress_queue()
    return queue.update_video_title(video_id, new_title, reason)

def add_video_to_playlist(video_id: str, playlist_id: str, reason: str = "added") -> bool:
    """Convenience function to add video to playlist."""
    queue = get_progress_queue()
    return queue.add_video_to_playlist(video_id, playlist_id, reason)

def remove_video_from_playlist(video_id: str, playlist_id: str, reason: str = "removed") -> bool:
    """Convenience function to remove video from playlist."""
    queue = get_progress_queue()
    return queue.remove_video_from_playlist(video_id, playlist_id, reason)

def update_video_upload_date(video_id: str, upload_date: str, reason: str = "discovered") -> bool:
    """Convenience function to update video upload date."""
    queue = get_progress_queue()
    return queue.update_video_upload_date(video_id, upload_date, reason)


if __name__ == "__main__":
    # Test the progress queue
    queue = VideoPipelineProgressQueue()
    
    # Add a test playlist and video
    queue.add_playlist("test_playlist_001", "Test Playlist", 1)
    queue.add_video("test_video_001", "Test Video", "test_playlist_001")
    
    # Update some steps
    queue.update_video_step_status("test_video_001", "step_01_playlist_processing", "completed")
    queue.update_video_step_status("test_video_001", "step_02_video_download", "processing")
    
    # Test change tracking
    print("\n=== Testing Change Tracking ===")
    
    # Update video title
    queue.update_video_title("test_video_001", "Updated Test Video", "title_updated")
    queue.update_video_title("test_video_001", "Final Test Video", "title_finalized")
    
    # Add to another playlist
    queue.add_video_to_playlist("test_video_001", "test_playlist_002", "cross_referenced")
    
    # Update upload date
    queue.update_video_upload_date("test_video_001", "20250816", "metadata_discovered")
    
    # Show video details
    video_status = queue.get_video_status("test_video_001")
    print("\nVideo Details:")
    print(f"Current Title: {video_status['video_title']['current']}")
    print(f"Current Playlists: {video_status['playlist_ids']['current']}")
    print(f"Current Upload Date: {video_status['upload_date']['current']}")
    
    print("\nTitle History:")
    for entry in video_status['video_title']['history']:
        print(f"  {entry['timestamp']}: {entry['action']} → '{entry['value']}'")
    
    print("\nPlaylist History:")
    for entry in video_status['playlist_ids']['history']:
        print(f"  {entry['timestamp']}: {entry['action']} playlist '{entry['playlist_id']}'")
    
    # Get summary
    summary = queue.get_pipeline_summary()
    print("\nPipeline Summary:")
    print(json.dumps(summary, indent=2))
