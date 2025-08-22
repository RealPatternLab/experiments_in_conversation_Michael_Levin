#!/usr/bin/env python3
"""
Conversations Video Pipeline Progress Queue Manager

Tracks the progress of videos and playlists through each step of the conversations video pipeline.
Provides thread-safe status updates and querying capabilities for the 8-step video processing workflow.
Specialized for handling conversations between Michael Levin and other researchers.
"""

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConversationsPipelineProgressQueue:
    """Manages pipeline progress tracking for conversation videos and playlists."""
    
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
                "speaker_identification": {},  # Track speaker labeling progress
                "qa_extraction": {},          # Track Q&A extraction progress
                "pipeline_info": {
                    "total_steps": 8,
                    "steps": [
                        "step_01_playlist_processing",
                        "step_02_video_download",
                        "step_03_transcription",
                        "step_04_extract_chunks",
                        "step_05_frame_extraction",
                        "step_06_frame_chunk_alignment",
                        "step_07_faiss_embeddings",
                        "step_08_cleanup"
                    ],
                    "pipeline_type": "conversations_1_on_1",
                    "description": "Pipeline for conversations between Michael Levin and other researchers"
                },
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            self._save_queue_data(initial_data)
            logger.info(f"Created new conversations pipeline progress queue: {self.queue_file}")
    
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
                
                # Initialize video with default status for all 8 steps
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
                    "step_01_playlist_processing": "completed",
                    "step_02_video_download": "pending",
                    "step_03_transcription": "pending",
                    "step_04_extract_chunks": "pending",
                    "step_05_frame_extraction": "pending",
                    "step_06_frame_chunk_alignment": "pending",
                    "step_07_faiss_embeddings": "pending",
                    "step_08_cleanup": "pending",
                    "speaker_identification": "pending",  # Track speaker labeling
                    "qa_extraction": "pending",          # Track Q&A extraction
                    "processing_metadata": {
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                        "pipeline_type": "conversations_1_on_1"
                    }
                }
                
                data["last_updated"] = datetime.now().isoformat()
                self._save_queue_data(data)
                
                logger.info(f"Added video {video_id} to progress queue")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add video {video_id}: {e}")
                return False
    
    def update_video_step_status(self, video_id: str, step: str, status: str, metadata: Dict[str, Any] = None) -> bool:
        """Update the status of a specific step for a video."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                # Update step status
                data["pipeline_progress"][video_id][step] = status
                
                # Update processing metadata
                if metadata:
                    if "processing_metadata" not in data["pipeline_progress"][video_id]:
                        data["pipeline_progress"][video_id]["processing_metadata"] = {}
                    
                    data["pipeline_progress"][video_id]["processing_metadata"].update(metadata)
                
                data["pipeline_progress"][video_id]["processing_metadata"]["last_updated"] = datetime.now().isoformat()
                data["last_updated"] = datetime.now().isoformat()
                
                self._save_queue_data(data)
                
                logger.info(f"Updated {video_id} {step} status to {status}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update video step status: {e}")
                return False
    
    def get_video_status(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a video."""
        with self.lock:
            try:
                data = self._load_queue_data()
                return data["pipeline_progress"].get(video_id)
            except Exception as e:
                logger.error(f"Failed to get video status: {e}")
                return None
    
    def get_video_step_status(self, video_id: str, step: str) -> Optional[str]:
        """Get the status of a specific step for a video."""
        video_status = self.get_video_status(video_id)
        if video_status:
            return video_status.get(step)
        return None
    
    def get_pending_videos_for_step(self, step: str) -> List[str]:
        """Get all videos that are pending for a specific step."""
        with self.lock:
            try:
                data = self._load_queue_data()
                pending_videos = []
                
                for video_id, video_data in data["pipeline_progress"].items():
                    if video_data.get(step) == "pending":
                        pending_videos.append(video_id)
                
                return pending_videos
                
            except Exception as e:
                logger.error(f"Failed to get pending videos for step: {e}")
                return []
    
    def get_completed_videos_for_step(self, step: str) -> List[str]:
        """Get all videos that have completed a specific step."""
        with self.lock:
            try:
                data = self._load_queue_data()
                completed_videos = []
                
                for video_id, video_data in data["pipeline_progress"].items():
                    if video_data.get(step) == "completed":
                        completed_videos.append(video_id)
                
                return completed_videos
                
            except Exception as e:
                logger.error(f"Failed to get completed videos for step: {e}")
                return []
    
    def update_speaker_identification(self, video_id: str, speakers: Dict[str, str], metadata: Dict[str, Any] = None) -> bool:
        """Update speaker identification for a video."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                # Update speaker identification
                data["pipeline_progress"][video_id]["speaker_identification"] = "completed"
                
                # Store speaker mapping
                if "speaker_mapping" not in data["pipeline_progress"][video_id]:
                    data["pipeline_progress"][video_id]["speaker_mapping"] = {}
                
                data["pipeline_progress"][video_id]["speaker_mapping"] = speakers
                
                # Update processing metadata
                if metadata:
                    if "processing_metadata" not in data["pipeline_progress"][video_id]:
                        data["pipeline_progress"][video_id]["processing_metadata"] = {}
                    
                    data["pipeline_progress"][video_id]["processing_metadata"].update(metadata)
                
                data["pipeline_progress"][video_id]["processing_metadata"]["last_updated"] = datetime.now().isoformat()
                data["last_updated"] = datetime.now().isoformat()
                
                self._save_queue_data(data)
                
                logger.info(f"Updated speaker identification for {video_id}: {speakers}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update speaker identification: {e}")
                return False
    
    def update_qa_extraction(self, video_id: str, qa_pairs: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> bool:
        """Update Q&A extraction for a video."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                # Update Q&A extraction
                data["pipeline_progress"][video_id]["qa_extraction"] = "completed"
                
                # Store Q&A pairs count
                if "qa_metadata" not in data["pipeline_progress"][video_id]:
                    data["pipeline_progress"][video_id]["qa_metadata"] = {}
                
                data["pipeline_progress"][video_id]["qa_metadata"] = {
                    "total_qa_pairs": len(qa_pairs),
                    "extraction_timestamp": datetime.now().isoformat()
                }
                
                # Update processing metadata
                if metadata:
                    if "processing_metadata" not in data["pipeline_progress"][video_id]:
                        data["pipeline_progress"][video_id]["processing_metadata"] = {}
                    
                    data["pipeline_progress"][video_id]["processing_metadata"].update(metadata)
                
                data["pipeline_progress"][video_id]["processing_metadata"]["last_updated"] = datetime.now().isoformat()
                data["last_updated"] = datetime.now().isoformat()
                
                self._save_queue_data(data)
                
                logger.info(f"Updated Q&A extraction for {video_id}: {len(qa_pairs)} pairs")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update Q&A extraction: {e}")
                return False
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire pipeline status."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                total_videos = len(data["pipeline_progress"])
                total_playlists = len(data["playlist_progress"])
                
                # Count videos by step status
                step_counts = {}
                for step in data["pipeline_info"]["steps"]:
                    step_counts[step] = {
                        "completed": 0,
                        "pending": 0,
                        "failed": 0
                    }
                
                for video_data in data["pipeline_progress"].values():
                    for step in data["pipeline_info"]["steps"]:
                        status = video_data.get(step, "unknown")
                        if status in step_counts[step]:
                            step_counts[step][status] += 1
                
                # Count speaker identification and Q&A extraction
                speaker_identified = sum(1 for v in data["pipeline_progress"].values() 
                                      if v.get("speaker_identification") == "completed")
                qa_extracted = sum(1 for v in data["pipeline_progress"].values() 
                                 if v.get("qa_extraction") == "completed")
                
                return {
                    "total_videos": total_videos,
                    "total_playlists": total_playlists,
                    "step_counts": step_counts,
                    "speaker_identification": {
                        "completed": speaker_identified,
                        "pending": total_videos - speaker_identified
                    },
                    "qa_extraction": {
                        "completed": qa_extracted,
                        "pending": total_videos - qa_extracted
                    },
                    "last_updated": data["last_updated"]
                }
                
            except Exception as e:
                logger.error(f"Failed to get pipeline summary: {e}")
                return {}
    
    def reset_video_step(self, video_id: str, step: str) -> bool:
        """Reset a specific step for a video to pending status."""
        return self.update_video_step_status(video_id, step, "pending")
    
    def reset_video(self, video_id: str) -> bool:
        """Reset all steps for a video to pending status."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if video_id not in data["pipeline_progress"]:
                    logger.error(f"Video {video_id} not found in progress queue")
                    return False
                
                # Reset all steps to pending
                for step in data["pipeline_info"]["steps"]:
                    data["pipeline_progress"][video_id][step] = "pending"
                
                # Reset speaker identification and Q&A extraction
                data["pipeline_progress"][video_id]["speaker_identification"] = "pending"
                data["pipeline_progress"][video_id]["qa_extraction"] = "pending"
                
                # Update metadata
                data["pipeline_progress"][video_id]["processing_metadata"]["last_updated"] = datetime.now().isoformat()
                data["last_updated"] = datetime.now().isoformat()
                
                self._save_queue_data(data)
                
                logger.info(f"Reset all steps for video {video_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to reset video: {e}")
                return False

# Global instance
_progress_queue_instance = None

def get_progress_queue() -> ConversationsPipelineProgressQueue:
    """Get the global progress queue instance."""
    global _progress_queue_instance
    if _progress_queue_instance is None:
        _progress_queue_instance = ConversationsPipelineProgressQueue()
    return _progress_queue_instance
