#!/usr/bin/env python3
"""
Step 08: Cleanup for Conversations Pipeline
Removes unnecessary files to free up disk space while preserving essential data.
"""

import os
import logging
from pathlib import Path
from pipeline_progress_queue import get_progress_queue

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('cleanup')

def cleanup_conversations_pipeline():
    """Clean up unnecessary files from the conversations pipeline"""
    logger.info("🧹 Starting cleanup for conversations pipeline...")
    
    # Get progress queue
    progress_queue = get_progress_queue()
    
    # Define cleanup actions
    cleanup_actions = [
        {
            'name': 'Remove temporary audio files',
            'path': 'step_03_transcription/*_audio.wav',
            'description': 'Temporary audio files used for transcription'
        },
        {
            'name': 'Remove intermediate chunk files',
            'path': 'step_04_extract_chunks/*_levin_chunks.json',
            'description': 'Intermediate Levin chunks (enhanced chunks are preserved)'
        },
        {
            'name': 'Remove intermediate Q&A files',
            'path': 'step_04_extract_chunks/*_qa_pairs.json',
            'description': 'Intermediate Q&A pairs (enhanced chunks are preserved)'
        }
    ]
    
    total_removed = 0
    
    for action in cleanup_actions:
        try:
            # Find files matching the pattern
            pattern = Path(action['path'])
            if pattern.parent.exists():
                matching_files = list(pattern.parent.glob(pattern.name))
                
                for file_path in matching_files:
                    if file_path.exists():
                        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                        file_path.unlink()
                        total_removed += file_size
                        logger.info(f"   🗑️ Removed: {file_path.name} ({file_size:.2f} MB)")
                
                if matching_files:
                    logger.info(f"✅ {action['name']}: {len(matching_files)} files removed")
                else:
                    logger.info(f"ℹ️ {action['name']}: No files to remove")
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up {action['name']}: {e}")
    
    # Log cleanup summary
    logger.info(f"🧹 Cleanup completed: {total_removed:.2f} MB freed")
    
    # Update progress queue if available
    try:
        # Find all video IDs that have completed step 7
        if hasattr(progress_queue, 'get_completed_videos_for_step'):
            completed_videos = progress_queue.get_completed_videos_for_step('step_07_faiss_embeddings')
            for video_id in completed_videos:
                progress_queue.update_video_step_status(video_id, 'step_08_cleanup', 'completed')
                logger.info(f"📊 Progress queue updated: step 8 completed for {video_id}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to update progress queue: {e}")
    
    logger.info("✅ Cleanup step completed successfully")
    return True

def main():
    """Main cleanup function"""
    try:
        cleanup_conversations_pipeline()
        return 0
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
