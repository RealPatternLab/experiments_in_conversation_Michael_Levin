#!/usr/bin/env python3
"""
Step 08: Cleanup for Conversations Pipeline
Removes unnecessary files to free up disk space while preserving essential data.
"""

import os
import json
import logging
from pathlib import Path
from pipeline_progress_queue import get_progress_queue

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('cleanup')

def get_referenced_frames(video_id: str) -> set:
    """Get set of frame IDs that are referenced in chunk alignments"""
    base_dir = Path(__file__).parent
    alignment_file = base_dir / "step_06_frame_chunk_alignment" / f"{video_id}_alignments.json"
    
    if not alignment_file.exists():
        return set()
    
    try:
        with open(alignment_file, 'r') as f:
            alignments = json.load(f)
        
        referenced_frames = set()
        for chunk in alignments:
            # With new single-frame approach, check for 'frame' field
            if 'frame' in chunk and chunk['frame']:
                frame_id = chunk['frame'].get('frame_id', '')
                if frame_id:
                    referenced_frames.add(frame_id)
        
        return referenced_frames
    except Exception as e:
        logger.warning(f"Failed to read alignments for {video_id}: {e}")
        return set()

def cleanup_unreferenced_frames(dry_run: bool = False) -> float:
    """Remove frame files that are not referenced in chunk alignments"""
    logger.info("🖼️ Cleaning up unreferenced frame files...")
    
    base_dir = Path(__file__).parent
    frames_dir = base_dir / "step_05_frames"
    if not frames_dir.exists():
        logger.warning("⚠️ Frames directory not found, skipping frame cleanup")
        return 0.0
    
    # Check if alignment files exist before proceeding
    alignment_dir = base_dir / "step_06_frame_chunk_alignment"
    if not alignment_dir.exists():
        logger.error("❌ Frame-chunk alignment directory not found!")
        logger.error("Cannot determine which frames are referenced. Skipping frame cleanup for safety.")
        return 0.0
    
    # Count alignment files to ensure we have the expected data
    alignment_files = list(alignment_dir.glob("*_alignments.json"))
    if not alignment_files:
        logger.error("❌ No alignment files found!")
        logger.error("Cannot determine which frames are referenced. Skipping frame cleanup for safety.")
        return 0.0
    
    logger.info(f"🔍 Found {len(alignment_files)} alignment files")
    
    total_freed = 0.0
    frames_checked = 0
    frames_removed = 0
    
    for video_subdir in frames_dir.iterdir():
        if video_subdir.is_dir():
            video_id = video_subdir.name
            referenced_frames = get_referenced_frames(video_id)
            
            if not referenced_frames:
                logger.warning(f"⚠️ No referenced frames found for {video_id}. Skipping frame cleanup for this video.")
                continue
            
            logger.info(f"🔍 Checking frames for {video_id}: {len(referenced_frames)} referenced")
            
            for frame_file in video_subdir.iterdir():
                if frame_file.is_file() and frame_file.suffix == '.jpg':
                    frames_checked += 1
                    # Extract frame ID from filename (e.g., "1IhHJgGNBUc_frame_001_15.0s.jpg")
                    frame_name = frame_file.stem
                    # The referenced frame IDs are in format "1IhHJgGNBUc_frame_001" (without timestamp)
                    if '_frame_' in frame_name:
                        parts = frame_name.split('_frame_')
                        if len(parts) >= 2:
                            # parts[1] contains "001_15.0s", we need just "001"
                            frame_number = parts[1].split('_')[0]  # Get just the number part
                            frame_id = parts[0] + '_frame_' + frame_number
                        else:
                            frame_id = frame_name  # Fallback
                    else:
                        frame_id = frame_name  # Fallback
                    
                    if frame_id not in referenced_frames:
                        file_size = frame_file.stat().st_size / (1024 * 1024)  # MB
                        if not dry_run:
                            frame_file.unlink()
                            total_freed += file_size
                            frames_removed += 1
                            logger.debug(f"🗑️ Removed unreferenced frame: {frame_file.name}")
                        else:
                            logger.info(f"🔍 Would remove unreferenced frame: {frame_file.name} ({file_size:.1f} MB)")
    
    if dry_run:
        logger.info(f"📊 Frame cleanup preview: {frames_checked} frames checked, {frames_removed} would be removed")
    else:
        logger.info(f"📊 Frame cleanup summary: {frames_checked} frames checked, {frames_removed} removed")
        logger.info(f"✅ Frame cleanup completed. Freed {total_freed:.1f} MB")
    
    return total_freed

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
    
    # Add frame cleanup (the main space saver!)
    logger.info("🖼️ Starting frame cleanup...")
    frame_space_freed = cleanup_unreferenced_frames(dry_run=False)
    total_removed += frame_space_freed
    
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
