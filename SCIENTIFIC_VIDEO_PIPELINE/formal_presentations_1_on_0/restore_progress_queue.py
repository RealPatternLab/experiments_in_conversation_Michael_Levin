#!/usr/bin/env python3
"""
Restore Progress Queue

This script analyzes the current state of the video pipeline and populates
the progress queue with the actual processing status of existing videos.
"""

import json
import logging
from pathlib import Path
from pipeline_progress_queue import VideoPipelineProgressQueue, get_progress_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_existing_processing():
    """Analyze the current state of video processing and update progress queue."""
    queue = get_progress_queue()
    
    # Get the existing video from step 2 metadata
    step_02_metadata = Path("step_02_extracted_playlist_content/download_summary.json")
    if not step_02_metadata.exists():
        logger.error("Step 2 metadata not found")
        return
    
    with open(step_02_metadata, 'r') as f:
        metadata = json.load(f)
    
    # Extract video information
    videos = metadata.get('videos', [])
    if not videos:
        logger.error("No videos found in metadata")
        return
    
    # Create a playlist entry for the current processing
    playlist_id = "formal_presentations_1_on_0"
    playlist_title = "Formal Presentations - 1 on 0"
    
    logger.info(f"Adding playlist: {playlist_id}")
    queue.add_playlist(playlist_id, playlist_title, len(videos))
    
    for video_info in videos:
        video_id = video_info['video_id']
        video_title = video_info['title']
        upload_date = video_info.get('upload_date')
        
        logger.info(f"Processing video: {video_id}")
        
        # Add video to progress queue
        queue.add_video(video_id, video_title, playlist_id)
        
        # Update upload date if available
        if upload_date:
            queue.update_video_upload_date(video_id, upload_date, "metadata_discovered")
        
        # Analyze each step to determine current status
        analyze_video_step_status(queue, video_id, video_info)
    
    logger.info("Progress queue restoration completed")

def analyze_video_step_status(queue, video_id: str, video_info: dict):
    """Analyze the current status of each step for a video."""
    
    # Step 1: Playlist Processing (always completed if we're here)
    queue.update_video_step_status(video_id, "step_01_playlist_processing", "completed")
    
    # Step 2: Video Download (completed if we have metadata)
    queue.update_video_step_status(
        video_id, 
        "step_02_video_download", 
        "completed",
        metadata={
            "duration": video_info.get('duration'),
            "file_size_mb": video_info.get('file_size_mb'),
            "upload_date": video_info.get('upload_date')
        }
    )
    
    # Step 3: Transcription
    transcript_file = Path("step_03_transcription") / f"{video_id}_transcript.json"
    if transcript_file.exists():
        queue.update_video_step_status(video_id, "step_03_transcription", "completed")
        logger.info(f"  Step 3: Completed (transcript exists)")
    else:
        queue.update_video_step_status(video_id, "step_03_transcription", "pending")
        logger.info(f"  Step 3: Pending (no transcript)")
    
    # Step 4: Semantic Chunking
    chunks_file = Path("step_04_extract_chunks") / f"{video_id}_chunks.json"
    if chunks_file.exists():
        # Count chunks
        try:
            with open(chunks_file, 'r') as f:
                chunks_data = json.load(f)
                chunk_count = len(chunks_data.get('chunks', []))
                queue.update_video_step_status(
                    video_id, 
                    "step_04_semantic_chunking", 
                    "completed",
                    metadata={"chunk_count": chunk_count}
                )
                logger.info(f"  Step 4: Completed ({chunk_count} chunks)")
        except Exception as e:
            logger.warning(f"  Step 4: Error reading chunks: {e}")
            queue.update_video_step_status(video_id, "step_04_semantic_chunking", "failed", error=str(e))
    else:
        queue.update_video_step_status(video_id, "step_04_semantic_chunking", "pending")
        logger.info(f"  Step 4: Pending (no chunks)")
    
    # Step 5: Frame Extraction
    frames_summary = Path("step_05_frames") / f"{video_id}_frames_summary.json"
    if frames_summary.exists():
        try:
            with open(frames_summary, 'r') as f:
                frames_data = json.load(f)
                frame_count = len(frames_data.get('frames', []))
                queue.update_video_step_status(
                    video_id, 
                    "step_05_frame_extraction", 
                    "completed",
                    metadata={"frame_count": frame_count}
                )
                logger.info(f"  Step 5: Completed ({frame_count} frames)")
        except Exception as e:
            logger.warning(f"  Step 5: Error reading frames: {e}")
            queue.update_video_step_status(video_id, "step_05_frame_extraction", "failed", error=str(e))
    else:
        queue.update_video_step_status(video_id, "step_05_frame_extraction", "pending")
        logger.info(f"  Step 5: Pending (no frames)")
    
    # Step 6: Frame-Chunk Alignment
    alignments_file = Path("step_06_frame_chunk_alignment") / f"{video_id}_alignments.json"
    if alignments_file.exists():
        queue.update_video_step_status(video_id, "step_06_frame_chunk_alignment", "completed")
        logger.info(f"  Step 6: Completed (alignments exist)")
    else:
        queue.update_video_step_status(video_id, "step_06_frame_chunk_alignment", "pending")
        logger.info(f"  Step 6: Pending (no alignments)")
    
    # Step 7: Consolidated Embedding
    # Check if any embedding files exist
    embeddings_dir = Path("step_07_faiss_embeddings")
    if embeddings_dir.exists():
        embedding_files = list(embeddings_dir.glob("*.faiss"))
        if embedding_files:
            queue.update_video_step_status(video_id, "step_07_consolidated_embedding", "completed")
            logger.info(f"  Step 7: Completed (embedding files exist)")
        else:
            queue.update_video_step_status(video_id, "step_07_consolidated_embedding", "pending")
            logger.info(f"  Step 7: Pending (no embedding files)")
    else:
        queue.update_video_step_status(video_id, "step_07_consolidated_embedding", "pending")
        logger.info(f"  Step 7: Pending (no embeddings directory)")

def display_current_status():
    """Display the current pipeline status."""
    queue = get_progress_queue()
    summary = queue.get_pipeline_summary()
    
    print("\n" + "="*60)
    print("VIDEO PIPELINE CURRENT STATUS")
    print("="*60)
    
    print(f"\n📊 Overall Pipeline Status:")
    print(f"  Total Videos: {summary['total_videos']}")
    print(f"  Total Playlists: {summary['total_playlists']}")
    print(f"  Last Updated: {summary['last_updated']}")
    
    print(f"\n🎬 Video Status Breakdown:")
    video_statuses = summary['video_statuses']
    for status, count in video_statuses.items():
        if count > 0:
            print(f"  {status.title()}: {count}")
    
    print(f"\n📚 Playlist Status Breakdown:")
    playlist_statuses = summary['playlist_statuses']
    for status, count in playlist_statuses.items():
        if count > 0:
            print(f"  {status.title()}: {count}")
    
    print(f"\n🔧 Step-by-Step Progress:")
    step_progress = summary['step_progress']
    for step, counts in step_progress.items():
        step_name = step.replace('_', ' ').title()
        completed = counts['completed']
        processing = counts['processing']
        failed = counts['failed']
        pending = counts['pending']
        skipped = counts['skipped']
        
        if completed > 0 or processing > 0 or failed > 0:
            print(f"  {step_name}:")
            if completed > 0:
                print(f"    ✅ Completed: {completed}")
            if processing > 0:
                print(f"    🔄 Processing: {processing}")
            if failed > 0:
                print(f"    ❌ Failed: {failed}")
            if skipped > 0:
                print(f"    ⏭️  Skipped: {skipped}")
            if pending > 0:
                print(f"    ⏳ Pending: {pending}")
    
    # Show detailed video status
    print(f"\n📹 Individual Video Status:")
    videos = queue._load_queue_data()["pipeline_progress"]
    for video_id, video_data in videos.items():
        print(f"\n  {video_id}:")
        print(f"    Title: {video_data['video_title'][:60]}...")
        print(f"    Current Status: {video_data['current_status']}")
        print(f"    Current Step: {video_data['current_step']}")
        
        # Show step completion
        steps = [
            "step_01_playlist_processing",
            "step_02_video_download",
            "step_03_transcription",
            "step_04_semantic_chunking",
            "step_05_frame_extraction",
            "step_06_frame_chunk_alignment",
            "step_07_consolidated_embedding"
        ]
        
        for step in steps:
            status = video_data.get(step, 'unknown')
            if status == 'completed':
                print(f"      ✅ {step.replace('_', ' ').title()}")
            elif status == 'processing':
                print(f"      🔄 {step.replace('_', ' ').title()}")
            elif status == 'failed':
                print(f"      ❌ {step.replace('_', ' ').title()}")
            elif status == 'skipped':
                print(f"      ⏭️  {step.replace('_', ' ').title()}")
            else:
                print(f"      ⏳ {step.replace('_', ' ').title()}")

if __name__ == "__main__":
    logger.info("Starting progress queue restoration...")
    
    try:
        # Analyze existing processing and update queue
        analyze_existing_processing()
        
        # Display current status
        display_current_status()
        
        logger.info("Progress queue restoration completed successfully!")
        
    except Exception as e:
        logger.error(f"Progress queue restoration failed: {e}")
        raise
