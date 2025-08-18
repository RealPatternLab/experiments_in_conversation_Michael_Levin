#!/usr/bin/env python3
"""
Frame Recovery Script
This script helps recover frames that were accidentally deleted by the cleanup process.
It will re-run the frame extraction step to regenerate the missing frames.
"""

import os
import sys
import logging
from pathlib import Path
import subprocess

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('recover_frames')

def check_frame_status():
    """Check the current status of frames in the pipeline"""
    logger.info("🔍 Checking frame status...")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "step_05_frame_extractor.py").exists():
        logger.error("❌ Please run this script from the pipeline directory")
        logger.error(f"Current directory: {current_dir}")
        logger.error("Expected: SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/")
        return False
    
    # Check frame directories
    frames_dir = current_dir / "step_05_frames"
    if not frames_dir.exists():
        logger.error("❌ Frames directory not found!")
        return False
    
    # Check what's in the frames directory
    frame_contents = list(frames_dir.iterdir())
    logger.info(f"📁 Frames directory contents: {len(frame_contents)} items")
    
    for item in frame_contents:
        if item.is_dir():
            frame_files = list(item.glob("*.jpg"))
            logger.info(f"  📹 {item.name}: {len(frame_files)} frame files")
        else:
            logger.info(f"  📄 {item.name}")
    
    # Check alignment files
    alignment_dir = current_dir / "step_06_frame_chunk_alignment"
    if alignment_dir.exists():
        alignment_files = list(alignment_dir.glob("*_alignments.json"))
        logger.info(f"🔗 Found {len(alignment_files)} alignment files")
        
        for align_file in alignment_files:
            logger.info(f"  📋 {align_file.name}")
    else:
        logger.warning("⚠️ No alignment directory found")
    
    return True

def recover_frames():
    """Recover frames by re-running the frame extraction step"""
    logger.info("🔄 Starting frame recovery...")
    
    # Check if step 5 frame extractor exists
    frame_extractor = Path("step_05_frame_extractor.py")
    if not frame_extractor.exists():
        logger.error("❌ Frame extractor script not found!")
        return False
    
    # Check if we have the required input (videos and transcripts)
    video_dir = Path("step_02_extracted_playlist_content")
    transcript_dir = Path("step_03_transcription")
    
    if not video_dir.exists():
        logger.error("❌ Video directory not found! Cannot recover frames without videos.")
        return False
    
    if not transcript_dir.exists():
        logger.error("❌ Transcript directory not found! Cannot recover frames without transcripts.")
        return False
    
    logger.info("✅ Prerequisites found. Ready to recover frames.")
    
    # Ask for confirmation
    response = input("Do you want to proceed with frame recovery? This will re-run step 5. (y/N): ")
    if response.lower() != 'y':
        logger.info("Frame recovery cancelled.")
        return False
    
    try:
        # Run the frame extraction step
        logger.info("🚀 Running frame extraction step...")
        result = subprocess.run(
            ["uv", "run", "python", "step_05_frame_extractor.py"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            logger.info("✅ Frame extraction completed successfully!")
            logger.info("📊 Output:")
            print(result.stdout)
        else:
            logger.error("❌ Frame extraction failed!")
            logger.error("Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run frame extraction: {e}")
        return False
    
    return True

def main():
    """Main recovery function"""
    logger.info("🆘 Frame Recovery Script")
    logger.info("=" * 50)
    
    # Check current status
    if not check_frame_status():
        logger.error("❌ Status check failed. Cannot proceed.")
        return
    
    # Offer recovery options
    print("\n" + "=" * 50)
    print("🆘 FRAME RECOVERY OPTIONS")
    print("=" * 50)
    print("1. Check frame status (already done)")
    print("2. Recover frames by re-running step 5")
    print("3. Exit")
    
    choice = input("\nChoose an option (1-3): ").strip()
    
    if choice == "1":
        logger.info("✅ Frame status already checked above.")
    elif choice == "2":
        if recover_frames():
            logger.info("🎉 Frame recovery completed!")
            logger.info("🔍 Checking updated frame status...")
            check_frame_status()
        else:
            logger.error("❌ Frame recovery failed!")
    elif choice == "3":
        logger.info("👋 Exiting frame recovery.")
    else:
        logger.error("❌ Invalid choice.")

if __name__ == "__main__":
    main()
