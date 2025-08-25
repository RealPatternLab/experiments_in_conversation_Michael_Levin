#!/usr/bin/env python3
"""
Unified Pipeline Runner for Conversations 1-on-3
Demonstrates how to use the unified framework with a custom pipeline configuration.
"""

import sys
import os
from pathlib import Path

# Add the core directory to the path so we can import unified scripts
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

# Load configuration with correct path
from config_loader import load_pipeline_config
from video_pipeline_scripts.step_01_playlist_processor import UnifiedPlaylistProcessor
from video_pipeline_scripts.step_02_video_downloader import UnifiedVideoDownloader
from video_pipeline_scripts.step_03_transcription import UnifiedTranscriptionHandler
from video_pipeline_scripts.step_04_extract_chunks import UnifiedChunkExtractor
from video_pipeline_scripts.step_05_frame_extractor import UnifiedFrameExtractor
from video_pipeline_scripts.step_06_frame_chunk_alignment import UnifiedFrameChunkAligner
from video_pipeline_scripts.step_07_faiss_embeddings import UnifiedFAISSEmbeddingGenerator
from video_pipeline_scripts.step_08_cleanup import UnifiedPipelineCleanup

def run_pipeline_step(step_name: str, step_class, config, progress_queue=None):
    """Run a single pipeline step with error handling"""
    print(f"\n🚀 Running {step_name}...")
    try:
        # Initialize the step
        step_instance = step_class(config, progress_queue)
        
        # Run the step
        if hasattr(step_instance, 'process_all_videos'):
            step_instance.process_all_videos()
        elif hasattr(step_instance, 'process_playlists'):
            step_instance.process_playlists()
        else:
            print(f"⚠️  {step_name} doesn't have a standard process method")
            return False
        
        print(f"✅ {step_name} completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ {step_name} failed: {e}")
        return False

def main():
    """Main pipeline execution function"""
    print("🎯 Unified Pipeline Runner for Conversations 1-on-3")
    print("=" * 60)
    
    # Load the pipeline configuration
    try:
        config = load_pipeline_config('conversations_1_on_3', '../core/pipeline_configs')
        print(f"✅ Loaded configuration for: {config['pipeline_name']}")
        print(f"   Speaker count: {config['speaker_count']}")
        print(f"   LLM enhancement: {config['llm_enhancement']}")
        print(f"   Frame extraction: {config['frame_extraction']['enabled']}")
        print(f"   Frame interval: {config['frame_extraction']['interval_seconds']}s")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Pipeline steps in order
    pipeline_steps = [
        ("Step 1: Playlist Processing", UnifiedPlaylistProcessor),
        ("Step 2: Video Downloading", UnifiedVideoDownloader),
        ("Step 3: Transcription", UnifiedTranscriptionHandler),
        ("Step 4: Extract Chunks", UnifiedChunkExtractor),
        ("Step 5: Frame Extraction", UnifiedFrameExtractor),
        ("Step 6: Frame-Chunk Alignment", UnifiedFrameChunkAligner),
        ("Step 7: FAISS Embeddings", UnifiedFAISSEmbeddingGenerator),
        ("Step 8: Cleanup", UnifiedPipelineCleanup),
    ]
    
    # Track success/failure
    successful_steps = 0
    total_steps = len(pipeline_steps)
    
    print(f"\n📋 Pipeline Overview:")
    print(f"   Total steps: {total_steps}")
    print(f"   Working directory: {Path.cwd()}")
    print(f"   Configuration: {config['pipeline_type']}")
    
    # Run each step
    for step_name, step_class in pipeline_steps:
        success = run_pipeline_step(step_name, step_class, config)
        if success:
            successful_steps += 1
    
    # Final summary
    print(f"\n🎉 Pipeline Execution Summary:")
    print(f"   Successful steps: {successful_steps}/{total_steps}")
    print(f"   Success rate: {(successful_steps/total_steps)*100:.1f}%")
    
    if successful_steps == total_steps:
        print("   🎯 All steps completed successfully!")
        return 0
    else:
        print(f"   ⚠️  {total_steps - successful_steps} steps failed")
        return 1

if __name__ == "__main__":
    exit(main())
