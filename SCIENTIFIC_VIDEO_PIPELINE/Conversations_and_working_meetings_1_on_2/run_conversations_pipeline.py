#!/usr/bin/env python3
"""
Conversations Video Pipeline Runner (1-on-2)
Runs all 7 steps of the 1-on-2 conversations video processing pipeline in sequence.
Specialized for handling conversations between Michael Levin and 2 other researchers.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import centralized logging configuration
from logging_config import setup_logging, get_logs_dir

# Configure logging
logger = setup_logging('pipeline_execution_1_on_2')

class ConversationsVideoPipeline1On2:
    """Consolidated pipeline runner for all 7 steps of 1-on-2 conversations processing"""
    
    def __init__(self):
        self.steps = [
            {
                'name': 'Step 1: Playlist Processing',
                'script': 'step_01_playlist_processor.py',
                'description': 'Process YouTube playlist URLs and extract metadata for 1-on-2 conversations'
            },
            {
                'name': 'Step 2: Video Download',
                'script': 'step_02_video_downloader.py',
                'description': 'Download videos using yt-dlp with metadata extraction'
            },
            {
                'name': 'Step 3: Transcription with Speaker Diarization',
                'script': 'step_03_transcription_webhook.py',
                'description': 'Create high-quality transcripts with 3-speaker identification using AssemblyAI'
            },
            {
                'name': 'Step 4: Multi-Speaker Semantic Chunking',
                'script': 'step_04_extract_chunks.py',
                'description': 'Create semantic chunks focusing on 3-speaker dynamics and Levin\'s views'
            },
            {
                'name': 'Step 5: Frame Extraction',
                'script': 'step_05_frame_extractor.py',
                'description': 'Extract video frames at regular intervals for visual context'
            },
            {
                'name': 'Step 6: Frame-Chunk Alignment',
                'script': 'step_06_frame_chunk_alignment.py',
                'description': 'Align extracted frames with transcript chunks for visual citations'
            },
            {
                'name': 'Step 7: Consolidated Embedding',
                'script': 'step_07_consolidated_embedding.py',
                'description': 'Create embeddings and build FAISS indices for 1-on-2 conversation search'
            }
        ]
        
        self.execution_results = []
        self.start_time = None
        self.end_time = None
    
    def check_step_dependencies(self, step_number: int) -> bool:
        """Check if a step's dependencies are met before execution"""
        logger.info(f"🔍 Checking dependencies for step {step_number}")
        
        try:
            if step_number == 1:  # Playlist Processing
                # No dependencies for step 1
                return True
                
            elif step_number == 2:  # Video Download
                # Depends on step 1: playlist metadata
                metadata_file = Path("step_01_raw/playlist_and_video_metadata.json")
                summary_file = Path("step_01_raw/playlist_summary.json")
                
                if not (metadata_file.exists() and summary_file.exists()):
                    logger.warning(f"   Step {step_number} dependencies not met: Missing playlist files")
                    return False
                
                # Check if metadata contains actual video information
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    if not metadata or 'playlists' not in metadata or not metadata['playlists']:
                        logger.warning(f"   Step {step_number} dependencies not met: Empty metadata")
                        return False
                    
                    if not summary or 'total_videos' not in summary or summary['total_videos'] == 0:
                        logger.warning(f"   Step {step_number} dependencies not met: No videos in summary")
                        return False
                        
                    logger.info(f"   Step {step_number} dependencies met: Found {summary['total_videos']} videos")
                    return True
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"   Step {step_number} dependencies not met: JSON error - {e}")
                    return False
                    
            elif step_number == 3:  # Transcription
                # Depends on step 2: downloaded videos
                video_dir = Path("step_02_extracted_playlist_content")
                if not video_dir.exists():
                    logger.warning(f"   Step {step_number} dependencies not met: Video directory doesn't exist")
                    return False
                
                # Look for actual video files
                video_files = []
                for video_subdir in video_dir.glob("video_*"):
                    if video_subdir.is_dir():
                        for video_file in video_subdir.glob("video.*"):
                            if video_file.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
                                if video_file.stat().st_size > 1000:  # At least 1KB
                                    video_files.append(video_file)
                
                if len(video_files) == 0:
                    logger.warning(f"   Step {step_number} dependencies not met: No video files found")
                    return False
                
                logger.info(f"   Step {step_number} dependencies met: Found {len(video_files)} video files")
                return True
                
            elif step_number == 4:  # Semantic Chunking
                # Depends on step 3: transcripts
                transcript_dir = Path("step_03_transcription")
                if not transcript_dir.exists():
                    logger.warning(f"   Step {step_number} dependencies not met: Transcript directory doesn't exist")
                    return False
                
                transcript_files = list(transcript_dir.glob("*_transcript.json"))
                if not transcript_files:
                    logger.warning(f"   Step {step_number} dependencies not met: No transcript files found")
                    return False
                
                # Check if at least one transcript has content
                for transcript_file in transcript_files:
                    try:
                        import json
                        with open(transcript_file, 'r') as f:
                            transcript = json.load(f)
                        
                        if transcript and 'transcript' in transcript and transcript['transcript'].strip():
                            logger.info(f"   Step {step_number} dependencies met: Found transcript with content")
                            return True
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                logger.warning(f"   Step {step_number} dependencies not met: No transcripts with content found")
                return False
                
            elif step_number == 5:  # Frame Extraction
                # Depends on step 2: downloaded videos
                video_dir = Path("step_02_extracted_playlist_content")
                if not video_dir.exists():
                    logger.warning(f"   Step {step_number} dependencies not met: Video directory doesn't exist")
                    return False
                
                # Look for actual video files
                video_files = []
                for video_subdir in video_dir.glob("video_*"):
                    if video_subdir.is_dir():
                        for video_file in video_subdir.glob("video.*"):
                            if video_file.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
                                if video_file.stat().st_size > 1000:  # At least 1KB
                                    video_files.append(video_file)
                
                if len(video_files) == 0:
                    logger.warning(f"   Step {step_number} dependencies not met: No video files found")
                    return False
                
                logger.info(f"   Step {step_number} dependencies met: Found {len(video_files)} video files")
                return True
                
            elif step_number == 6:  # Frame-Chunk Alignment
                # Depends on steps 4 and 5: chunks and frames
                chunks_dir = Path("step_04_extract_chunks")
                frames_dir = Path("step_05_frames")
                
                if not (chunks_dir.exists() and frames_dir.exists()):
                    logger.warning(f"   Step {step_number} dependencies not met: Missing chunks or frames directories")
                    return False
                
                chunk_files = list(chunks_dir.glob("*_chunks.json"))
                frame_summary_files = list(frames_dir.glob("*_frames_summary.json"))
                
                if not chunk_files or not frame_summary_files:
                    logger.warning(f"   Step {step_number} dependencies not met: Missing chunk or frame files")
                    return False
                
                logger.info(f"   Step {step_number} dependencies met: Found chunks and frames")
                return True
                
            elif step_number == 7:  # FAISS Embeddings
                # Depends on step 6: alignments
                alignment_dir = Path("step_06_frame_chunk_alignment")
                if not alignment_dir.exists():
                    logger.warning(f"   Step {step_number} dependencies not met: Alignment directory doesn't exist")
                    return False
                
                alignment_files = list(alignment_dir.glob("*_alignment_summary.json"))
                if not alignment_files:
                    logger.warning(f"   Step {step_number} dependencies not met: No alignment files found")
                    return False
                
                # Check if at least one alignment has content
                for alignment_file in alignment_files:
                    try:
                        import json
                        with open(alignment_file, 'r') as f:
                            alignment = json.load(f)
                        
                        if alignment and 'total_alignments' in alignment and alignment['total_alignments'] > 0:
                            logger.info(f"   Step {step_number} dependencies met: Found alignments with content")
                            return True
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                logger.warning(f"   Step {step_number} dependencies not met: No alignments with content found")
                return False
                
            else:
                return True  # Unknown step, assume dependencies met
                
        except Exception as e:
            logger.warning(f"Error checking dependencies for step {step_number}: {e}")
            return False
    
    def validate_step_output(self, step_number: int, step_name: str) -> bool:
        """Validate that a step produced the expected outputs with actual content"""
        logger.info(f"🔍 Validating step {step_number} output: {step_name}")
        
        try:
            if step_number == 1:  # Playlist Processing
                # Check if playlist metadata files exist and have content
                metadata_file = Path("step_01_raw/playlist_and_video_metadata.json")
                summary_file = Path("step_01_raw/playlist_summary.json")
                
                logger.info(f"   Checking playlist files: {metadata_file.exists()}, {summary_file.exists()}")
                
                if not (metadata_file.exists() and summary_file.exists()):
                    logger.warning(f"   Step {step_number} validation failed: Missing files")
                    return False
                
                # Check if files have actual content (not empty)
                try:
                    import json
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    # Check if metadata contains actual video information
                    if not metadata or 'playlists' not in metadata or not metadata['playlists']:
                        logger.warning(f"   Step {step_number} validation failed: Empty metadata")
                        return False
                    
                    # Check if summary shows actual videos
                    if not summary or 'total_videos' not in summary or summary['total_videos'] == 0:
                        logger.warning(f"   Step {step_number} validation failed: No videos in summary")
                        return False
                        
                    logger.info(f"   Step {step_number} validation passed: Found {summary['total_videos']} videos")
                    return True
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"   Step {step_number} validation failed: JSON error - {e}")
                    return False
                    
            elif step_number == 2:  # Video Download
                # Check if at least one video file was downloaded
                video_dir = Path("step_02_extracted_playlist_content")
                if not video_dir.exists():
                    logger.warning(f"   Step {step_number} validation failed: Video directory doesn't exist")
                    return False
                
                # Look for actual video files (not just metadata)
                video_files = []
                for video_subdir in video_dir.glob("video_*"):
                    if video_subdir.is_dir():
                        # Look for actual video files
                        for video_file in video_subdir.glob("video.*"):
                            if video_file.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov', '.webm']:
                                # Check if file has actual content (not 0 bytes)
                                if video_file.stat().st_size > 1000:  # At least 1KB
                                    video_files.append(video_file)
                
                logger.info(f"   Step {step_number} validation: Found {len(video_files)} video files")
                if len(video_files) == 0:
                    logger.warning(f"   Step {step_number} validation failed: No video files found")
                    return False
                
                return True
                
            elif step_number == 3:  # Transcription
                # Check if transcript files exist and have content
                transcript_dir = Path("step_03_transcription")
                if not transcript_dir.exists():
                    logger.warning(f"   Step {step_number} validation failed: Transcript directory doesn't exist")
                    return False
                
                transcript_files = list(transcript_dir.glob("*_transcript.json"))
                if not transcript_files:
                    logger.warning(f"   Step {step_number} validation failed: No transcript files found")
                    return False
                
                # Check if at least one transcript has actual content
                for transcript_file in transcript_files:
                    try:
                        import json
                        with open(transcript_file, 'r') as f:
                            transcript = json.load(f)
                        
                        # Check if transcript has actual content
                        if transcript and 'text' in transcript and transcript['text'].strip():
                            logger.info(f"   Step {step_number} validation passed: Found transcript with content")
                            return True
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                logger.warning(f"   Step {step_number} validation failed: No transcripts with content found")
                return False
                
            elif step_number == 4:  # Semantic Chunking
                # Check if chunk files exist and have content
                chunks_dir = Path("step_04_extract_chunks")
                if not chunks_dir.exists():
                    logger.warning(f"   Step {step_number} validation failed: Chunks directory doesn't exist")
                    return False
                
                chunk_files = list(chunks_dir.glob("*_chunks.json"))
                if not chunk_files:
                    logger.warning(f"   Step {step_number} validation failed: No chunk files found")
                    return False
                
                # Check if at least one chunk file has actual content
                for chunk_file in chunk_files:
                    try:
                        import json
                        with open(chunk_file, 'r') as f:
                            chunks = json.load(f)
                        
                        # Check if chunks have actual content
                        if chunks and len(chunks) > 0:
                            logger.info(f"   Step {step_number} validation passed: Found chunks with content")
                            return True
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                logger.warning(f"   Step {step_number} validation failed: No chunks with content found")
                return False
                
            elif step_number == 5:  # Frame Extraction
                # Check if frame files exist and have content
                frames_dir = Path("step_05_frames")
                if not frames_dir.exists():
                    logger.warning(f"   Step {step_number} validation failed: Frames directory doesn't exist")
                    return False
                
                frame_summary_files = list(frames_dir.glob("*_frames_summary.json"))
                if not frame_summary_files:
                    logger.warning(f"   Step {step_number} validation failed: No frame summary files found")
                    return False
                
                # Check if at least one frame summary shows actual frames
                for summary_file in frame_summary_files:
                    try:
                        import json
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                        
                        # Check if summary shows actual frames were extracted
                        if summary and 'total_frames_extracted' in summary and summary['total_frames_extracted'] > 0:
                            logger.info(f"   Step {step_number} validation passed: Found frames with content")
                            return True
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                logger.warning(f"   Step {step_number} validation failed: No frames with content found")
                return False
                
            elif step_number == 6:  # Frame-Chunk Alignment
                # Check if alignment files exist and have content
                alignment_dir = Path("step_06_frame_chunk_alignment")
                if not alignment_dir.exists():
                    logger.warning(f"   Step {step_number} validation failed: Alignment directory doesn't exist")
                    return False
                
                alignment_files = list(alignment_dir.glob("*_alignment_summary.json"))
                if not alignment_files:
                    logger.warning(f"   Step {step_number} validation failed: No alignment files found")
                    return False
                
                # Check if at least one alignment file has actual content
                for alignment_file in alignment_files:
                    try:
                        import json
                        with open(alignment_file, 'r') as f:
                            alignment = json.load(f)
                        
                        # Check if alignment has actual content
                        if alignment and 'total_alignments' in alignment and alignment['total_alignments'] > 0:
                            logger.info(f"   Step {step_number} validation passed: Found alignments with content")
                            return True
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                logger.warning(f"   Step {step_number} validation failed: No alignments with content found")
                return False
                
            elif step_number == 7:  # FAISS Embeddings
                # Check if embedding files exist and have content
                embeddings_dir = Path("step_07_faiss_embeddings")
                if not embeddings_dir.exists():
                    logger.warning(f"   Step {step_number} validation failed: Embeddings directory doesn't exist")
                    return False
                
                # Look for any embedding run directories with actual files
                run_dirs = list(embeddings_dir.glob("run_*"))
                if not run_dirs:
                    logger.warning(f"   Step {step_number} validation failed: No embedding run directories found")
                    return False
                
                # Check if at least one run directory has the required files
                for run_dir in run_dirs:
                    if run_dir.is_dir():
                        required_files = ['chunks_embeddings.npy', 'chunks_metadata.pkl', 'chunks.index']
                        if all((run_dir / file).exists() for file in required_files):
                            # Check if files have actual content
                            embeddings_file = run_dir / 'chunks_embeddings.npy'
                            if embeddings_file.stat().st_size > 1000:  # At least 1KB
                                logger.info(f"   Step {step_number} validation passed: Found embeddings with content")
                                return True
                
                logger.warning(f"   Step {step_number} validation failed: No embeddings with content found")
                return False
                
            else:
                return True  # Unknown step, assume success
                
        except Exception as e:
            logger.warning(f"Error validating step {step_number} output: {e}")
            return False
    
    def run_step(self, step_info: Dict[str, str]) -> Dict[str, Any]:
        """Run a single pipeline step"""
        step_name = step_info['name']
        script_name = step_info['script']
        description = step_info['description']
        
        # Extract step number from step name
        step_number = int(step_name.split(':')[0].split()[-1])
        
        logger.info(f"🚀 Starting {step_name}")
        logger.info(f"   Description: {description}")
        logger.info(f"   Script: {script_name}")
        
        step_start = time.time()
        result = {
            'step_name': step_name,
            'script_name': script_name,
            'description': description,
            'step_number': step_number,
            'start_time': datetime.now().isoformat(),
            'status': 'unknown',
            'error': None,
            'duration_seconds': 0
        }
        
        try:
            # Check dependencies first
            if not self.check_step_dependencies(step_number):
                result['status'] = 'pending'
                result['error'] = 'Dependencies not met'
                logger.warning(f"⏳ {step_name} marked as pending: Dependencies not met")
                return result
            
            # Check if script exists
            if not Path(script_name).exists():
                raise FileNotFoundError(f"Script {script_name} not found")
            
            # Run the script using uv
            import subprocess
            cmd = ['uv', 'run', 'python', script_name]
            
            logger.info(f"   Running command: {' '.join(cmd)}")
            
            # Run the script and capture output
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            step_duration = time.time() - step_start
            result['duration_seconds'] = step_duration
            
            if process.returncode == 0:
                # Script ran successfully, now validate the output
                if self.validate_step_output(step_number, step_name):
                    result['status'] = 'success'
                    logger.info(f"✅ {step_name} completed successfully in {step_duration:.2f}s")
                    
                    # Log any output
                    if process.stdout.strip():
                        logger.info(f"   Output: {process.stdout.strip()}")
                else:
                    result['status'] = 'failed'
                    result['error'] = 'Output validation failed'
                    logger.error(f"❌ {step_name} failed: Output validation failed")
            else:
                result['status'] = 'failed'
                result['error'] = process.stderr.strip() or f"Script failed with return code {process.returncode}"
                logger.error(f"❌ {step_name} failed: {result['error']}")
                
                # Log error output
                if process.stderr.strip():
                    logger.error(f"   Error: {process.stderr.strip()}")
                
        except Exception as e:
            step_duration = time.time() - step_start
            result['duration_seconds'] = step_duration
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"❌ {step_name} failed with exception: {e}")
        
        return result
    
    def run_pipeline(self, start_from_step: int = 1) -> bool:
        """Run the complete pipeline starting from the specified step"""
        logger.info("🎬 Starting 1-on-2 Conversations Video Pipeline")
        logger.info("=" * 60)
        
        self.start_time = datetime.now()
        logger.info(f"Pipeline started at: {self.start_time.isoformat()}")
        
        # Filter steps based on start_from_step
        steps_to_run = self.steps[start_from_step - 1:]
        
        if start_from_step > 1:
            logger.info(f"Starting from step {start_from_step}: {steps_to_run[0]['name']}")
        
        # Run each step
        for i, step_info in enumerate(steps_to_run, start=start_from_step):
            logger.info(f"\n📋 Step {i} of {len(self.steps)}")
            logger.info("=" * 40)
            
            result = self.run_step(step_info)
            self.execution_results.append(result)
            
            # Check if step failed
            if result['status'] == 'error':
                logger.error(f"❌ Pipeline failed at step {i}: {step_info['name']}")
                logger.error(f"Error: {result['error']}")
                return False
            
            # Add step number to result
            result['step_number'] = i
            
            # Brief pause between steps
            if i < len(self.steps):
                logger.info("   Waiting 2 seconds before next step...")
                time.sleep(2)
        
        self.end_time = datetime.now()
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        logger.info("\n🎉 Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {total_duration:.2f} seconds")
        logger.info(f"Pipeline completed at: {self.end_time.isoformat()}")
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution"""
        if not self.execution_results:
            return {"status": "No execution results available"}
        
        total_steps = len(self.execution_results)
        successful_steps = sum(1 for r in self.execution_results if r['status'] == 'success')
        failed_steps = sum(1 for r in self.execution_results if r['status'] == 'failed')
        pending_steps = sum(1 for r in self.execution_results if r['status'] == 'pending')
        
        total_duration = sum(r['duration_seconds'] for r in self.execution_results)
        
        summary = {
            "pipeline_type": "conversations_1_on_2",
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "pending_steps": pending_steps,
            "success_rate": f"{(successful_steps/total_steps)*100:.1f}%" if total_steps > 0 else "0.0%",
            "total_duration_seconds": total_duration,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "steps": self.execution_results
        }
        
        return summary
    
    def save_execution_log(self):
        """Save execution results to a log file"""
        if not self.execution_results:
            logger.warning("No execution results to save")
            return
        
        logs_dir = get_logs_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"pipeline_execution_1_on_2_{timestamp}.json"
        
        summary = self.get_summary()
        
        try:
            import json
            with open(log_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Execution log saved to: {log_file}")
        except Exception as e:
            logger.error(f"Failed to save execution log: {e}")

def main():
    """Main entry point for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the 1-on-2 Conversations Video Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_conversations_pipeline.py
  
  # Start from step 3 (transcription)
  python run_conversations_pipeline.py --start-from 3
  
  # Run with verbose logging
  python run_conversations_pipeline.py --verbose
        """
    )
    
    parser.add_argument(
        '--start-from',
        type=int,
        default=1,
        help='Start pipeline from this step number (1-7)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate start step
    if args.start_from < 1 or args.start_from > 7:
        logger.error("Start step must be between 1 and 7")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run pipeline
    pipeline = ConversationsVideoPipeline1On2()
    
    try:
        success = pipeline.run_pipeline(start_from_step=args.start_from)
        
        if success:
            # Save execution log
            pipeline.save_execution_log()
            
            # Print summary
            summary = pipeline.get_summary()
            print("\n" + "="*60)
            print("🎯 PIPELINE EXECUTION SUMMARY")
            print("="*60)
            print(f"Pipeline Type: {summary['pipeline_type']}")
            print(f"Total Steps: {summary['total_steps']}")
            print(f"Successful: {summary['successful_steps']}")
            print(f"Failed: {summary['failed_steps']}")
            print(f"Pending: {summary['pending_steps']}")
            print(f"Success Rate: {summary['success_rate']}")
            print(f"Total Duration: {summary['total_duration_seconds']:.2f} seconds")
            print("="*60)
            
            sys.exit(0)
        else:
            logger.error("Pipeline execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
