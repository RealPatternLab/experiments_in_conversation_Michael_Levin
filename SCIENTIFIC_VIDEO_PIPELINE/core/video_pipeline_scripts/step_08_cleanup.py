#!/usr/bin/env python3
"""
Step 08: Pipeline Cleanup (Unified)
Removes unnecessary files to free up disk space while preserving essential data.
Supports both formal presentations and multi-speaker conversations.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from datetime import datetime
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UnifiedPipelineCleanup:
    """Unified pipeline cleanup supporting multiple pipeline types"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the unified pipeline cleanup
        
        Args:
            pipeline_config: Pipeline-specific configuration
            progress_queue: Progress tracking queue
        """
        self.config = pipeline_config
        self.progress_queue = progress_queue
        
        # Pipeline metadata
        self.pipeline_type = pipeline_config.get('pipeline_type', 'unknown')
        self.cleanup_config = pipeline_config.get('cleanup_config', {})
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Determine the correct base directory
        current_dir = Path.cwd()
        
        # Check if we're already in the pipeline directory
        if (current_dir / "step_01_playlist_processor.py").exists():
            self.base_dir = current_dir
        elif (current_dir / "SCIENTIFIC_VIDEO_PIPELINE" / self.pipeline_type / "step_01_playlist_processor.py").exists():
            self.base_dir = current_dir / "SCIENTIFIC_VIDEO_PIPELINE" / self.pipeline_type
        else:
            # Fallback: assume we're in the pipeline directory
            self.base_dir = current_dir
            self.logger.warning(f"⚠️ Could not determine pipeline directory. Using current directory: {current_dir}")
        
        self.logger.info(f"🔍 Using base directory: {self.base_dir}")
        
        # Verify we can find key pipeline files
        if not (self.base_dir / "step_01_playlist_processor.py").exists():
            self.logger.error(f"❌ Pipeline directory not found at {self.base_dir}")
            self.logger.error("Please run this script from the pipeline directory or the project root")
            raise FileNotFoundError(f"Pipeline directory not found at {self.base_dir}")
        
        # Track cleanup statistics
        self.cleanup_stats = {
            'videos_removed': 0,
            'frames_removed': 0,
            'unreferenced_frames_removed': 0,
            'temp_scripts_removed': 0,
            'old_logs_removed': 0,
            'old_reports_removed': 0,
            'cache_cleaned': 0,
            'yt_dlp_files_removed': 0,
            'total_space_freed_mb': 0
        }
        
        # Files to preserve (essential for Streamlit and pipeline)
        self.essential_files = self._get_essential_files()
        
        # Essential step scripts
        self.essential_scripts = {
            'step_01_playlist_processor.py',
            'step_02_video_downloader.py',
            'step_03_transcription_webhook.py',
            'step_04_extract_chunks.py',
            'step_05_frame_extractor.py',
            'step_06_frame_chunk_alignment.py',
            'step_07_consolidated_embedding.py',
            'step_08_cleanup.py'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        try:
            # Try to import centralized logging
            from logging_config import setup_logging
            return setup_logging(f'{self.pipeline_type}_cleanup')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_cleanup')
    
    def _get_essential_files(self) -> Set[str]:
        """Get essential files based on pipeline type"""
        base_essential = {
            'logs/pipeline_progress_queue.json',
            'logs/assemblyai_webhooks.json',
            '*.py',
            '*.md',
            'enhanced_chunks/*.json',
            'finetune_data/*.json'
        }
        
        # Add pipeline-specific essential files
        if self.pipeline_type.startswith('conversations'):
            base_essential.update({
                'speaker_mappings.json',
                'qa_pairs/*.json'
            })
        
        return base_essential
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def cleanup_video_files(self) -> float:
        """Remove large video files (.mp4) that are no longer needed"""
        self.logger.info("🎬 Cleaning up video files...")
        
        video_dir = self.base_dir / "step_02_extracted_playlist_content"
        if not video_dir.exists():
            return 0.0
        
        space_freed = 0.0
        
        for video_subdir in video_dir.iterdir():
            if video_subdir.is_dir():
                for video_file in video_subdir.glob("*.mp4"):
                    try:
                        file_size = self.get_file_size_mb(video_file)
                        video_file.unlink()
                        space_freed += file_size
                        self.cleanup_stats['videos_removed'] += 1
                        self.logger.info(f"  🗑️ Removed video: {video_file.name} ({file_size:.1f} MB)")
                    except Exception as e:
                        self.logger.warning(f"  ⚠️ Failed to remove {video_file.name}: {e}")
        
        self.logger.info(f"  📊 Video cleanup: {space_freed:.1f} MB freed")
        return space_freed
    
    def cleanup_unreferenced_frames(self) -> float:
        """Remove frames that are not referenced by any chunks"""
        self.logger.info("🖼️ Cleaning up unreferenced frames...")
        
        frames_dir = self.base_dir / "step_05_frames"
        if not frames_dir.exists():
            return 0.0
        
        space_freed = 0.0
        
        # Get referenced frames from alignments
        referenced_frames = self._get_referenced_frames()
        
        # Process each video's frame directory
        for video_dir in frames_dir.iterdir():
            if video_dir.is_dir() and video_dir.name != "latest":
                video_id = video_dir.name
                video_referenced = referenced_frames.get(video_id, set())
                
                for frame_file in video_dir.glob("*.jpg"):
                    if frame_file.name not in video_referenced:
                        try:
                            file_size = self.get_file_size_mb(frame_file)
                            frame_file.unlink()
                            space_freed += file_size
                            self.cleanup_stats['unreferenced_frames_removed'] += 1
                        except Exception as e:
                            self.logger.warning(f"  ⚠️ Failed to remove {frame_file.name}: {e}")
        
        self.logger.info(f"  📊 Frame cleanup: {space_freed:.1f} MB freed")
        return space_freed
    
    def _get_referenced_frames(self) -> Dict[str, Set[str]]:
        """Get frames referenced by chunks from alignment files"""
        referenced_frames = {}
        
        alignments_dir = self.base_dir / "step_06_frame_chunk_alignment"
        if not alignments_dir.exists():
            return referenced_frames
        
        # Look for alignment files
        alignment_files = list(alignments_dir.glob("*_alignments.json"))
        
        for alignment_file in alignment_files:
            try:
                with open(alignment_file, 'r') as f:
                    alignments = json.load(f)
                
                video_id = alignment_file.stem.replace('_alignments', '')
                video_frames = set()
                
                for alignment in alignments:
                    if self.pipeline_type.startswith('conversations'):
                        # Single frame strategy
                        frame = alignment.get('frame')
                        if frame and frame.get('filename'):
                            video_frames.add(frame['filename'])
                    else:
                        # Multi-frame strategy
                        frames = alignment.get('frames', [])
                        for frame in frames:
                            if frame.get('filename'):
                                video_frames.add(frame['filename'])
                
                if video_frames:
                    referenced_frames[video_id] = video_frames
                    
            except Exception as e:
                self.logger.warning(f"Failed to read alignments for {alignment_file.name}: {e}")
        
        return referenced_frames
    
    def cleanup_temp_files(self) -> float:
        """Remove temporary files and scripts"""
        self.logger.info("🧹 Cleaning up temporary files...")
        
        space_freed = 0.0
        
        # Remove temp scripts
        temp_patterns = [
            "temp_*.py",
            "test_*.py",
            "debug_*.py",
            "*.tmp",
            "*.temp"
        ]
        
        for pattern in temp_patterns:
            for temp_file in self.base_dir.glob(pattern):
                if temp_file.name not in self.essential_scripts:
                    try:
                        file_size = self.get_file_size_mb(temp_file)
                        temp_file.unlink()
                        space_freed += file_size
                        self.cleanup_stats['temp_scripts_removed'] += 1
                        self.logger.info(f"  🗑️ Removed temp file: {temp_file.name}")
                    except Exception as e:
                        self.logger.warning(f"  ⚠️ Failed to remove {temp_file.name}: {e}")
        
        self.logger.info(f"  📊 Temp file cleanup: {space_freed:.1f} MB freed")
        return space_freed
    
    def cleanup_old_logs(self) -> float:
        """Remove old log files"""
        self.logger.info("📝 Cleaning up old logs...")
        
        logs_dir = self.base_dir / "logs"
        if not logs_dir.exists():
            return 0.0
        
        space_freed = 0.0
        
        # Keep only recent logs (last 7 days)
        cutoff_time = datetime.now().timestamp() - (7 * 24 * 60 * 60)
        
        for log_file in logs_dir.glob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    file_size = self.get_file_size_mb(log_file)
                    log_file.unlink()
                    space_freed += file_size
                    self.cleanup_stats['old_logs_removed'] += 1
                    self.logger.info(f"  🗑️ Removed old log: {log_file.name}")
            except Exception as e:
                self.logger.warning(f"  ⚠️ Failed to remove {log_file.name}: {e}")
        
        self.logger.info(f"  📊 Log cleanup: {space_freed:.1f} MB freed")
        return space_freed
    
    def cleanup_cache_and_temp(self) -> float:
        """Clean up cache and temporary directories"""
        self.logger.info("🗂️ Cleaning up cache and temp directories...")
        
        space_freed = 0.0
        
        # Common cache/temp directories
        cache_dirs = [
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "node_modules",
            ".cache"
        ]
        
        for cache_dir in cache_dirs:
            cache_path = self.base_dir / cache_dir
            if cache_path.exists() and cache_path.is_dir():
                try:
                    dir_size = self._get_directory_size_mb(cache_path)
                    shutil.rmtree(cache_path)
                    space_freed += dir_size
                    self.cleanup_stats['cache_cleaned'] += 1
                    self.logger.info(f"  🗑️ Removed cache directory: {cache_dir} ({dir_size:.1f} MB)")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Failed to remove {cache_dir}: {e}")
        
        self.logger.info(f"  📊 Cache cleanup: {space_freed:.1f} MB freed")
        return space_freed
    
    def _get_directory_size_mb(self, directory: Path) -> float:
        """Get total size of directory in MB"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except:
            pass
        return total_size / (1024 * 1024)
    
    def cleanup_yt_dlp_files(self) -> float:
        """Remove yt-dlp temporary files"""
        self.logger.info("📺 Cleaning up yt-dlp files...")
        
        space_freed = 0.0
        
        # Look for yt-dlp temp files
        yt_dlp_patterns = [
            "*.part",
            "*.ytdl",
            "*.fragment",
            "*.webm.part",
            "*.mp4.part"
        ]
        
        for pattern in yt_dlp_patterns:
            for temp_file in self.base_dir.rglob(pattern):
                try:
                    file_size = self.get_file_size_mb(temp_file)
                    temp_file.unlink()
                    space_freed += file_size
                    self.cleanup_stats['yt_dlp_files_removed'] += 1
                    self.logger.info(f"  🗑️ Removed yt-dlp file: {temp_file.name}")
                except Exception as e:
                    self.logger.warning(f"  ⚠️ Failed to remove {temp_file.name}: {e}")
        
        self.logger.info(f"  📊 yt-dlp cleanup: {space_freed:.1f} MB freed")
        return space_freed
    
    def run_cleanup(self, dry_run: bool = False) -> float:
        """Run the complete cleanup process"""
        self.logger.info(f"🚀 Starting {self.pipeline_type.upper()} Pipeline Cleanup")
        self.logger.info(f"  Base directory: {self.base_dir}")
        self.logger.info(f"  Dry run: {dry_run}")
        
        if dry_run:
            self.logger.info("🔍 DRY RUN MODE - No files will be deleted")
        
        total_space_freed = 0.0
        
        # Run cleanup steps
        cleanup_steps = [
            ("Video Files", self.cleanup_video_files),
            ("Unreferenced Frames", self.cleanup_unreferenced_frames),
            ("Temporary Files", self.cleanup_temp_files),
            ("Old Logs", self.cleanup_old_logs),
            ("Cache and Temp", self.cleanup_cache_and_temp),
            ("yt-dlp Files", self.cleanup_yt_dlp_files)
        ]
        
        for step_name, cleanup_func in cleanup_steps:
            try:
                self.logger.info(f"\n--- {step_name} Cleanup ---")
                space_freed = cleanup_func()
                total_space_freed += space_freed
                
                if not dry_run:
                    self.cleanup_stats['total_space_freed_mb'] += space_freed
                    
            except Exception as e:
                self.logger.error(f"❌ {step_name} cleanup failed: {e}")
        
        # Log final summary
        self.log_cleanup_summary(dry_run)
        
        return total_space_freed
    
    def log_cleanup_summary(self, dry_run: bool = False):
        """Log a summary of the cleanup process"""
        mode = "DRY RUN" if dry_run else "ACTUAL"
        
        self.logger.info(f"\n🎉 {mode} Cleanup Summary for {self.pipeline_type.upper()}")
        self.logger.info(f"  Videos removed: {self.cleanup_stats['videos_removed']}")
        self.logger.info(f"  Frames removed: {self.cleanup_stats['frames_removed']}")
        self.logger.info(f"  Unreferenced frames: {self.cleanup_stats['unreferenced_frames_removed']}")
        self.logger.info(f"  Temp scripts: {self.cleanup_stats['temp_scripts_removed']}")
        self.logger.info(f"  Old logs: {self.cleanup_stats['old_logs_removed']}")
        self.logger.info(f"  Cache directories: {self.cleanup_stats['cache_cleaned']}")
        self.logger.info(f"  yt-dlp files: {self.cleanup_stats['yt_dlp_files_removed']}")
        
        if dry_run:
            self.logger.info(f"  📊 Estimated space to be freed: {self.cleanup_stats['total_space_freed_mb']:.1f} MB")
        else:
            self.logger.info(f"  📊 Total space freed: {self.cleanup_stats['total_space_freed_mb']:.1f} MB")
        
        self.logger.info(f"  🏠 Base directory: {self.base_dir}")
        self.logger.info(f"  ✅ Essential files preserved")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Unified Pipeline Cleanup")
    parser.add_argument("--pipeline-type", required=True, 
                       choices=["formal_presentations", "conversations_1_on_1", "conversations_1_on_2"],
                       help="Pipeline type to clean up")
    parser.add_argument("--config-dir", default="core/pipeline_configs",
                       help="Directory containing pipeline configurations")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    try:
        # Load pipeline configuration
        from core.config_loader import load_pipeline_config
        config = load_pipeline_config(args.pipeline_type, args.config_dir)
        
        # Initialize cleanup
        cleanup = UnifiedPipelineCleanup(config)
        
        # Run cleanup
        space_freed = cleanup.run_cleanup(dry_run=args.dry_run)
        
        if args.dry_run:
            print(f"\n🔍 DRY RUN COMPLETED - Estimated space to be freed: {space_freed:.1f} MB")
        else:
            print(f"\n✅ CLEANUP COMPLETED - Total space freed: {space_freed:.1f} MB")
            
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
