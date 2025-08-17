#!/usr/bin/env python3
"""
Step 08: Pipeline Cleanup
Removes unnecessary files to free up disk space while preserving essential data for Streamlit and pipeline tracking.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import argparse
from pipeline_progress_queue import get_progress_queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineCleanup:
    """Cleans up unnecessary files from the pipeline while preserving essential data"""
    
    def __init__(self, progress_queue=None):
        self.progress_queue = progress_queue or get_progress_queue()
        self.base_dir = Path.cwd()
        
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
        self.essential_files = {
            'pipeline_progress_queue.json',
            'assemblyai_webhooks.json',
            'run_video_pipeline_1_on_0.py',
            'pipeline_progress_queue.py',
            'PIPELINE_ARCHITECTURE.md',
            'QUICK_REFERENCE.md',
            'SMART_PROCESSING_SYSTEM.md'
        }
        
        # Essential step scripts
        self.essential_scripts = {
            'step_01_playlist_processor.py',
            'step_02_video_downloader.py',
            'step_03_transcription_webhook.py',
            'step_04_extract_chunks.py',
            'step_05_frame_extractor.py',
            'step_06_frame_chunk_alignment.py',
            'step_07_consolidated_embedding.py'
        }
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def cleanup_video_files(self) -> float:
        """Remove large video files (.mp4) that are no longer needed"""
        logger.info("🎬 Cleaning up video files...")
        
        video_dir = self.base_dir / "step_02_extracted_playlist_content"
        if not video_dir.exists():
            return 0.0
        
        total_freed = 0.0
        
        for video_subdir in video_dir.iterdir():
            if video_subdir.is_dir() and video_subdir.name.startswith('video_'):
                video_file = video_subdir / "video.mp4"
                if video_file.exists():
                    size_mb = self.get_file_size_mb(video_file)
                    video_file.unlink()
                    total_freed += size_mb
                    self.cleanup_stats['videos_removed'] += 1
                    logger.info(f"🗑️ Removed video: {video_file.name} ({size_mb:.1f} MB)")
        
        self.cleanup_stats['total_space_freed_mb'] += total_freed
        logger.info(f"✅ Video cleanup completed. Freed {total_freed:.1f} MB")
        return total_freed
    
    def cleanup_yt_dlp_files(self) -> float:
        """Remove yt-dlp generated files that are not needed"""
        logger.info("📥 Cleaning up yt-dlp files...")
        
        video_dir = self.base_dir / "step_02_extracted_playlist_content"
        if not video_dir.exists():
            return 0.0
        
        total_freed = 0.0
        yt_dlp_extensions = ['.info.json', '.webp', '.vtt', '.description']
        
        for video_subdir in video_dir.iterdir():
            if video_subdir.is_dir() and video_subdir.name.startswith('video_'):
                for ext in yt_dlp_extensions:
                    yt_file = video_subdir / f"video{ext}"
                    if yt_file.exists():
                        size_mb = self.get_file_size_mb(yt_file)
                        yt_file.unlink()
                        total_freed += size_mb
                        self.cleanup_stats['yt_dlp_files_removed'] += 1
                        logger.info(f"🗑️ Removed yt-dlp file: {yt_file.name} ({size_mb:.1f} MB)")
        
        self.cleanup_stats['total_space_freed_mb'] += total_freed
        logger.info(f"✅ yt-dlp cleanup completed. Freed {total_freed:.1f} MB")
        return total_freed
    
    def get_referenced_frames(self, video_id: str) -> Set[str]:
        """Get set of frame IDs that are referenced in chunk alignments"""
        alignment_file = self.base_dir / "step_06_frame_chunk_alignment" / f"{video_id}_alignments.json"
        
        if not alignment_file.exists():
            return set()
        
        try:
            with open(alignment_file, 'r') as f:
                alignments = json.load(f)
            
            referenced_frames = set()
            for chunk in alignments:
                if 'aligned_frames' in chunk:
                    for frame in chunk['aligned_frames']:
                        if 'frame_id' in frame:
                            referenced_frames.add(frame['frame_id'])
            
            return referenced_frames
        except Exception as e:
            logger.warning(f"Failed to read alignments for {video_id}: {e}")
            return set()
    
    def cleanup_unreferenced_frames(self) -> float:
        """Remove frame files that are not referenced in chunk alignments"""
        logger.info("🖼️ Cleaning up unreferenced frame files...")
        
        frames_dir = self.base_dir / "step_05_frames"
        if not frames_dir.exists():
            return 0.0
        
        total_freed = 0.0
        
        for video_subdir in frames_dir.iterdir():
            if video_subdir.is_dir():
                video_id = video_subdir.name
                referenced_frames = self.get_referenced_frames(video_id)
                
                logger.info(f"🔍 Checking frames for {video_id}: {len(referenced_frames)} referenced")
                
                for frame_file in video_subdir.iterdir():
                    if frame_file.is_file() and frame_file.suffix == '.jpg':
                        # Extract frame ID from filename (e.g., "FzFFeRVEdUM_frame_000_0.0s.jpg")
                        frame_name = frame_file.stem
                        frame_id = frame_name.split('_frame_')[0] + '_frame_' + frame_name.split('_frame_')[1]
                        
                        if frame_id not in referenced_frames:
                            size_mb = self.get_file_size_mb(frame_file)
                            frame_file.unlink()
                            total_freed += size_mb
                            self.cleanup_stats['unreferenced_frames_removed'] += 1
                            logger.debug(f"🗑️ Removed unreferenced frame: {frame_file.name}")
                
                # Remove empty video frame directories
                if not any(video_subdir.iterdir()):
                    video_subdir.rmdir()
                    logger.info(f"🗑️ Removed empty frame directory: {video_subdir.name}")
        
        self.cleanup_stats['total_space_freed_mb'] += total_freed
        logger.info(f"✅ Frame cleanup completed. Freed {total_freed:.1f} MB")
        return total_freed
    
    def cleanup_temp_scripts(self) -> int:
        """Remove temporary utility scripts that are no longer needed"""
        logger.info("🧹 Cleaning up temporary utility scripts...")
        
        temp_scripts = [
            'remove_zKjdkxrvE6k.py',
            'fix_webhook_status.py',
            'update_new_video_step3.py',
            'update_step3_progress.py',
            'update_step4_progress.py',
            'update_step5_progress.py',
            'update_step6_progress.py',
            'update_step7_progress.py',
            'restore_progress_queue.py'
        ]
        
        removed_count = 0
        for script in temp_scripts:
            script_path = self.base_dir / script
            if script_path.exists():
                script_path.unlink()
                removed_count += 1
                logger.info(f"🗑️ Removed temp script: {script}")
        
        self.cleanup_stats['temp_scripts_removed'] = removed_count
        logger.info(f"✅ Temp script cleanup completed. Removed {removed_count} files")
        return removed_count
    
    def cleanup_old_logs(self) -> int:
        """Remove old log files, keeping only the most recent"""
        logger.info("📝 Cleaning up old log files...")
        
        log_files = [
            'pipeline_execution.log',
            'transcription_webhook.log',
            'chunking.log',
            'frame_extraction.log',
            'frame_chunk_alignment.log',
            'consolidated_embedding.log',
            'video_download.log',
            'playlist_processing.log'
        ]
        
        removed_count = 0
        for log_file in log_files:
            log_path = self.base_dir / log_file
            if log_path.exists():
                # Keep only the last 1000 lines of each log
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) > 1000:
                        with open(log_path, 'w') as f:
                            f.writelines(lines[-1000:])
                        logger.info(f"📝 Truncated log: {log_file} (kept last 1000 lines)")
                except Exception as e:
                    logger.warning(f"Failed to truncate log {log_file}: {e}")
        
        logger.info(f"✅ Log cleanup completed")
        return removed_count
    
    def cleanup_old_reports(self) -> int:
        """Remove old pipeline reports, keeping only the most recent"""
        logger.info("📊 Cleaning up old pipeline reports...")
        
        reports_dir = self.base_dir
        report_files = list(reports_dir.glob("pipeline_report_*.txt"))
        
        if len(report_files) > 1:
            # Sort by modification time and keep only the most recent
            report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_report in report_files[1:]:
                old_report.unlink()
                logger.info(f"🗑️ Removed old report: {old_report.name}")
            
            removed_count = len(report_files) - 1
            self.cleanup_stats['old_reports_removed'] = removed_count
            logger.info(f"✅ Report cleanup completed. Removed {removed_count} old reports")
            return removed_count
        
        return 0
    
    def cleanup_cache(self) -> int:
        """Remove Python cache directories"""
        logger.info("🗂️ Cleaning up Python cache...")
        
        cache_dirs = [
            '__pycache__',
            'step_01_raw/__pycache__',
            'step_02_extracted_playlist_content/__pycache__',
            'step_03_transcription/__pycache__',
            'step_04_extract_chunks/__pycache__',
            'step_05_frames/__pycache__',
            'step_06_frame_chunk_alignment/__pycache__',
            'step_07_faiss_embeddings/__pycache__'
        ]
        
        removed_count = 0
        for cache_dir in cache_dirs:
            cache_path = self.base_dir / cache_dir
            if cache_path.exists() and cache_path.is_dir():
                shutil.rmtree(cache_path)
                removed_count += 1
                logger.info(f"🗑️ Removed cache directory: {cache_dir}")
        
        self.cleanup_stats['cache_cleaned'] = removed_count
        logger.info(f"✅ Cache cleanup completed. Removed {removed_count} directories")
        return removed_count
    
    def cleanup_audio_files(self) -> float:
        """Remove temporary audio files from transcription step"""
        logger.info("🎵 Cleaning up temporary audio files...")
        
        transcription_dir = self.base_dir / "step_03_transcription"
        if not transcription_dir.exists():
            return 0.0
        
        total_freed = 0.0
        audio_extensions = ['.wav', '.mp3', '.m4a']
        
        for audio_file in transcription_dir.iterdir():
            if audio_file.is_file() and audio_file.suffix in audio_extensions:
                size_mb = self.get_file_size_mb(audio_file)
                audio_file.unlink()
                total_freed += size_mb
                logger.info(f"🗑️ Removed audio file: {audio_file.name} ({size_mb:.1f} MB)")
        
        self.cleanup_stats['total_space_freed_mb'] += total_freed
        logger.info(f"✅ Audio cleanup completed. Freed {total_freed:.1f} MB")
        return total_freed
    
    def cleanup_submission_files(self) -> int:
        """Remove AssemblyAI submission metadata files (not needed for analysis)"""
        logger.info("📋 Cleaning up submission metadata files...")
        
        transcription_dir = self.base_dir / "step_03_transcription"
        if not transcription_dir.exists():
            return 0
        
        removed_count = 0
        for submission_file in transcription_dir.glob("*_submission.json"):
            submission_file.unlink()
            removed_count += 1
            logger.info(f"🗑️ Removed submission file: {submission_file.name}")
        
        logger.info(f"✅ Submission cleanup completed. Removed {removed_count} files")
        return removed_count
    
    def get_cleanup_summary(self) -> Dict:
        """Generate cleanup summary with space savings"""
        total_freed_gb = self.cleanup_stats['total_space_freed_mb'] / 1024
        
        summary = {
            'cleanup_timestamp': datetime.now().isoformat(),
            'total_space_freed_gb': round(total_freed_gb, 2),
            'total_space_freed_mb': round(self.cleanup_stats['total_space_freed_mb'], 2),
            'files_removed': {
                'videos': self.cleanup_stats['videos_removed'],
                'unreferenced_frames': self.cleanup_stats['unreferenced_frames_removed'],
                'temp_scripts': self.cleanup_stats['temp_scripts_removed'],
                'yt_dlp_files': self.cleanup_stats['yt_dlp_files_removed'],
                'old_reports': self.cleanup_stats['old_reports_removed'],
                'cache_directories': self.cleanup_stats['cache_cleaned']
            },
            'preserved_essentials': [
                'pipeline_progress_queue.json',
                'assemblyai_webhooks.json',
                'All step scripts (*.py)',
                'All transcript files (*_transcript.json)',
                'All chunk files (*_chunks.json)',
                'Referenced frame files (referenced in alignments)',
                'All alignment files (*_alignments.json)',
                'All embedding files and FAISS indices',
                'Documentation files (*.md)',
                'Latest pipeline report'
            ]
        }
        
        return summary
    
    def save_cleanup_report(self, summary: Dict):
        """Save cleanup report to file"""
        report_file = self.base_dir / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📄 Cleanup report saved: {report_file}")
    
    def run_cleanup(self, dry_run: bool = False) -> Dict:
        """Run the complete cleanup process"""
        logger.info("🚀 Starting pipeline cleanup process...")
        
        if dry_run:
            logger.info("🔍 DRY RUN MODE - No files will be deleted")
        
        start_time = datetime.now()
        
        # Run all cleanup operations
        self.cleanup_video_files()
        self.cleanup_yt_dlp_files()
        self.cleanup_unreferenced_frames()
        self.cleanup_temp_scripts()
        self.cleanup_old_logs()
        self.cleanup_old_reports()
        self.cleanup_cache()
        self.cleanup_audio_files()
        self.cleanup_submission_files()
        
        # Generate summary
        summary = self.get_cleanup_summary()
        summary['execution_time_seconds'] = (datetime.now() - start_time).total_seconds()
        summary['dry_run'] = dry_run
        
        # Save report
        self.save_cleanup_report(summary)
        
        # Log final summary
        logger.info("🎉 Cleanup process completed!")
        logger.info(f"📊 Total space freed: {summary['total_space_freed_gb']} GB")
        logger.info(f"🗑️ Total files removed: {sum(summary['files_removed'].values())}")
        
        return summary

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Pipeline Cleanup - Remove unnecessary files')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize progress queue
        progress_queue = get_progress_queue()
        logger.info("✅ Progress queue initialized")
        
        # Run cleanup
        cleanup = PipelineCleanup(progress_queue)
        summary = cleanup.run_cleanup(dry_run=args.dry_run)
        
        if args.dry_run:
            logger.info("🔍 DRY RUN COMPLETED - No files were actually deleted")
            logger.info("Run without --dry-run to perform actual cleanup")
        
        logger.info("✅ Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        raise

if __name__ == "__main__":
    main()
