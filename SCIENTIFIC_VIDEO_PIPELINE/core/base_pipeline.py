#!/usr/bin/env python3
"""
Base Pipeline Class for Scientific Video Processing
All video pipelines inherit from this base class to ensure consistency and best practices.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BasePipeline(ABC):
    """Abstract base class for all video processing pipelines"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the base pipeline
        
        Args:
            pipeline_config: Pipeline-specific configuration dictionary
            progress_queue: Progress tracking queue instance
        """
        self.pipeline_config = pipeline_config
        self.progress_queue = progress_queue
        
        # Pipeline metadata
        self.pipeline_type = pipeline_config.get('pipeline_type', 'unknown')
        self.speaker_count = pipeline_config.get('speaker_count', 1)
        self.transcription_service = pipeline_config.get('transcription_service', 'assemblyai')
        
        # Base directories
        self.base_dir = Path.cwd()
        self.output_dir = Path("step_04_extract_chunks")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client if LLM enhancement is enabled
        self.openai_client = None
        if pipeline_config.get('llm_enhancement', False):
            self._initialize_openai_client()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client for LLM enhancement"""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                self.logger.info("✅ OpenAI client initialized for LLM enhancement")
            else:
                self.logger.warning("⚠️ OPENAI_API_KEY not found, LLM enhancement disabled")
        except ImportError:
            self.logger.warning("⚠️ OpenAI package not available, LLM enhancement disabled")
        except Exception as e:
            self.logger.warning(f"⚠️ OpenAI client initialization failed: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        try:
            # Try to import centralized logging
            from logging_config import setup_logging
            return setup_logging(f'{self.pipeline_type}_pipeline')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_pipeline')
    
    def check_progress(self, video_id: str, step_name: str) -> bool:
        """
        Check if a step has already been completed for a video
        
        Args:
            video_id: The video ID to check
            step_name: The step name to check
            
        Returns:
            True if step is completed, False otherwise
        """
        if not self.progress_queue:
            return False
        
        try:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get(step_name) == 'completed':
                self.logger.info(f"Step {step_name} already completed for {video_id}, skipping")
                return True
        except Exception as e:
            self.logger.warning(f"Error checking progress for {video_id}: {e}")
        
        return False
    
    def update_progress(self, video_id: str, step_name: str, status: str, metadata: Dict[str, Any] = None):
        """
        Update progress for a video and step
        
        Args:
            video_id: The video ID
            step_name: The step name
            status: The status (pending, in_progress, completed, failed)
            metadata: Additional metadata about the step
        """
        if not self.progress_queue:
            return
        
        try:
            self.progress_queue.update_video_step(video_id, step_name, status, metadata or {})
            self.logger.debug(f"Updated progress: {video_id} - {step_name}: {status}")
        except Exception as e:
            self.logger.warning(f"Error updating progress for {video_id}: {e}")
    
    def log_processing_summary(self):
        """Log a summary of processing statistics"""
        if self.processing_stats['end_time'] is None:
            self.processing_stats['end_time'] = datetime.now()
        
        duration = self.processing_stats['end_time'] - self.processing_stats['start_time']
        
        self.logger.info(f"🎉 {self.pipeline_type.upper()} Pipeline Processing Summary:")
        self.logger.info(f"  Total processed: {self.processing_stats['total_processed']}")
        self.logger.info(f"  Successful: {self.processing_stats['successful']}")
        self.logger.info(f"  Failed: {self.processing_stats['failed']}")
        self.logger.info(f"  Skipped: {self.processing_stats['skipped']}")
        self.logger.info(f"  Duration: {duration}")
        
        if self.processing_stats['total_processed'] > 0:
            success_rate = (self.processing_stats['successful'] / self.processing_stats['total_processed']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    @abstractmethod
    def process_video(self, video_id: str) -> Dict[str, Any]:
        """
        Process a single video (must be implemented by subclasses)
        
        Args:
            video_id: The video ID to process
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    @abstractmethod
    def process_all_videos(self):
        """Process all videos in the pipeline (must be implemented by subclasses)"""
        pass
    
    def cleanup(self):
        """Cleanup resources and finalize processing"""
        self.processing_stats['end_time'] = datetime.now()
        self.log_processing_summary()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        if exc_type:
            self.logger.error(f"Pipeline failed with exception: {exc_val}")
            return False
        return True
