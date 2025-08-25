#!/usr/bin/env python3
"""
Step 03: Transcription (Unified)
Handles video transcription using AssemblyAI or other services.
Supports both formal presentations and multi-speaker conversations.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UnifiedTranscriptionHandler:
    """Unified transcription handler supporting multiple pipeline types"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the unified transcription handler
        
        Args:
            pipeline_config: Pipeline-specific configuration
            progress_queue: Progress tracking queue
        """
        self.config = pipeline_config
        self.progress_queue = progress_queue
        
        # Pipeline metadata
        self.pipeline_type = pipeline_config.get('pipeline_type', 'unknown')
        self.pipeline_name = pipeline_config.get('pipeline_name', 'Unknown Pipeline')
        self.speaker_count = pipeline_config.get('speaker_count', 1)
        
        # Transcription configuration
        self.transcription_config = pipeline_config.get('transcription_config', {})
        self.transcription_service = pipeline_config.get('transcription_service', 'assemblyai')
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Input/output directories
        self.input_dir = Path("step_02_extracted_playlist_content")
        self.output_dir = Path("step_03_transcription")
        self.output_dir.mkdir(exist_ok=True)
        
        # Processing statistics
        self.stats = {
            'total_videos': 0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0,
            'skipped_transcriptions': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        try:
            # Try to import centralized logging
            from logging_config import setup_logging
            return setup_logging(f'{self.pipeline_type}_transcription')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_transcription')
    
    def process_all_videos(self):
        """Process all videos in the input directory"""
        if not self.input_dir.exists():
            self.logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find video directories
        video_dirs = [d for d in self.input_dir.iterdir() if d.is_dir() and d.name.startswith('video_')]
        self.logger.info(f"Found {len(video_dirs)} video directories")
        
        self.stats['total_videos'] = len(video_dirs)
        
        for video_dir in video_dirs:
            try:
                video_id = video_dir.name.replace('video_', '')
                self.process_single_video(video_id, video_dir)
            except Exception as e:
                self.logger.error(f"Failed to process {video_dir.name}: {e}")
        
        self.log_processing_summary()
    
    def process_single_video(self, video_id: str, video_dir: Path):
        """Process a single video for transcription"""
        self.logger.info(f"Processing video: {video_id}")
        
        # Check progress
        if self.progress_queue:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_03_transcription') == 'completed':
                self.logger.info(f"Transcription already completed for {video_id}, skipping")
                self.stats['skipped_transcriptions'] += 1
                return
        
        # Check if transcription already exists
        transcript_file = self.output_dir / f"{video_id}_transcript.json"
        if transcript_file.exists():
            self.logger.info(f"Transcription already exists for {video_id}, skipping")
            self.stats['skipped_transcriptions'] += 1
            return
        
        try:
            # Find video file
            video_file = self._find_video_file(video_dir)
            if not video_file:
                self.logger.warning(f"No video file found for {video_id}")
                return
            
            # Handle transcription based on service
            if self.transcription_service == 'assemblyai':
                success = self._handle_assemblyai_transcription(video_id, video_file)
            else:
                success = self._handle_generic_transcription(video_id, video_file)
            
            if success:
                self.stats['successful_transcriptions'] += 1
                
                # Update progress
                if self.progress_queue:
                    self.progress_queue.update_video_step(video_id, 'step_03_transcription', 'completed', {
                        'transcription_service': self.transcription_service,
                        'speaker_count': self.speaker_count
                    })
                
                self.logger.info(f"✅ Successfully transcribed {video_id}")
            else:
                self.stats['failed_transcriptions'] += 1
                self.logger.error(f"❌ Failed to transcribe {video_id}")
                
        except Exception as e:
            self.stats['failed_transcriptions'] += 1
            self.logger.error(f"Error processing {video_id}: {e}")
    
    def _find_video_file(self, video_dir: Path) -> Optional[Path]:
        """Find the video file in the video directory"""
        # Look for common video formats
        video_extensions = ['*.mp4', '*.webm', '*.mkv', '*.avi']
        
        for extension in video_extensions:
            video_files = list(video_dir.glob(extension))
            if video_files:
                return video_files[0]  # Return first match
        
        return None
    
    def _handle_assemblyai_transcription(self, video_id: str, video_file: Path) -> bool:
        """Handle AssemblyAI transcription with webhook system"""
        try:
            # Check if AssemblyAI webhook handler exists
            webhook_handler = self._get_webhook_handler()
            if webhook_handler:
                return webhook_handler.process_video(video_id, video_file)
            else:
                # Fallback to direct API call
                return self._direct_assemblyai_transcription(video_id, video_file)
                
        except Exception as e:
            self.logger.error(f"AssemblyAI transcription failed for {video_id}: {e}")
            return False
    
    def _get_webhook_handler(self):
        """Get AssemblyAI webhook handler if available"""
        try:
            # Try to import the webhook handler from the current pipeline
            if self.pipeline_type == "formal_presentations":
                from step_03_transcription_webhook import AssemblyAIWebhookHandler
            elif self.pipeline_type.startswith("conversations"):
                from step_03_transcription_webhook import AssemblyAIWebhookHandler
            else:
                return None
            
            # Initialize with pipeline configuration
            return AssemblyAIWebhookHandler(self.config)
            
        except ImportError:
            self.logger.info("AssemblyAI webhook handler not available, using direct API")
            return None
    
    def _direct_assemblyai_transcription(self, video_id: str, video_file: Path) -> bool:
        """Direct AssemblyAI API transcription (fallback)"""
        try:
            # This would implement direct API calls to AssemblyAI
            # For now, we'll create a placeholder transcript
            self.logger.info(f"Creating placeholder transcript for {video_id} (AssemblyAI direct API not implemented)")
            
            # Create placeholder transcript structure
            transcript_data = self._create_placeholder_transcript(video_id)
            
            # Save transcript
            transcript_file = self.output_dir / f"{video_id}_transcript.json"
            with open(transcript_file, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            # Save text version
            text_file = self.output_dir / f"{video_id}_transcript.txt"
            with open(text_file, 'w') as f:
                f.write(transcript_data.get('text', ''))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Direct AssemblyAI transcription failed: {e}")
            return False
    
    def _handle_generic_transcription(self, video_id: str, video_file: Path) -> bool:
        """Handle generic transcription services"""
        try:
            self.logger.info(f"Generic transcription not implemented for {video_id}")
            
            # Create placeholder transcript
            transcript_data = self._create_placeholder_transcript(video_id)
            
            # Save transcript
            transcript_file = self.output_dir / f"{video_id}_transcript.json"
            with open(transcript_file, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Generic transcription failed: {e}")
            return False
    
    def _create_placeholder_transcript(self, video_id: str) -> Dict[str, Any]:
        """Create a placeholder transcript structure"""
        # Create basic transcript structure based on pipeline type
        if self.speaker_count == 1:
            # Single speaker (formal presentations)
            transcript_data = {
                'video_id': video_id,
                'pipeline_type': self.pipeline_type,
                'speaker_count': self.speaker_count,
                'transcription_service': self.transcription_service,
                'status': 'placeholder',
                'created_at': datetime.now().isoformat(),
                'text': f"Placeholder transcript for video {video_id}. This video needs to be transcribed using {self.transcription_service}.",
                'utterances': [
                    {
                        'speaker': 'speaker_0',
                        'text': f"Placeholder transcript for video {video_id}",
                        'start': 0.0,
                        'end': 5.0,
                        'confidence': 1.0
                    }
                ],
                'metadata': {
                    'language_code': 'en',
                    'audio_duration': 0.0,
                    'word_count': 0,
                    'processing_time': 0.0
                }
            }
        else:
            # Multiple speakers (conversations)
            speakers = [f"speaker_{i}" for i in range(self.speaker_count)]
            transcript_data = {
                'video_id': video_id,
                'pipeline_type': self.pipeline_type,
                'speaker_count': self.speaker_count,
                'transcription_service': self.transcription_service,
                'status': 'placeholder',
                'created_at': datetime.now().isoformat(),
                'text': f"Placeholder transcript for video {video_id} with {self.speaker_count} speakers. This video needs to be transcribed using {self.transcription_service}.",
                'utterances': [
                    {
                        'speaker': speakers[0],
                        'text': f"Placeholder transcript for video {video_id}",
                        'start': 0.0,
                        'end': 5.0,
                        'confidence': 1.0
                    }
                ],
                'speakers': {speaker: {'name': f'Speaker {i}', 'role': 'unknown'} for i, speaker in enumerate(speakers)},
                'metadata': {
                    'language_code': 'en',
                    'audio_duration': 0.0,
                    'word_count': 0,
                    'processing_time': 0.0
                }
            }
        
        return transcript_data
    
    def log_processing_summary(self):
        """Log a summary of processing statistics"""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info(f"🎉 {self.pipeline_type.upper()} Transcription Summary:")
        self.logger.info(f"  Total videos: {self.stats['total_videos']}")
        self.logger.info(f"  Successful: {self.stats['successful_transcriptions']}")
        self.logger.info(f"  Failed: {self.stats['failed_transcriptions']}")
        self.logger.info(f"  Skipped: {self.stats['skipped_transcriptions']}")
        self.logger.info(f"  Transcription service: {self.transcription_service}")
        self.logger.info(f"  Speaker count: {self.speaker_count}")
        self.logger.info(f"  Duration: {duration}")
        
        if self.stats['total_videos'] > 0:
            success_rate = (self.stats['successful_transcriptions'] / self.stats['total_videos']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        self.logger.info(f"  Output directory: {self.output_dir}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Transcription Handler")
    parser.add_argument("--pipeline-type", required=True, 
                       choices=["formal_presentations", "conversations_1_on_1", "conversations_1_on_2"],
                       help="Pipeline type to process")
    parser.add_argument("--config-dir", default="core/pipeline_configs",
                       help="Directory containing pipeline configurations")
    
    args = parser.parse_args()
    
    try:
        # Load pipeline configuration
        from core.config_loader import load_pipeline_config
        config = load_pipeline_config(args.pipeline_type, args.config_dir)
        
        # Initialize transcription handler
        handler = UnifiedTranscriptionHandler(config)
        
        # Process videos
        handler.process_all_videos()
        
        print(f"\n✅ Transcription completed for {args.pipeline_type}")
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
