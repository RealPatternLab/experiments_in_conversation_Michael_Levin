#!/usr/bin/env python3
"""
Step 03: Enhanced Transcription with Webhooks and Polling
Downloads videos, extracts audio, submits transcriptions using AssemblyAI webhooks,
and optionally monitors completion via polling for reliability.
"""

import os
import json
import logging
import subprocess
import tempfile
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import yt_dlp
from dotenv import load_dotenv
import requests
from pipeline_progress_queue import get_progress_queue

# Load environment variables
load_dotenv()

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('transcription_webhook')

class VideoTranscriberWebhook:
    def __init__(self, progress_queue=None):
        self.assemblyai_api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not self.assemblyai_api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
        
        self.input_dir = Path("step_02_extracted_playlist_content")
        self.output_dir = Path("step_03_transcription")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = progress_queue
        
        # AssemblyAI configuration
        self.assemblyai_url = "https://api.assemblyai.com/v2"
        self.headers = {
            "authorization": self.assemblyai_api_key,
            "content-type": "application/json"
        }
        
        # Webhook configuration
        self.webhook_url = os.getenv('WEBHOOK_URL', 'https://experimentsinconversationmichaellevin-neodev.streamlit.app/webhook')
        self.webhook_auth_header = os.getenv('WEBHOOK_AUTH_HEADER', 'X-Webhook-Secret')
        self.webhook_auth_value = os.getenv('WEBHOOK_AUTH_VALUE', 'your-secret-value')
        
        logger.info(f"Webhook URL: {self.webhook_url}")
    
    def process_all_videos(self):
        """Process all videos in the input directory"""
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Load video metadata
        metadata_file = self.input_dir / "download_summary.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file {metadata_file} not found")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Handle both old and new metadata structures
        videos = metadata.get('results', [])  # New structure
        if not videos:
            videos = metadata.get('videos', [])  # Fallback to old structure
        logger.info(f"Found {len(videos)} videos to process")
        
        # Track processing statistics
        total_videos = len(videos)
        new_submissions = 0
        existing_transcripts = 0
        failed_submissions = 0
        
        for video in videos:
            try:
                result = self.process_single_video(video)
                if result == 'new':
                    new_submissions += 1
                elif result == 'existing':
                    existing_transcripts += 1
                elif result == 'failed':
                    failed_submissions += 1
            except Exception as e:
                logger.error(f"Failed to process video {video.get('video_id', 'unknown')}: {e}")
                failed_submissions += 1
        
        # Log summary
        logger.info(f"Transcription Submission Summary:")
        logger.info(f"  Total videos: {total_videos}")
        logger.info(f"  New submissions: {new_submissions}")
        logger.info(f"  Existing transcripts: {existing_transcripts}")
        logger.info(f"  Failed submissions: {failed_submissions}")
        if total_videos > 0:
            success_rate = ((new_submissions + existing_transcripts) / total_videos * 100)
            logger.info(f"  Success rate: {success_rate:.1f}%")
        else:
            logger.info(f"  Success rate: N/A (no videos to process)")
        
        logger.info(f"📝 Note: Transcripts are being processed asynchronously via webhooks.")
        logger.info(f"📝 Check the webhook status page in your Streamlit app to monitor progress.")
    
    def process_single_video(self, video_info: Dict[str, Any]):
        """Process a single video"""
        video_id = video_info['video_id']
        video_title = video_info.get('title', 'Unknown Title')
        
        # Handle different metadata structures
        video_path = None
        if 'local_path' in video_info:
            video_path = video_info['local_path']
        elif 'download_result' in video_info and 'video_file' in video_info['download_result']:
            # Construct full path from relative path in metadata
            relative_path = video_info['download_result']['video_file']
            # The relative path is from the pipeline directory, so construct the full path
            video_path = Path.cwd() / relative_path
        
        if not video_path or not Path(video_path).exists():
            logger.warning(f"Video file not found for {video_id} at path: {video_path}")
            return 'failed'
        
        # Check if transcript already exists using progress queue
        if self.progress_queue:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_03_transcription') == 'completed':
                logger.info(f"Transcript already completed for {video_id} (progress queue), skipping")
                # Sync webhook file to mark any pending transcription as completed
                self.sync_webhook_for_existing_transcript(video_id, video_title)
                return 'existing'
        
        # Fallback: Check if transcript file exists (for backward compatibility)
        transcript_file = self.output_dir / f"{video_id}_transcript.json"
        if transcript_file.exists():
            logger.info(f"Transcript already exists for {video_id} (file check), skipping")
            # Sync webhook file to mark any pending transcription as completed
            self.sync_webhook_for_existing_transcript(video_id, video_title)
            
            # CRITICAL: Update progress queue to mark step 3 as completed
            # This prevents the pipeline from hanging on this step
            if self.progress_queue:
                self.progress_queue.update_video_step_status(
                    video_id,
                    'step_03_transcription',
                    'completed',
                    metadata={
                        'transcript_file': str(transcript_file),
                        'text_file': str(self.output_dir / f"{video_id}_transcript.txt"),
                        'completed_at': datetime.now().isoformat(),
                        'status': 'existing_transcript_detected'
                    }
                )
                logger.info(f"📊 Progress queue updated: step 3 completed for {video_id} (existing transcript)")
            
            return 'existing'
        
        logger.info(f"Processing video: {video_id} - {video_title}")
        
        # Extract audio
        audio_path = self.extract_audio(video_path, video_id)
        if not audio_path:
            logger.error(f"Failed to extract audio for {video_id}")
            return 'failed'
        
        # Submit transcription with webhook
        transcript_id = self.submit_transcript_with_webhook(audio_path, video_id, video_title)
        if not transcript_id:
            logger.error(f"Failed to submit transcription for {video_id}")
            return 'failed'
        
        # Save submission metadata
        self.save_submission_metadata(transcript_id, video_id, video_info)
        
        # Clean up audio file
        if audio_path.exists():
            audio_path.unlink()
        
        logger.info(f"Successfully submitted transcription for video: {video_id} (ID: {transcript_id})")
        return 'new'
    
    def extract_audio(self, video_path: str, video_id: str) -> Optional[Path]:
        """Extract audio from video file"""
        try:
            audio_path = self.output_dir / f"{video_id}_audio.wav"
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '44100',  # 44.1kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return None
            
            logger.info(f"Audio extracted: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return None
    
    def submit_transcript_with_webhook(self, audio_path: Path, video_id: str, video_title: str) -> Optional[str]:
        """Submit transcription using AssemblyAI with webhook"""
        try:
            # Upload audio file
            upload_url = f"{self.assemblyai_url}/upload"
            with open(audio_path, 'rb') as f:
                response = requests.post(
                    upload_url,
                    headers=self.headers,
                    data=f
                )
            
            if response.status_code != 200:
                logger.error(f"Upload failed: {response.text}")
                return None
            
            upload_url = response.json()['upload_url']
            logger.info(f"Audio uploaded: {upload_url}")
            
            # Request transcription with webhook
            transcript_url = f"{self.assemblyai_url}/transcript"
            transcript_request = {
                "audio_url": upload_url,
                "speaker_labels": True,  # Enable speaker diarization
                "speakers_expected": 1,  # We expect 1 speaker (Michael Levin)
                "auto_highlights": True,  # Enable automatic topic detection
                "entity_detection": True,  # Enable entity detection
                "auto_chapters": True,    # Enable automatic chapter detection
                "punctuate": True,        # Enable punctuation
                "format_text": True,      # Enable text formatting
                "webhook_url": self.webhook_url,  # Webhook endpoint
                "webhook_auth_header_name": self.webhook_auth_header,
                "webhook_auth_header_value": self.webhook_auth_value
            }
            
            response = requests.post(
                transcript_url,
                json=transcript_request,
                headers=self.headers
            )
            
            if response.status_code != 200:
                logger.error(f"Transcription request failed: {response.text}")
                return None
            
            transcript_id = response.json()['id']
            logger.info(f"Transcription submitted with webhook: {transcript_id}")
            
            # Add to pending transcriptions in webhook storage
            self.add_pending_transcription(transcript_id, video_id, video_title)
            
            return transcript_id
            
        except Exception as e:
            logger.error(f"Failed to submit transcript: {e}")
            return None
    
    def add_pending_transcription(self, transcript_id: str, video_id: str, video_title: str):
        """Add transcription to pending list in webhook storage"""
        try:
            webhook_file = "logs/assemblyai_webhooks.json"
            webhook_data = {}
            
            # Load existing webhook data
            if os.path.exists(webhook_file):
                with open(webhook_file, 'r') as f:
                    webhook_data = json.load(f)
            
            # Initialize if needed
            if "pending_transcriptions" not in webhook_data:
                webhook_data["pending_transcriptions"] = {}
            if "completed_transcriptions" not in webhook_data:
                webhook_data["completed_transcriptions"] = {}
            
            # Add pending transcription
            webhook_data["pending_transcriptions"][transcript_id] = {
                "video_id": video_id,
                "video_title": video_title,
                "submitted_at": self.get_current_timestamp(),
                "status": "pending"
            }
            
            # Save updated webhook data
            with open(webhook_file, 'w') as f:
                json.dump(webhook_data, f, indent=2)
            
            logger.info(f"Added {transcript_id} to pending transcriptions")
            
        except Exception as e:
            logger.error(f"Failed to add pending transcription: {e}")
    
    def get_current_timestamp(self):
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_submission_metadata(self, transcript_id: str, video_id: str, video_info: Dict[str, Any]):
        """Save submission metadata to file"""
        try:
            submission_metadata = {
                'video_id': video_id,
                'video_metadata': video_info,
                'transcript_submission': {
                    'assemblyai_id': transcript_id,
                    'submitted_at': self.get_current_timestamp(),
                    'webhook_url': self.webhook_url,
                    'status': 'submitted'
                }
            }
            
            # Save to file
            output_file = self.output_dir / f"{video_id}_submission.json"
            with open(output_file, 'w') as f:
                json.dump(submission_metadata, f, indent=2)
            
            logger.info(f"Submission metadata saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save submission metadata: {e}")

    def monitor_completion(self, poll_interval: int = 120, max_monitoring_time: int = 3600):
        """
        Monitor pending transcriptions for completion via polling.
        
        Args:
            poll_interval: Time between API checks in seconds (default: 4 minutes)
            max_monitoring_time: Maximum time to monitor in seconds (default: 1 hour)
        """
        logger.info(f"🔍 Starting completion monitoring with {poll_interval}s intervals...")
        logger.info(f"⏰ Maximum monitoring time: {max_monitoring_time}s")
        
        start_time = time.time()
        
        while True:
            try:
                # Check if we've exceeded maximum monitoring time
                elapsed_time = time.time() - start_time
                if elapsed_time > max_monitoring_time:
                    logger.warning(f"⏰ Maximum monitoring time ({max_monitoring_time}s) exceeded. Stopping monitoring.")
                    break
                
                # Load current webhook data
                webhook_file = "logs/assemblyai_webhooks.json"
                if not os.path.exists(webhook_file):
                    logger.warning("Webhook file not found, waiting...")
                    time.sleep(poll_interval)
                    continue
                
                with open(webhook_file, 'r') as f:
                    webhook_data = json.load(f)
                
                # Clean up any duplicate entries (safety check)
                self.cleanup_duplicate_entries(webhook_data)
                
                pending = webhook_data.get("pending_transcriptions", {})
                if not pending:
                    logger.info("✅ No pending transcriptions found. All done!")
                    break
                
                # Additional safety check: if all videos in pending are already completed, exit
                completed_video_ids = {info['video_id'] for info in webhook_data.get("completed_transcriptions", {}).values()}
                all_pending_completed = all(info['video_id'] in completed_video_ids for info in pending.values())
                if all_pending_completed:
                    logger.warning("🔧 All pending transcriptions are already completed. Cleaning up and exiting.")
                    # Clean up the pending entries
                    webhook_data["pending_transcriptions"] = {}
                    with open(webhook_file, 'w') as f:
                        json.dump(webhook_data, f, indent=2)
                    logger.info("✅ Pending entries cleaned up. Exiting monitoring.")
                    break
                
                logger.info(f"📊 Checking {len(pending)} pending transcriptions...")
                
                # Check each pending transcription
                completed_this_round = 0
                for transcript_id, info in list(pending.items()):
                    video_id = info['video_id']
                    video_title = info['video_title']
                    
                    logger.info(f"🔍 Checking {transcript_id} for video {video_id}...")
                    
                    # Check status with AssemblyAI
                    status = self.check_transcript_status(transcript_id)
                    
                    if status == 'completed':
                        # Download and save transcript
                        if self.download_completed_transcript(transcript_id, video_id, video_title):
                            # Note: move_to_completed is now called inside download_completed_transcript
                            completed_this_round += 1
                            logger.info(f"✅ {transcript_id} completed and downloaded!")
                        else:
                            logger.error(f"❌ Failed to download transcript for {transcript_id}")
                    elif status == 'error':
                        logger.error(f"❌ {transcript_id} failed with error")
                        # Could move to failed status here
                    else:
                        logger.info(f"⏳ {transcript_id} still processing...")
                
                if completed_this_round > 0:
                    logger.info(f"🎉 Completed {completed_this_round} transcriptions this round!")
                    
                    # Check if we're done after processing this round
                    with open(webhook_file, 'r') as f:
                        webhook_data = json.load(f)
                    pending = webhook_data.get("pending_transcriptions", {})
                    if not pending:
                        logger.info("✅ All transcriptions completed! Exiting monitoring.")
                        break  # Exit the loop if no more pending
                else:
                    logger.info(f"⏳ No completions yet. Waiting {poll_interval}s...")
                
                # Wait before next check (only if still pending)
                if pending:
                    time.sleep(poll_interval)
                else:
                    logger.info("✅ No more pending transcriptions, exiting monitoring loop.")
                    break
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                time.sleep(poll_interval)
    
    def check_transcript_status(self, transcript_id: str) -> str:
        """Check the status of a transcript with AssemblyAI"""
        try:
            url = f"{self.assemblyai_url}/transcript/{transcript_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('status', 'unknown')
            else:
                logger.error(f"Failed to check status: {response.text}")
                return 'error'
                
        except Exception as e:
            logger.error(f"Error checking transcript status: {e}")
            return 'error'
    
    def download_completed_transcript(self, transcript_id: str, video_id: str, video_title: str) -> bool:
        """Download a completed transcript from AssemblyAI"""
        try:
            # Get full transcript data
            url = f"{self.assemblyai_url}/transcript/{transcript_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to retrieve transcript: {response.text}")
                return False
            
            transcript_data = response.json()
            
            # Save full transcript JSON
            transcript_file = self.output_dir / f"{video_id}_transcript.json"
            with open(transcript_file, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            # Save plain text version
            text_file = self.output_dir / f"{video_id}_transcript.txt"
            with open(text_file, 'w') as f:
                f.write(transcript_data.get('text', ''))
            
            logger.info(f"📄 Transcript saved: {transcript_file}")
            logger.info(f"📝 Text saved: {text_file}")
            
            # Update progress queue
            if self.progress_queue:
                self.progress_queue.update_video_step_status(
                    video_id,
                    'step_03_transcription',
                    'completed',
                    metadata={
                        'transcript_file': str(transcript_file),
                        'text_file': str(text_file),
                        'completed_at': datetime.now().isoformat()
                    }
                )
                logger.info(f"📊 Progress queue updated: step 3 completed for {video_id}")
            
            # CRITICAL: Immediately update webhook file to mark as completed
            # This prevents the monitoring loop from hanging
            self.move_to_completed(transcript_id, video_id, video_title)
            logger.info(f"🔗 Webhook file updated: {transcript_id} marked as completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download transcript: {e}")
            return False
    
    def move_to_completed(self, transcript_id: str, video_id: str, video_title: str):
        """Move transcript from pending to completed in webhook storage"""
        try:
            webhook_file = "logs/assemblyai_webhooks.json"
            
            with open(webhook_file, 'r') as f:
                webhook_data = json.load(f)
            
            # Check if already completed (safety check)
            if transcript_id in webhook_data.get("completed_transcriptions", {}):
                logger.info(f"Transcript {transcript_id} already marked as completed, skipping")
                return
            
            # Remove from pending
            if transcript_id in webhook_data.get("pending_transcriptions", {}):
                pending_info = webhook_data["pending_transcriptions"].pop(transcript_id)
                logger.info(f"Removed {transcript_id} from pending transcriptions")
            else:
                logger.warning(f"Transcript {transcript_id} not found in pending, may have been moved already")
            
            # Add to completed
            webhook_data["completed_transcriptions"][transcript_id] = {
                "video_id": video_id,
                "video_title": video_title,
                "completed_at": self.get_current_timestamp(),
                "status": "completed"
            }
            
            # Save updated data
            with open(webhook_file, 'w') as f:
                json.dump(webhook_data, f, indent=2)
            
            logger.info(f"Added {transcript_id} to completed transcriptions")
            
        except Exception as e:
            logger.error(f"Failed to move transcript to completed: {e}")
    
    def sync_webhook_for_existing_transcript(self, video_id: str, video_title: str):
        """Sync webhook file when existing transcript is detected"""
        try:
            webhook_file = "logs/assemblyai_webhooks.json"
            
            # Check if webhook file exists
            if not os.path.exists(webhook_file):
                logger.info(f"No webhook file found for {video_id}, skipping webhook sync")
                return
            
            with open(webhook_file, 'r') as f:
                webhook_data = json.load(f)
            
            # Find any pending transcription for this video
            pending = webhook_data.get("pending_transcriptions", {})
            transcript_id_to_complete = None
            
            for transcript_id, info in pending.items():
                if info.get('video_id') == video_id:
                    transcript_id_to_complete = transcript_id
                    break
            
            if transcript_id_to_complete:
                # Move from pending to completed
                pending_info = pending.pop(transcript_id_to_complete)
                webhook_data["completed_transcriptions"][transcript_id_to_complete] = {
                    "video_id": video_id,
                    "video_title": video_title,
                    "completed_at": self.get_current_timestamp(),
                    "status": "completed"
                }
                
                # Save updated data
                with open(webhook_file, 'w') as f:
                    json.dump(webhook_data, f, indent=2)
                
                logger.info(f"📊 Webhook synced: {transcript_id_to_complete} marked as completed for {video_id}")
            else:
                logger.debug(f"No pending transcription found in webhook for {video_id}")
                
        except Exception as e:
            logger.warning(f"Failed to sync webhook for {video_id}: {e}")
    
    def cleanup_duplicate_entries(self, webhook_data: dict):
        """Clean up any duplicate entries between pending and completed"""
        try:
            pending = webhook_data.get("pending_transcriptions", {})
            completed = webhook_data.get("completed_transcriptions", {})
            
            # Find videos that appear in both pending and completed
            pending_video_ids = {info['video_id'] for info in pending.values()}
            completed_video_ids = {info['video_id'] for info in completed.values()}
            duplicates = pending_video_ids.intersection(completed_video_ids)
            
            if duplicates:
                logger.warning(f"🔧 Found duplicate entries for videos: {duplicates}")
                
                # Remove pending entries for videos that are already completed
                for transcript_id, info in list(pending.items()):
                    if info['video_id'] in duplicates:
                        logger.info(f"🧹 Removing duplicate pending entry: {transcript_id} for {info['video_id']}")
                        pending.pop(transcript_id)
                
                # Save cleaned data
                webhook_file = "logs/assemblyai_webhooks.json"
                with open(webhook_file, 'w') as f:
                    json.dump(webhook_data, f, indent=2)
                
                logger.info("✅ Duplicate entries cleaned up")
                
        except Exception as e:
            logger.error(f"Failed to cleanup duplicate entries: {e}")

def main():
    """Main execution function with command line arguments"""
    parser = argparse.ArgumentParser(description='Video Transcription with AssemblyAI')
    parser.add_argument('--monitor', action='store_true', 
                       help='Monitor transcript completion after submission')
    parser.add_argument('--poll-interval', type=int, default=240,
                       help='Polling interval in seconds (default: 240)')
    parser.add_argument('--submit-only', action='store_true',
                       help='Only submit transcriptions, do not monitor')
    
    args = parser.parse_args()
    
    try:
        # Initialize progress queue
        progress_queue = get_progress_queue()
        logger.info("✅ Progress queue initialized")
        
        transcriber = VideoTranscriberWebhook(progress_queue)
        
        if args.submit_only:
            # Only submit transcriptions
            logger.info("📤 Running in submit-only mode...")
            transcriber.process_all_videos()
            logger.info("✅ Transcription submission completed")
            logger.info("📝 Next steps:")
            logger.info("📝 1. Run with --monitor to watch for completion")
            logger.info("📝 2. Check webhook status in Streamlit app")
            logger.info("📝 3. Run step 04 when transcriptions are complete")
        else:
            # Submit transcriptions first
            logger.info("📤 Submitting transcriptions...")
            transcriber.process_all_videos()
            
            if args.monitor:
                # Then monitor for completion
                logger.info("🔍 Starting completion monitoring...")
                transcriber.monitor_completion(args.poll_interval)
            else:
                # Default behavior: submit then monitor
                logger.info("🔍 Starting completion monitoring (default)...")
                transcriber.monitor_completion(args.poll_interval)
                
    except Exception as e:
        logger.error(f"Transcription step failed: {e}")
        raise

if __name__ == "__main__":
    main()
