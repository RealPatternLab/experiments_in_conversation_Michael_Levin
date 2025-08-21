#!/usr/bin/env python3
"""
Step 03: Enhanced Transcription with Speaker Diarization for Conversations
Downloads videos, extracts audio, submits transcriptions using AssemblyAI webhooks
with speaker diarization enabled for conversations between Michael Levin and other researchers.
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

class ConversationsVideoTranscriberWebhook:
    def __init__(self, progress_queue=None):
        self.assemblyai_api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not self.assemblyai_api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
        
        self.input_dir = Path("step_02_extracted_playlist_content")
        self.output_dir = Path("step_03_transcription")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = progress_queue or get_progress_queue()
        
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
        logger.info(f"🎭 Conversations Pipeline: Speaker diarization enabled for 2 speakers")
    
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
        videos = metadata.get('download_details', [])  # New structure
        if not videos:
            videos = metadata.get('videos', [])  # Fallback to old structure
        logger.info(f"Found {len(videos)} videos to process")
        
        # Track processing statistics
        total_videos = len(videos)
        completed_transcripts = 0
        submitted_transcripts = 0
        existing_transcripts = 0
        failed_submissions = 0
        
        for video in videos:
            try:
                result = self.process_single_video(video)
                if result == 'completed':
                    completed_transcripts += 1
                elif result == 'submitted':
                    submitted_transcripts += 1
                elif result == 'existing':
                    existing_transcripts += 1
                elif result == 'failed':
                    failed_submissions += 1
            except Exception as e:
                logger.error(f"Failed to process video {video.get('video_id', 'unknown')}: {e}")
                failed_submissions += 1
        
        # Log summary
        logger.info(f"Transcription Processing Summary:")
        logger.info(f"  Total videos: {total_videos}")
        logger.info(f"  Completed transcripts: {completed_transcripts}")
        logger.info(f"  Submitted (pending): {submitted_transcripts}")
        logger.info(f"  Existing transcripts: {existing_transcripts}")
        logger.info(f"  Failed submissions: {failed_submissions}")
        if total_videos > 0:
            success_rate = ((completed_transcripts + existing_transcripts) / total_videos * 100)
            logger.info(f"  Success rate: {success_rate:.1f}%")
        else:
            logger.info(f"  Success rate: N/A (no videos to process)")
        
        if completed_transcripts > 0:
            logger.info(f"🎭 Speaker diarization completed for {completed_transcripts} videos")
            logger.info(f"🎭 Speaker A and Speaker B identified and ready for labeling")
            logger.info(f"📁 Transcript files saved and ready for Step 4 (Semantic Chunking)")
        
        if submitted_transcripts > 0:
            logger.info(f"🚀 {submitted_transcripts} transcriptions submitted and processing asynchronously")
            logger.info(f"📝 Check webhook status or re-run without --submit-only to wait for completion")
    
    def process_single_video(self, video_info: Dict[str, Any]):
        """Process a single video"""
        video_id = video_info['video_id']
        video_title = video_info.get('title', 'Unknown Title')
        
        # Handle different metadata structures
        video_path = None
        if 'download_path' in video_info:
            video_path = video_info['download_path']
        elif 'local_path' in video_info:
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
                        'status': 'existing_transcript_detected',
                        'speaker_diarization': True,
                        'speakers_expected': 2
                    }
                )
                logger.info(f"📊 Progress queue updated: step 3 completed for {video_id} (existing transcript)")
            
            return 'existing'
        
        logger.info(f"🎭 Processing conversation video: {video_id} - {video_title}")
        logger.info(f"🎭 Expected: 2 speakers (Michael Levin + 1 other researcher)")
        
        # Extract audio
        audio_path = self.extract_audio(video_path, video_id)
        if not audio_path:
            logger.error(f"Failed to extract audio for {video_id}")
            return 'failed'
        
        # Submit transcription with webhook and speaker diarization
        transcript_id = self.submit_transcript_with_webhook(audio_path, video_id, video_title)
        if not transcript_id:
            logger.error(f"Failed to submit transcription for {video_id}")
            return 'failed'
        
        # Save submission metadata
        self.save_submission_metadata(transcript_id, video_id, video_info)
        
        # Clean up audio file
        if audio_path.exists():
            audio_path.unlink()
        
        logger.info(f"🎭 Successfully submitted transcription with speaker diarization for video: {video_id} (ID: {transcript_id})")
        
        # Check if we should wait for completion
        if hasattr(self, 'submit_only_mode') and self.submit_only_mode:
            logger.info(f"🚀 Submit-only mode: Exiting after submission")
            return 'submitted'
        
        # Wait for transcription to complete and download results
        logger.info(f"⏳ Now waiting for transcription to complete...")
        success = self.wait_for_transcription_completion(transcript_id, video_id, max_wait_minutes=getattr(self, 'max_wait_minutes', 60))
        
        if success:
            logger.info(f"✅ Transcription completed and downloaded for {video_id}")
            return 'completed'
        else:
            logger.error(f"❌ Transcription failed or timed out for {video_id}")
            return 'failed'
    
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
        """Submit transcription using AssemblyAI with webhook and speaker diarization for conversations"""
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
            
            # Request transcription with webhook and conversation-specific settings
            transcript_url = f"{self.assemblyai_url}/transcript"
            transcript_request = {
                "audio_url": upload_url,
                "speaker_labels": True,  # Enable speaker diarization
                "speakers_expected": 2,  # Expect 2 speakers (Michael Levin + 1 other researcher)
                "auto_highlights": True,  # Enable automatic topic detection
                "entity_detection": True,  # Enable entity detection
                "auto_chapters": True,    # Enable automatic chapter detection
                "punctuate": True,        # Enable punctuation
                "format_text": True,      # Enable text formatting
                "webhook_url": self.webhook_url,  # Webhook endpoint
                "webhook_auth_header_name": self.webhook_auth_header,
                "webhook_auth_header_value": self.webhook_auth_value,
                # Conversation-specific enhancements
                "sentiment_analysis": True,  # Analyze sentiment for conversation dynamics
                "iab_categories": True,      # Content categorization for scientific topics
                "content_safety": False,     # Skip content safety (academic content)
                "language_detection": True,  # Ensure English detection
                "boost_param": "high"        # Higher accuracy for scientific content
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
            logger.info(f"🎭 Transcription submitted with speaker diarization: {transcript_id}")
            logger.info(f"🎭 Configuration: 2 speakers expected, conversation-optimized")
            
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
            
            # Add pending transcription with conversation-specific metadata
            webhook_data["pending_transcriptions"][transcript_id] = {
                "video_id": video_id,
                "video_title": video_title,
                "submitted_at": self.get_current_timestamp(),
                "status": "pending",
                "pipeline_type": "conversations_1_on_1",
                "speaker_diarization": True,
                "speakers_expected": 2,
                "content_type": "conversation",
                "processing_notes": [
                    "Speaker diarization enabled for 2 speakers",
                    "Michael Levin + 1 other researcher expected",
                    "Conversation-optimized transcription settings",
                    "Q&A patterns will be extracted in Step 4"
                ]
            }
            
            # Save updated webhook data
            with open(webhook_file, 'w') as f:
                json.dump(webhook_data, f, indent=2)
            
            logger.info(f"Added transcription {transcript_id} to pending list")
            
        except Exception as e:
            logger.error(f"Failed to add pending transcription: {e}")
    
    def sync_webhook_for_existing_transcript(self, video_id: str, video_title: str):
        """Sync webhook storage for existing transcripts"""
        try:
            webhook_file = "logs/assemblyai_webhooks.json"
            if not os.path.exists(webhook_file):
                return
            
            with open(webhook_file, 'r') as f:
                webhook_data = json.load(f)
            
            # Move any pending transcriptions to completed
            pending = webhook_data.get("pending_transcriptions", {})
            completed = webhook_data.get("completed_transcriptions", {})
            
            for transcript_id, data in pending.items():
                if data.get("video_id") == video_id:
                    # Move to completed
                    data["status"] = "completed"
                    data["completed_at"] = self.get_current_timestamp()
                    data["completion_method"] = "existing_transcript"
                    data["speaker_diarization"] = True
                    data["speakers_expected"] = 2
                    
                    completed[transcript_id] = data
                    del pending[transcript_id]
                    
                    logger.info(f"Synced existing transcript {transcript_id} in webhook storage")
                    break
            
            # Save updated webhook data
            with open(webhook_file, 'w') as f:
                json.dump(webhook_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to sync webhook for existing transcript: {e}")
    
    def save_submission_metadata(self, transcript_id: str, video_id: str, video_info: Dict[str, Any]):
        """Save metadata about the transcription submission"""
        try:
            metadata = {
                "transcript_id": transcript_id,
                "video_id": video_id,
                "video_title": video_info.get('title', 'Unknown Title'),
                "submitted_at": self.get_current_timestamp(),
                "pipeline_type": "conversations_1_on_1",
                "transcription_config": {
                    "speaker_labels": True,
                    "speakers_expected": 2,
                    "auto_highlights": True,
                    "entity_detection": True,
                    "auto_chapters": True,
                    "sentiment_analysis": True,
                    "iab_categories": True,
                    "content_safety": False,
                    "language_detection": True,
                    "boost_param": "high"
                },
                "conversation_features": {
                    "speaker_diarization": True,
                    "qa_extraction_ready": True,
                    "levin_knowledge_focus": True,
                    "conversation_context_preservation": True
                },
                "webhook_url": self.webhook_url,
                "status": "pending"
            }
            
            metadata_file = self.output_dir / f"{video_id}_submission_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved submission metadata: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save submission metadata: {e}")
    
    def get_current_timestamp(self):
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def check_transcription_status(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """Check the status of a transcription"""
        try:
            status_url = f"{self.assemblyai_url}/transcript/{transcript_id}"
            response = requests.get(status_url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get status: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to check transcription status: {e}")
            return None
    
    def wait_for_transcription_completion(self, transcript_id: str, video_id: str, max_wait_minutes: int = 60) -> bool:
        """Wait for transcription to complete and download results"""
        logger.info(f"⏳ Waiting for transcription {transcript_id} to complete...")
        logger.info(f"⏳ Will check every 2 minutes, max wait: {max_wait_minutes} minutes")
        
        check_interval = 120  # 2 minutes in seconds
        max_checks = (max_wait_minutes * 60) // check_interval
        checks_performed = 0
        
        while checks_performed < max_checks:
            checks_performed += 1
            logger.info(f"⏳ Check {checks_performed}/{max_checks}: Checking transcription status...")
            
            status_data = self.check_transcription_status(transcript_id)
            if not status_data:
                logger.warning(f"⚠️ Could not check status for {transcript_id}, retrying...")
                time.sleep(check_interval)
                continue
            
            status = status_data.get('status')
            logger.info(f"📊 Transcription status: {status}")
            
            if status == 'completed':
                logger.info(f"✅ Transcription completed! Downloading results...")
                return self.download_transcription_results(transcript_id, video_id, status_data)
            elif status == 'error':
                error_msg = status_data.get('error', 'Unknown error')
                logger.error(f"❌ Transcription failed: {error_msg}")
                return False
            elif status in ['queued', 'processing']:
                logger.info(f"⏳ Transcription {status}, waiting...")
            else:
                logger.info(f"ℹ️ Unknown status '{status}', waiting...")
            
            # Wait before next check
            if checks_performed < max_checks:
                logger.info(f"⏳ Waiting {check_interval//60} minutes before next check...")
                time.sleep(check_interval)
        
        logger.error(f"⏰ Max wait time ({max_wait_minutes} minutes) exceeded for transcription {transcript_id}")
        return False
    
    def download_transcription_results(self, transcript_id: str, video_id: str, status_data: Dict[str, Any]) -> bool:
        """Download completed transcription results"""
        try:
            logger.info(f"📥 Downloading transcription results for {video_id}...")
            
            # Get the transcript text
            transcript_url = f"{self.assemblyai_url}/transcript/{transcript_id}"
            response = requests.get(transcript_url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to download transcript: {response.text}")
                return False
            
            transcript_data = response.json()
            
            # Save full transcript JSON
            transcript_json_file = self.output_dir / f"{video_id}_transcript.json"
            with open(transcript_json_file, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            # Save plain text transcript
            transcript_text_file = self.output_dir / f"{video_id}_transcript.txt"
            with open(transcript_text_file, 'w') as f:
                f.write(transcript_data.get('text', ''))
            
            # Extract and save speaker information
            speakers_data = self.extract_speaker_information(transcript_data, video_id)
            
            # Update progress queue to mark step 3 as completed
            if self.progress_queue:
                self.progress_queue.update_video_step_status(
                    video_id,
                    'step_03_transcription',
                    'completed',
                    metadata={
                        'transcript_file': str(transcript_json_file),
                        'text_file': str(transcript_text_file),
                        'speakers_file': str(self.output_dir / f"{video_id}_speakers.json"),
                        'completed_at': datetime.now().isoformat(),
                        'transcript_id': transcript_id,
                        'speaker_diarization': True,
                        'speakers_detected': len(speakers_data.get('speakers', [])),
                        'total_utterances': len(transcript_data.get('utterances', [])),
                        'duration_seconds': transcript_data.get('audio_duration', 0) / 1000.0 if transcript_data.get('audio_duration') else 0
                    }
                )
                logger.info(f"📊 Progress queue updated: step 3 completed for {video_id}")
            
            # Update webhook storage
            self.mark_transcription_completed(transcript_id, video_id)
            
            logger.info(f"✅ Transcription results downloaded and saved for {video_id}")
            logger.info(f"📁 Files saved:")
            logger.info(f"   - {transcript_json_file}")
            logger.info(f"   - {transcript_text_file}")
            logger.info(f"   - {self.output_dir / f'{video_id}_speakers.json'}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download transcription results: {e}")
            return False
    
    def extract_speaker_information(self, transcript_data: Dict[str, Any], video_id: str) -> Dict[str, Any]:
        """Extract and organize speaker information from transcript"""
        try:
            speakers_data = {
                'video_id': video_id,
                'pipeline_type': 'conversations_1_on_1',
                'extraction_timestamp': datetime.now().isoformat(),
                'speakers': [],
                'speaker_turns': [],
                'conversation_summary': {}
            }
            
            # Extract unique speakers
            speakers = set()
            utterances = transcript_data.get('utterances', [])
            
            for utterance in utterances:
                speaker = utterance.get('speaker', 'Unknown')
                speakers.add(speaker)
                
                # Track speaker turns
                speaker_turn = {
                    'speaker': speaker,
                    'text': utterance.get('text', ''),
                    'start': utterance.get('start', 0),
                    'end': utterance.get('end', 0),
                    'confidence': utterance.get('confidence', 0),
                    'words': utterance.get('words', [])
                }
                speakers_data['speaker_turns'].append(speaker_turn)
            
            # Convert speakers set to list
            speakers_data['speakers'] = list(speakers)
            
            # Add conversation summary
            if transcript_data.get('summary'):
                speakers_data['conversation_summary']['ai_summary'] = transcript_data['summary']
            
            if transcript_data.get('chapters'):
                speakers_data['conversation_summary']['chapters'] = [
                    {
                        'gist': chapter.get('gist', ''),
                        'headline': chapter.get('headline', ''),
                        'start': chapter.get('start', 0),
                        'end': chapter.get('end', 0)
                    } for chapter in transcript_data['chapters']
                ]
            
            # Save speakers data
            speakers_file = self.output_dir / f"{video_id}_speakers.json"
            with open(speakers_file, 'w') as f:
                json.dump(speakers_data, f, indent=2)
            
            logger.info(f"🎭 Extracted speaker information: {len(speakers)} speakers detected")
            logger.info(f"🎭 Speakers: {speakers_data['speakers']}")
            logger.info(f"🎭 Total utterances: {len(speakers_data['speaker_turns'])}")
            
            return speakers_data
            
        except Exception as e:
            logger.error(f"Failed to extract speaker information: {e}")
            return {}
    
    def mark_transcription_completed(self, transcript_id: str, video_id: str):
        """Mark transcription as completed in webhook storage"""
        try:
            webhook_file = "logs/assemblyai_webhooks.json"
            if not os.path.exists(webhook_file):
                return
            
            with open(webhook_file, 'r') as f:
                webhook_data = json.load(f)
            
            # Move from pending to completed
            pending = webhook_data.get("pending_transcriptions", {})
            completed = webhook_data.get("completed_transcriptions", {})
            
            if transcript_id in pending:
                # Update status and move to completed
                pending[transcript_id]["status"] = "completed"
                pending[transcript_id]["completed_at"] = self.get_current_timestamp()
                pending[transcript_id]["completion_method"] = "polling_download"
                
                # Move to completed
                completed[transcript_id] = pending[transcript_id]
                del pending[transcript_id]
                
                # Save updated webhook data
                with open(webhook_file, 'w') as f:
                    json.dump(webhook_data, f, indent=2)
                
                logger.info(f"📊 Webhook storage updated: {transcript_id} marked as completed")
                
        except Exception as e:
            logger.error(f"Failed to mark transcription completed: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Step 3: Transcribe conversations with speaker diarization")
    parser.add_argument("--check-status", help="Check status of specific transcript ID")
    parser.add_argument("--submit-only", action="store_true", help="Submit transcription and exit (don't wait)")
    parser.add_argument("--max-wait", type=int, default=60, help="Maximum wait time in minutes (default: 60)")
    
    args = parser.parse_args()
    
    if args.check_status:
        # Check status of specific transcript
        transcriber = ConversationsVideoTranscriberWebhook()
        status = transcriber.check_transcription_status(args.check_status)
        if status:
            print(json.dumps(status, indent=2))
        return
    
    # Process all videos
    try:
        transcriber = ConversationsVideoTranscriberWebhook()
        
        if args.submit_only:
            logger.info("🚀 Submit-only mode: Submitting transcriptions without waiting")
            transcriber.submit_only_mode = True
            transcriber.process_all_videos()
        else:
            logger.info(f"⏳ Full processing mode: Will wait up to {args.max_wait} minutes for completion")
            transcriber.max_wait_minutes = args.max_wait
            transcriber.process_all_videos()
        
    except Exception as e:
        logger.error(f"Failed to process transcriptions: {e}")
        raise


if __name__ == "__main__":
    main()
