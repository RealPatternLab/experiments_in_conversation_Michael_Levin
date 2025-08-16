#!/usr/bin/env python3
"""
Step 03: Enhanced Transcription
Downloads videos, extracts audio, and creates high-quality transcripts using AssemblyAI.
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import yt_dlp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoTranscriber:
    def __init__(self):
        self.assemblyai_api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not self.assemblyai_api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
        
        self.input_dir = Path("step_02_extracted_playlist_content")
        self.output_dir = Path("step_03_transcription")
        self.output_dir.mkdir(exist_ok=True)
        
        # AssemblyAI configuration
        self.assemblyai_url = "https://api.assemblyai.com/v2"
        self.headers = {
            "authorization": self.assemblyai_api_key,
            "content-type": "application/json"
        }
    
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
        
        # Use the new results structure from updated step 2
        videos = metadata.get('results', [])
        logger.info(f"Found {len(videos)} videos to process")
        
        # Track processing statistics
        total_videos = len(videos)
        new_transcripts = 0
        existing_transcripts = 0
        failed_transcripts = 0
        
        for video in videos:
            try:
                result = self.process_single_video(video)
                if result == 'new':
                    new_transcripts += 1
                elif result == 'existing':
                    existing_transcripts += 1
                elif result == 'failed':
                    failed_transcripts += 1
            except Exception as e:
                logger.error(f"Failed to process video {video.get('video_id', 'unknown')}: {e}")
                failed_transcripts += 1
        
        # Log summary
        logger.info(f"Transcription Summary:")
        logger.info(f"  Total videos: {total_videos}")
        logger.info(f"  New transcripts: {new_transcripts}")
        logger.info(f"  Existing transcripts: {existing_transcripts}")
        logger.info(f"  Failed: {failed_transcripts}")
        logger.info(f"  Success rate: {((new_transcripts + existing_transcripts) / total_videos * 100):.1f}%")
    
    def process_single_video(self, video_info: Dict[str, Any]):
        """Process a single video"""
        video_id = video_info['video_id']
        video_path = video_info.get('local_path')
        
        if not video_path or not Path(video_path).exists():
            logger.warning(f"Video file not found for {video_id}")
            return 'failed'
        
        # Check if transcript already exists
        transcript_file = self.output_dir / f"{video_id}_transcript.json"
        if transcript_file.exists():
            logger.info(f"Transcript already exists for {video_id}, skipping")
            return 'existing'
        
        logger.info(f"Processing video: {video_id}")
        
        # Extract audio
        audio_path = self.extract_audio(video_path, video_id)
        if not audio_path:
            logger.error(f"Failed to extract audio for {video_id}")
            return 'failed'
        
        # Create transcript
        transcript = self.create_transcript(audio_path, video_id)
        if not transcript:
            logger.error(f"Failed to create transcript for {video_id}")
            return 'failed'
        
        # Save transcript
        self.save_transcript(transcript, video_id, video_info)
        
        # Clean up audio file
        if audio_path.exists():
            audio_path.unlink()
        
        logger.info(f"Successfully processed video: {video_id}")
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
    
    def create_transcript(self, audio_path: Path, video_id: str) -> Optional[Dict[str, Any]]:
        """Create transcript using AssemblyAI"""
        try:
            import requests
            
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
            
            # Request transcription
            transcript_url = f"{self.assemblyai_url}/transcript"
            transcript_request = {
                "audio_url": upload_url,
                "speaker_labels": True,  # Enable speaker diarization
                "speakers_expected": 1,  # We expect 1 speaker (Michael Levin)
                "auto_highlights": True,  # Enable automatic topic detection
                "entity_detection": True,  # Enable entity detection
                "auto_chapters": True,    # Enable automatic chapter detection
                "punctuate": True,        # Enable punctuation
                "format_text": True       # Enable text formatting
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
            logger.info(f"Transcription requested: {transcript_id}")
            
            # Poll for completion
            transcript = self.poll_transcript(transcript_id)
            return transcript
            
        except Exception as e:
            logger.error(f"Failed to create transcript: {e}")
            return None
    
    def poll_transcript(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """Poll for transcript completion"""
        import requests
        import time
        
        max_attempts = 60  # 5 minutes with 5-second intervals
        attempts = 0
        
        while attempts < max_attempts:
            response = requests.get(
                f"{self.assemblyai_url}/transcript/{transcript_id}",
                headers=self.headers
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get transcript status: {response.text}")
                return None
            
            status = response.json()['status']
            
            if status == 'completed':
                logger.info(f"Transcription completed: {transcript_id}")
                return response.json()
            elif status == 'error':
                logger.error(f"Transcription failed: {response.json()}")
                return None
            
            logger.info(f"Transcription status: {status}, waiting...")
            time.sleep(5)
            attempts += 1
        
        logger.error(f"Transcription timed out: {transcript_id}")
        return None
    
    def save_transcript(self, transcript: Dict[str, Any], video_id: str, video_info: Dict[str, Any]):
        """Save transcript to file"""
        try:
            # Create enhanced transcript with metadata
            enhanced_transcript = {
                'video_id': video_id,
                'video_metadata': video_info,
                'transcript_metadata': {
                    'assemblyai_id': transcript.get('id'),
                    'audio_duration': transcript.get('audio_duration'),
                    'confidence': transcript.get('confidence'),
                    'language_code': transcript.get('language_code'),
                    'punctuate': transcript.get('punctuate'),
                    'format_text': transcript.get('format_text'),
                    'speaker_labels': transcript.get('speaker_labels'),
                    'auto_highlights': transcript.get('auto_highlights'),
                    'auto_chapters': transcript.get('auto_chapters'),
                    'entity_detection': transcript.get('entity_detection')
                },
                'transcript': transcript.get('text', ''),
                'utterances': transcript.get('utterances', []),
                'chapters': transcript.get('chapters', []),
                'highlights': transcript.get('auto_highlights_result', {}),
                'entities': transcript.get('entity_detection_result', {}),
                'processing_timestamp': transcript.get('created_at')
            }
            
            # Save to file
            output_file = self.output_dir / f"{video_id}_transcript.json"
            with open(output_file, 'w') as f:
                json.dump(enhanced_transcript, f, indent=2)
            
            logger.info(f"Transcript saved: {output_file}")
            
            # Also save plain text version
            text_file = self.output_dir / f"{video_id}_transcript.txt"
            with open(text_file, 'w') as f:
                f.write(enhanced_transcript['transcript'])
            
            logger.info(f"Plain text transcript saved: {text_file}")
            
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")

def main():
    """Main execution function"""
    try:
        transcriber = VideoTranscriber()
        transcriber.process_all_videos()
        logger.info("Transcription step completed successfully")
    except Exception as e:
        logger.error(f"Transcription step failed: {e}")
        raise

if __name__ == "__main__":
    main()
