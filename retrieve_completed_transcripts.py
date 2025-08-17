#!/usr/bin/env python3
"""
Retrieve completed transcriptions from AssemblyAI and convert them to pipeline format.

This script reads the webhook storage file, finds completed transcriptions,
retrieves the full transcript data from AssemblyAI, and saves it in the
format expected by the pipeline.

Usage:
    python retrieve_completed_transcripts.py
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptRetriever:
    def __init__(self):
        self.assemblyai_api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not self.assemblyai_api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")
        
        self.assemblyai_url = "https://api.assemblyai.com/v2"
        self.headers = {
            "authorization": self.assemblyai_api_key,
            "content-type": "application/json"
        }
        
        self.webhook_file = "assemblyai_webhooks.json"
        self.output_dir = Path("step_03_transcription")
        self.output_dir.mkdir(exist_ok=True)
    
    def process_completed_transcriptions(self):
        """Process all completed transcriptions"""
        if not os.path.exists(self.webhook_file):
            logger.error(f"Webhook file {self.webhook_file} not found")
            return
        
        # Load webhook data
        with open(self.webhook_file, 'r') as f:
            webhook_data = json.load(f)
        
        completed = webhook_data.get("completed_transcriptions", {})
        pending = webhook_data.get("pending_transcriptions", {})
        
        logger.info(f"Found {len(completed)} completed transcriptions")
        logger.info(f"Found {len(pending)} pending transcriptions")
        
        # Process completed transcriptions
        processed_count = 0
        for transcript_id, info in completed.items():
            if info.get("status") == "completed":
                try:
                    if self.process_single_transcript(transcript_id, info):
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Failed to process transcript {transcript_id}: {e}")
        
        logger.info(f"Successfully processed {processed_count} completed transcriptions")
    
    def process_single_transcript(self, transcript_id: str, info: Dict[str, Any]) -> bool:
        """Process a single completed transcript"""
        video_id = info.get("video_id")
        video_title = info.get("video_title", "Unknown Title")
        
        if not video_id:
            logger.warning(f"No video_id found for transcript {transcript_id}")
            return False
        
        # Check if transcript file already exists
        transcript_file = self.output_dir / f"{video_id}_transcript.json"
        if transcript_file.exists():
            logger.info(f"Transcript file already exists for {video_id}, skipping")
            return True
        
        logger.info(f"Retrieving transcript {transcript_id} for video {video_id}")
        
        # Retrieve transcript from AssemblyAI
        transcript_data = self.retrieve_transcript(transcript_id)
        if not transcript_data:
            logger.error(f"Failed to retrieve transcript {transcript_id}")
            return False
        
        # Save transcript in pipeline format
        self.save_transcript(transcript_data, video_id, info)
        
        logger.info(f"Successfully processed transcript for {video_id}")
        return True
    
    def retrieve_transcript(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve transcript data from AssemblyAI"""
        try:
            url = f"{self.assemblyai_url}/transcript/{transcript_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to retrieve transcript: {response.text}")
                return None
            
            transcript_data = response.json()
            
            # Check if transcription is actually completed
            if transcript_data.get("status") != "completed":
                logger.warning(f"Transcript {transcript_id} status is {transcript_data.get('status')}, not completed")
                return None
            
            return transcript_data
            
        except Exception as e:
            logger.error(f"Error retrieving transcript {transcript_id}: {e}")
            return None
    
    def save_transcript(self, transcript_data: Dict[str, Any], video_id: str, info: Dict[str, Any]):
        """Save transcript in pipeline format"""
        try:
            # Create enhanced transcript with metadata
            enhanced_transcript = {
                'video_id': video_id,
                'video_metadata': {
                    'title': info.get('video_title', 'Unknown Title'),
                    'submitted_at': info.get('submitted_at'),
                    'completed_at': info.get('completed_at')
                },
                'transcript_metadata': {
                    'assemblyai_id': transcript_data.get('id'),
                    'audio_duration': transcript_data.get('audio_duration'),
                    'confidence': transcript_data.get('confidence'),
                    'language_code': transcript_data.get('language_code'),
                    'punctuate': transcript_data.get('punctuate'),
                    'format_text': transcript_data.get('format_text'),
                    'speaker_labels': transcript_data.get('speaker_labels'),
                    'auto_highlights': transcript_data.get('auto_highlights'),
                    'auto_chapters': transcript_data.get('auto_chapters'),
                    'entity_detection': transcript_data.get('entity_detection')
                },
                'transcript': transcript_data.get('text', ''),
                'utterances': transcript_data.get('utterances', []),
                'chapters': transcript_data.get('chapters', []),
                'highlights': transcript_data.get('auto_highlights_result', {}),
                'entities': transcript_data.get('entity_detection_result', {}),
                'processing_timestamp': transcript_data.get('created_at')
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
            raise

def main():
    """Main execution function"""
    try:
        retriever = TranscriptRetriever()
        retriever.process_completed_transcriptions()
        logger.info("Transcript retrieval completed successfully")
    except Exception as e:
        logger.error(f"Transcript retrieval failed: {e}")
        raise

if __name__ == "__main__":
    main()
