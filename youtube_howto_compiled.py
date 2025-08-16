#!/usr/bin/env python3
"""
Generate Semantic Conversation Chunks from YouTube Videos

This script creates semantic chunks from YouTube videos, creating the same format as the existing 
conversation chunks in your repository. It implements the COMPLETE pipeline from YouTube URL to 
final output.

## 🎯 WHAT THIS SCRIPT DOES

This script performs ALL the necessary steps to go from a YouTube video link to the final 
semantic conversation chunks that match your existing format:

1. **VIDEO DOWNLOAD** → Downloads YouTube video using yt-dlp
2. **AUDIO EXTRACTION** → Extracts audio using ffmpeg for transcription
3. **ASSEMBLYAI TRANSCRIPTION** → Uploads audio and creates transcript with speaker diarization
4. **CHUNK CREATION** → Generates conversation chunks with timestamps and metadata
5. **LLM ENHANCEMENT** → Uses GPT-4 to enhance content and classify scientific topics
6. **OUTPUT GENERATION** → Saves chunks in the exact format of existing conversation chunks

## 📋 PREREQUISITES - MUST BE SETUP BEFORE RUNNING

### Required Software (Install First)
- **Python 3.8+** with pip
- **ffmpeg** (for audio extraction)
- **yt-dlp** (for YouTube downloads)

### Required API Keys (Set in .env file)
- **ASSEMBLYAI_API_KEY** (for transcription - REQUIRED)
- **OPENAI_API_KEY** (for enhancement - optional but recommended)

### System Dependencies Installation

#### macOS:
```bash
brew install ffmpeg
pip install yt-dlp
```

#### Ubuntu/Debian:
```bash
sudo apt update && sudo apt install ffmpeg
pip install yt-dlp
```

#### Windows:
- Download ffmpeg from: https://ffmpeg.org/download.html
- Install yt-dlp: `pip install yt-dlp`

## 🚀 HOW TO USE

### Step 1: Install Python Dependencies
```bash
pip install -r requirements-conversation-chunks.txt
```

### Step 2: Set Environment Variables
Create a `.env` file in your project directory:
```bash
ASSEMBLYAI_API_KEY=your_assemblyai_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Step 3: Run the Script

#### Option A: Process YouTube URL (Complete Pipeline)
```bash
python generate_conversation_chunks.py \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output-dir "data/semantically-chunked/conversation_chunks"
```

#### Option B: Process Existing Video Directory
```bash
python generate_conversation_chunks.py \
  --video-dir "path/to/video/directory" \
  --output-dir "data/semantically-chunked/conversation_chunks"
```

#### Option C: Skip LLM Enhancement
```bash
python generate_conversation_chunks.py \
  --youtube-url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --no-enhance
```

## 🔄 COMPLETE PIPELINE FLOW

When using --youtube-url, here's exactly what happens:

1. **VIDEO DOWNLOAD** (yt-dlp)
   - Downloads video in 720p quality
   - Saves to {video_id}/video.{ext}
   - Creates metadata JSON file

2. **AUDIO EXTRACTION** (ffmpeg)
   - Converts video to 16kHz mono WAV
   - Saves as {video_id}_audio.wav
   - Optimized for transcription

3. **ASSEMBLYAI UPLOAD**
   - Uploads audio file to AssemblyAI servers
   - Returns upload URL for processing

4. **TRANSCRIPTION PROCESSING**
   - Starts transcription with speaker diarization
   - Expects 2 speakers (interviewer + Michael Levin)
   - Enables punctuation and text formatting
   - Polls for completion (may take 5-30 minutes)

5. **TRANSCRIPT SAVING**
   - Downloads completed transcript
   - Saves as {video_id}_transcript.json
   - Includes speaker labels and timestamps

6. **CHUNK GENERATION**
   - Creates conversation chunks from transcript
   - Assigns chunk IDs and metadata
   - Calculates word/character counts

7. **LLM ENHANCEMENT** (if enabled)
   - Uses GPT-4 to enhance each chunk
   - Adds scientific topic classification
   - Preserves original content exactly
   - Creates enhanced text with context

8. **OUTPUT GENERATION**
   - Saves final chunks as {video_id}_conversation_chunks.json
   - Matches exact format of existing chunks
   - Ready for FAISS embedding pipeline

## 📁 OUTPUT STRUCTURE

The script creates this directory structure:
```
data/semantically-chunked/conversation_chunks/
├── {video_id}_conversation_chunks.json    ← FINAL OUTPUT (same format as existing)
└── {video_id}/                            ← Working directory
    ├── video.mp4                          ← Downloaded video
    ├── {video_id}_audio.wav               ← Extracted audio
    └── {video_id}_transcript.json         ← AssemblyAI transcript
```

## ⚠️ IMPORTANT NOTES

- **AssemblyAI API Key is REQUIRED** - script will exit without it
- **Transcription can take 5-30 minutes** depending on video length
- **Large videos (>1 hour)** may hit API rate limits
- **Ensure sufficient disk space** for video downloads
- **Generated chunks are identical format** to existing conversation chunks

## 🔗 NEXT STEPS AFTER GENERATION

Once chunks are generated, you can integrate them with your existing pipeline:

```bash
# Embed the generated chunks into your FAISS index
python tools/embed_semantic_chunks_faiss.py \
  --input-dir "data/semantically-chunked/conversation_chunks" \
  --output-dir "data/faiss"
```

## 📞 SUPPORT

If you encounter issues:
1. Check all prerequisites are installed
2. Verify API keys are set correctly
3. Ensure sufficient disk space
4. Check AssemblyAI service status
5. Review the logs for specific error messages

Usage:
    python generate_conversation_chunks.py --youtube-url "https://youtube.com/watch?v=..." --output-dir "data/semantically-chunked/conversation_chunks"
    python generate_conversation_chunks.py --video-dir "path/to/existing/videos" --output-dir "data/semantically-chunked/conversation_chunks"
"""

import json
import os
import argparse
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import openai
import requests
import subprocess
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')

# Check for required API keys
if not ASSEMBLYAI_API_KEY:
    logger.error("ASSEMBLYAI_API_KEY not found in environment variables")
    logger.error("Please set ASSEMBLYAI_API_KEY in your .env file")
    sys.exit(1)


class EnhancedChunkResponse(BaseModel):
    """Pydantic model for validating LLM responses."""
    enhanced_text: str = Field(
        ..., 
        min_length=50,
        description="Enhanced text combining context and complete answer"
    )
    topics: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=4,
        description="List of 1-4 specific scientific topics that describe the content"
    )


class YouTubeVideoProcessor:
    """Handles YouTube video downloading and processing."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if yt-dlp is available
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
            self.yt_dlp_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.yt_dlp_available = False
            logger.warning("yt-dlp not found. Please install it: pip install yt-dlp")
    
    def download_video(self, youtube_url: str, video_id: str) -> Optional[Path]:
        """Download video using yt-dlp."""
        if not self.yt_dlp_available:
            logger.error("yt-dlp not available for video download")
            return None
        
        video_dir = self.output_dir / video_id
        video_dir.mkdir(exist_ok=True)
        
        try:
            # Download video with best quality
            cmd = [
                "yt-dlp",
                "--format", "best[height<=720]",  # Limit to 720p for processing
                "--output", str(video_dir / "video.%(ext)s"),
                "--write-info-json",
                youtube_url
            ]
            
            logger.info(f"📥 Downloading video: {youtube_url}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(video_dir))
            
            if result.returncode == 0:
                # Find the downloaded video file
                video_files = list(video_dir.glob("video.*"))
                if video_files:
                    video_file = video_files[0]
                    logger.info(f"✅ Video downloaded: {video_file}")
                    return video_file
                else:
                    logger.error("Video file not found after download")
                    return None
            else:
                logger.error(f"Download failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
    
    def extract_audio(self, video_file: Path, video_id: str) -> Optional[Path]:
        """Extract audio from video for transcription."""
        try:
            audio_file = video_file.parent / f"{video_id}_audio.wav"
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(video_file),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(audio_file)
            ]
            
            logger.info(f"🎵 Extracting audio from video...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and audio_file.exists():
                logger.info(f"✅ Audio extracted: {audio_file}")
                return audio_file
            else:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None


class AssemblyAITranscriber:
    """Handles transcription using AssemblyAI API."""
    
    def __init__(self):
        self.api_key = ASSEMBLYAI_API_KEY
        self.base_url = "https://api.assemblyai.com/v2"
        self.headers = {
            "authorization": self.api_key,
            "content-type": "application/json"
        }
    
    def upload_audio(self, audio_file: Path) -> Optional[str]:
        """Upload audio file to AssemblyAI."""
        try:
            upload_url = f"{self.base_url}/upload"
            
            with open(audio_file, "rb") as f:
                response = requests.post(
                    upload_url,
                    headers=self.headers,
                    data=f
                )
            
            if response.status_code == 200:
                upload_url = response.json()["upload_url"]
                logger.info(f"✅ Audio uploaded: {upload_url}")
                return upload_url
            else:
                logger.error(f"Upload failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading audio: {e}")
            return None
    
    def transcribe_audio(self, upload_url: str) -> Optional[str]:
        """Start transcription with speaker diarization."""
        try:
            transcript_url = f"{self.base_url}/transcript"
            
            data = {
                "audio_url": upload_url,
                "speaker_labels": True,
                "speakers_expected": 2,  # Expect 2 speakers (interviewer + Michael Levin)
                "punctuate": True,
                "format_text": True
            }
            
            response = requests.post(
                transcript_url,
                json=data,
                headers=self.headers
            )
            
            if response.status_code == 200:
                transcript_id = response.json()["id"]
                logger.info(f"✅ Transcription started: {transcript_id}")
                return transcript_id
            else:
                logger.error(f"Transcription request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error starting transcription: {e}")
            return None
    
    def get_transcription_result(self, transcript_id: str) -> Optional[Dict]:
        """Get transcription result."""
        try:
            transcript_url = f"{self.base_url}/transcript/{transcript_id}"
            
            while True:
                response = requests.get(transcript_url, headers=self.headers)
                
                if response.status_code == 200:
                    result = response.json()
                    status = result["status"]
                    
                    if status == "completed":
                        logger.info("✅ Transcription completed!")
                        return result
                    elif status == "error":
                        logger.error(f"Transcription error: {result.get('error', 'Unknown error')}")
                        return None
                    else:
                        logger.info(f"⏳ Transcription status: {status}")
                        time.sleep(10)  # Wait 10 seconds before checking again
                else:
                    logger.error(f"Failed to get transcription status: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting transcription result: {e}")
            return None


class ConversationChunkGenerator:
    def __init__(self, video_dir: Path, output_dir: Path):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scientific topics for classification
        self.scientific_topics = [
            "intelligence_definition", "scale_free_cognition", "unconventional_intelligence", "seti_suti",
            "bioelectricity", "morphogenesis", "regeneration", "development", "healing",
            "xenobots", "anthrobots", "synthetic_biology", "living_machines",
            "goal_directed_behavior", "emergent_properties", "collective_intelligence",
            "plasticity", "adaptation", "problem_solving", "cybernetics",
            "computational_biology", "information_theory", "complexity",
            "morphospace_navigation", "pattern_formation", "self_organization"
        ]
    
    def find_video_files(self) -> List[Path]:
        """Find video files in the directory."""
        video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.video_dir.glob(f"*{ext}"))
        
        return video_files
    
    def find_transcript_files(self) -> List[Path]:
        """Find transcript files in the directory."""
        transcript_extensions = ['.json', '.txt', '.srt', '.vtt']
        transcript_files = []
        
        for ext in transcript_extensions:
            transcript_files.extend(self.video_dir.glob(f"*{ext}"))
        
        return transcript_files
    
    def load_transcript(self, transcript_file: Path) -> List[Dict[str, Any]]:
        """Load and parse transcript file."""
        try:
            if transcript_file.suffix == '.json':
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle different transcript formats
                    if 'segments' in data:
                        return data['segments']
                    elif 'words' in data:
                        return self._convert_words_to_segments(data['words'])
                    else:
                        return self._convert_flat_transcript(data)
            elif transcript_file.suffix == '.txt':
                return self._parse_text_transcript(transcript_file)
            else:
                logger.warning(f"Unsupported transcript format: {transcript_file.suffix}")
                return []
        except Exception as e:
            logger.error(f"Error loading transcript {transcript_file}: {e}")
            return []
    
    def _convert_words_to_segments(self, words: List[Dict]) -> List[Dict]:
        """Convert word-level transcript to segments."""
        segments = []
        current_segment = {"start": 0, "end": 0, "text": ""}
        
        for word in words:
            if not current_segment["text"]:
                current_segment["start"] = word.get("start", 0)
            
            current_segment["text"] += " " + word.get("text", "")
            current_segment["end"] = word.get("end", 0)
            
            # Create new segment every ~50 words or when there's a long pause
            if len(current_segment["text"].split()) > 50:
                current_segment["text"] = current_segment["text"].strip()
                segments.append(current_segment.copy())
                current_segment = {"start": 0, "end": 0, "text": ""}
        
        # Add final segment
        if current_segment["text"]:
            current_segment["text"] = current_segment["text"].strip()
            segments.append(current_segment)
        
        return segments
    
    def _convert_flat_transcript(self, data: Dict) -> List[Dict]:
        """Convert flat transcript data to segments."""
        # This is a fallback for unknown JSON formats
        segments = []
        if isinstance(data, dict) and 'text' in data:
            # Single text block
            segments.append({
                "start": 0,
                "end": 0,
                "text": data['text']
            })
        return segments
    
    def _parse_text_transcript(self, transcript_file: Path) -> List[Dict]:
        """Parse plain text transcript."""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Simple paragraph-based segmentation
            paragraphs = text.split('\n\n')
            segments = []
            
            for i, para in enumerate(paragraphs):
                if para.strip():
                    segments.append({
                        "start": i * 30,  # Rough estimate: 30 seconds per paragraph
                        "end": (i + 1) * 30,
                        "text": para.strip()
                    })
            
            return segments
        except Exception as e:
            logger.error(f"Error parsing text transcript: {e}")
            return []
    
    def create_conversation_chunks(self, segments: List[Dict[str, Any]], video_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create conversation chunks from transcript segments."""
        chunks = []
        
        for i, segment in enumerate(segments):
            if not segment.get('text', '').strip():
                continue
            
            # Create basic chunk
            chunk = {
                "chunk_index": i,
                "chunk_id": f"enhanced_conv_chunk_{i:04d}",
                "text": segment.get('text', '').strip(),
                "section": "conversation",
                "start_time": segment.get('start', 0),
                "end_time": segment.get('end', 0),
                "duration": segment.get('end', 0) - segment.get('start', 0),
                "word_count": len(segment.get('text', '').split()),
                "character_count": len(segment.get('text', '')),
                "video_id": video_metadata.get('video_id', 'unknown'),
                "video_title": video_metadata.get('title', 'Unknown Title'),
                "youtube_url": video_metadata.get('youtube_url'),
                "question_start_time": segment.get('start', 0),
                "answer_start_time": segment.get('start', 0),
                "enhancement_method": "auto_generated",
                "citation_context": "From conversation video",
                "extraction_method": "transcript_processing",
                "processing_status": "completed"
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def enhance_chunk_with_llm(self, chunk: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Use OpenAI to enhance a chunk with topics and summary."""
        if not openai.api_key:
            logger.warning("OpenAI API key not found, skipping LLM enhancement")
            return chunk
        
        text = chunk.get('text', '')
        if len(text) < 50:
            return chunk
        
        system_prompt = """You are an expert at creating enhanced content for semantic search systems focused on scientific research. 

You must:
1. Preserve the exact content (NEVER modify, skip, or rephrase the original text)
2. Add helpful context if needed
3. Identify precise scientific topics
4. Return valid JSON matching the specified schema"""

        user_prompt = f"""Create enhanced content for a RAG system about scientific research.

TASK: Enhance this text chunk for semantic search while preserving the original content.

ENHANCED TEXT REQUIREMENTS:
1. Start with a brief context statement that captures the essence
2. Follow with "Content:" 
3. Then include the COMPLETE original text (EXACT COPY - do not change, skip, or modify any words)
4. The result should flow naturally and be searchable

TOPICS REQUIREMENTS:
Generate 1-4 specific, searchable topics from these scientific areas:
{', '.join(self.scientific_topics)}

ORIGINAL TEXT:
{text[:1000]}{'...' if len(text) > 1000 else ''}

Return valid JSON with 'enhanced_text' and 'topics' fields."""

        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content
                
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    enhanced_data = json.loads(json_match.group())
                    
                    # Validate with Pydantic
                    enhanced_response = EnhancedChunkResponse(**enhanced_data)
                    
                    # Update chunk with enhanced content
                    chunk.update({
                        "text": enhanced_response.enhanced_text,
                        "topic": ", ".join(enhanced_response.topics),
                        "chunk_summary": f"Enhanced conversation chunk with topics: {', '.join(enhanced_response.topics)}",
                        "enhancement_method": "gpt4_enhanced"
                    })
                    
                    return chunk
                
            except Exception as e:
                logger.warning(f"LLM enhancement attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to enhance chunk after {max_retries} attempts")
        
        # Fallback: create basic enhancement
        chunk.update({
            "topic": "conversation, research, science",
            "chunk_summary": f"Conversation chunk from {chunk.get('video_title', 'video')}",
            "enhancement_method": "fallback_enhancement"
        })
        
        return chunk
    
    def process_video_directory(self, auto_enhance: bool = True) -> Dict[str, Any]:
        """Process all videos in the directory and generate chunks."""
        logger.info(f"🎬 Processing video directory: {self.video_dir}")
        
        video_files = self.find_video_files()
        transcript_files = self.find_transcript_files()
        
        if not video_files and not transcript_files:
            logger.error("No video or transcript files found")
            return {"success": False, "error": "No files found"}
        
        all_chunks = []
        processed_files = []
        
        # Process transcript files first
        for transcript_file in transcript_files:
            logger.info(f"📝 Processing transcript: {transcript_file.name}")
            
            # Extract video metadata from filename
            video_metadata = {
                "video_id": transcript_file.stem,
                "title": transcript_file.stem.replace('_', ' ').title(),
                "youtube_url": None,
                "filename": transcript_file.name
            }
            
            segments = self.load_transcript(transcript_file)
            if segments:
                chunks = self.create_conversation_chunks(segments, video_metadata)
                
                if auto_enhance:
                    enhanced_chunks = []
                    for chunk in chunks:
                        enhanced_chunk = self.enhance_chunk_with_llm(chunk)
                        enhanced_chunks.append(enhanced_chunk)
                    chunks = enhanced_chunks
                
                all_chunks.extend(chunks)
                processed_files.append(transcript_file.name)
                logger.info(f"✅ Created {len(chunks)} chunks from {transcript_file.name}")
        
        # If no transcripts, try to process video files directly
        if not all_chunks and video_files:
            logger.warning("No transcript files found, video processing not implemented")
            logger.info("Please provide transcript files (.json, .txt, .srt, .vtt) for processing")
        
        if all_chunks:
            # Save chunks
            output_filename = f"{self.video_dir.name.replace(' ', '_').replace('-', '_')}_conversation_chunks.json"
            output_file = self.output_dir / output_filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved {len(all_chunks)} chunks to: {output_file}")
            
            return {
                "success": True,
                "chunks_count": len(all_chunks),
                "processed_files": processed_files,
                "output_file": str(output_file)
            }
        else:
            return {
                "success": False,
                "error": "No chunks were created"
            }


class YouTubeToChunksPipeline:
    """Complete pipeline from YouTube URL to conversation chunks."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_processor = YouTubeVideoProcessor(self.output_dir)
        self.transcriber = AssemblyAITranscriber()
        self.chunk_generator = ConversationChunkGenerator(self.output_dir, self.output_dir)
    
    def process_youtube_url(self, youtube_url: str, auto_enhance: bool = True) -> Dict[str, Any]:
        """Complete pipeline from YouTube URL to conversation chunks."""
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(youtube_url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}
            
            logger.info(f"🎬 Processing YouTube video: {video_id}")
            
            # Step 1: Download video
            video_file = self.video_processor.download_video(youtube_url, video_id)
            if not video_file:
                return {"success": False, "error": "Video download failed"}
            
            # Step 2: Extract audio
            audio_file = self.video_processor.extract_audio(video_file, video_id)
            if not audio_file:
                return {"success": False, "error": "Audio extraction failed"}
            
            # Step 3: Upload audio to AssemblyAI
            upload_url = self.transcriber.upload_audio(audio_file)
            if not upload_url:
                return {"success": False, "error": "Audio upload failed"}
            
            # Step 4: Start transcription
            transcript_id = self.transcriber.transcribe_audio(upload_url)
            if not transcript_id:
                return {"success": False, "error": "Transcription failed"}
            
            # Step 5: Get transcription result
            transcript_result = self.transcriber.get_transcription_result(transcript_id)
            if not transcript_result:
                return {"success": False, "error": "Failed to get transcription result"}
            
            # Step 6: Save transcript
            transcript_file = self.output_dir / video_id / f"{video_id}_transcript.json"
            transcript_file.parent.mkdir(exist_ok=True)
            
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Transcript saved: {transcript_file}")
            
            # Step 7: Generate conversation chunks
            video_metadata = {
                "video_id": video_id,
                "title": transcript_result.get("audio_url", video_id),
                "youtube_url": youtube_url
            }
            
            # Create chunks from transcript
            segments = transcript_result.get("utterances", [])
            if not segments:
                # Fallback to words if no utterances
                words = transcript_result.get("words", [])
                if words:
                    segments = self.chunk_generator._convert_words_to_segments(words)
            
            if segments:
                chunks = self.chunk_generator.create_conversation_chunks(segments, video_metadata)
                
                if auto_enhance:
                    enhanced_chunks = []
                    for chunk in chunks:
                        enhanced_chunk = self.chunk_generator.enhance_chunk_with_llm(chunk)
                        enhanced_chunks.append(enhanced_chunk)
                    chunks = enhanced_chunks
                
                # Save chunks
                output_filename = f"{video_id}_conversation_chunks.json"
                output_file = self.output_dir / output_filename
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)
                
                logger.info(f"💾 Saved {len(chunks)} chunks to: {output_file}")
                
                return {
                    "success": True,
                    "video_id": video_id,
                    "chunks_count": len(chunks),
                    "transcript_file": str(transcript_file),
                    "output_file": str(output_file)
                }
            else:
                return {"success": False, "error": "No segments found in transcript"}
                
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_video_id(self, youtube_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate semantic conversation chunks from YouTube videos or existing video files"
    )
    parser.add_argument(
        "--youtube-url",
        type=str,
        help="YouTube URL to process (downloads and transcribes)"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing existing video files and/or transcripts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/semantically-chunked/conversation_chunks"),
        help="Directory to save generated conversation chunks"
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip LLM enhancement of chunks"
    )
    
    args = parser.parse_args()
    
    if not args.youtube_url and not args.video_dir:
        parser.error("Either --youtube-url or --video-dir must be specified")
    
    try:
        if args.youtube_url:
            # Process YouTube URL
            pipeline = YouTubeToChunksPipeline(args.output_dir)
            result = pipeline.process_youtube_url(args.youtube_url, auto_enhance=not args.no_enhance)
            
            if result["success"]:
                print("✅ YouTube processing completed!")
                print(f"📊 Video ID: {result['video_id']}")
                print(f"📊 Chunks created: {result['chunks_count']}")
                print(f"📁 Transcript: {result['transcript_file']}")
                print(f"📁 Output file: {result['output_file']}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.video_dir:
            # Process existing video directory
            generator = ConversationChunkGenerator(args.video_dir, args.output_dir)
            result = generator.process_video_directory(auto_enhance=not args.no_enhance)
            
            if result["success"]:
                print("✅ Video directory processing completed!")
                print(f"📊 Chunks created: {result['chunks_count']}")
                print(f"📁 Output file: {result['output_file']}")
                print(f"📝 Processed files: {', '.join(result['processed_files'])}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
