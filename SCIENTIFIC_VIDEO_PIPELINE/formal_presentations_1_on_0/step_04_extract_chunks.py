#!/usr/bin/env python3
"""
Step 04: Extract Semantic Chunks
Creates semantically meaningful chunks from transcripts using LLM-based parsing.
"""

import os
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import openai
from pipeline_progress_queue import get_progress_queue

# Load environment variables
load_dotenv()

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('chunking')

class SemanticChunker:
    def __init__(self, progress_queue=None):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        self.input_dir = Path("step_03_transcription")
        self.output_dir = Path("step_04_extract_chunks")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = progress_queue
        
        # Chunking parameters - Optimized for Michael Levin's information-dense style
        self.max_chunk_tokens = 300   # Much smaller chunks for granular concepts
        self.min_chunk_tokens = 100   # Lower minimum for focused topics
        self.overlap_tokens = 50      # Reduced overlap for cleaner separation
    
    def process_all_transcripts(self):
        """Process all transcripts in the input directory"""
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find all transcript files
        transcript_files = list(self.input_dir.glob("*_transcript.json"))
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        # Track processing statistics
        total_transcripts = len(transcript_files)
        new_chunks = 0
        existing_chunks = 0
        failed_chunks = 0
        
        for transcript_file in transcript_files:
            try:
                result = self.process_single_transcript(transcript_file)
                if result == 'new':
                    new_chunks += 1
                elif result == 'existing':
                    existing_chunks += 1
                elif result == 'failed':
                    failed_chunks += 1
            except Exception as e:
                logger.error(f"Failed to process {transcript_file.name}: {e}")
                failed_chunks += 1
        
        # Log summary
        logger.info(f"Chunking Summary:")
        logger.info(f"  Total transcripts: {total_transcripts}")
        logger.info(f"  New chunks: {new_chunks}")
        logger.info(f"  Existing chunks: {existing_chunks}")
        logger.info(f"  Failed: {failed_chunks}")
        if total_transcripts > 0:
            success_rate = ((new_chunks + existing_chunks) / total_transcripts * 100)
            logger.info(f"  Success rate: {success_rate:.1f}%")
        else:
            logger.info(f"  Success rate: N/A (no transcripts to process)")
    
    def process_single_transcript(self, transcript_file: Path):
        """Process a single transcript file"""
        video_id = transcript_file.stem.replace('_transcript', '')
        logger.info(f"Processing transcript: {video_id}")
        
        # Check if chunks already exist using progress queue
        if self.progress_queue:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_04_semantic_chunking') == 'completed':
                logger.info(f"Chunks already completed for {video_id} (progress queue), skipping")
                return 'existing'
        
        # Fallback: Check if chunks file exists (for backward compatibility)
        chunks_file = self.output_dir / f"{video_id}_chunks.json"
        if chunks_file.exists():
            logger.info(f"Chunks already exist for {video_id} (file check), skipping")
            return 'existing'
        
        # Load transcript
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        
        # Extract text and metadata
        transcript_text = transcript_data.get('text', '')  # AssemblyAI uses 'text' field
        video_metadata = transcript_data.get('video_metadata', {})
        transcript_metadata = transcript_data.get('transcript_metadata', {})
        utterances = transcript_data.get('words', [])  # AssemblyAI uses 'words' instead of 'utterances'
        
        if not transcript_text:
            logger.warning(f"No transcript text found for {video_id}")
            return 'failed'
        
        # Create semantic chunks with timestamp information
        chunks = self.create_semantic_chunks_with_timestamps(transcript_text, utterances, video_id)
        
        # Enhance chunks with LLM analysis
        enhanced_chunks = self.enhance_chunks_with_llm(chunks, video_id)
        
        # Save chunks
        self.save_chunks(enhanced_chunks, video_id, video_metadata, transcript_metadata)
        
        logger.info(f"Successfully processed transcript: {video_id}")
        return 'new'
    
    def create_semantic_chunks_with_timestamps(self, text: str, utterances: List[Dict], video_id: str) -> List[Dict[str, Any]]:
        """Create semantic chunks with timestamp information from AssemblyAI utterances"""
        chunks = []
        
        # Create a mapping from sentence text to timestamp ranges
        # This now builds sentences directly from word-level data
        sentence_timestamps = self.map_sentences_to_timestamps([], utterances)
        
        # Get sentences from the timestamp mapping (they're built from words)
        sentences = list(sentence_timestamps.keys())
        
        current_chunk = []
        current_tokens = 0
        current_start_time = None
        current_end_time = None
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(sentence.split())  # Rough token estimation
            sentence_timing = sentence_timestamps.get(sentence, {})
            
            # Check if this sentence would exceed max size
            if current_tokens + sentence_tokens > self.max_chunk_tokens and current_chunk:
                # Save current chunk with timing
                chunk_text = ' '.join(current_chunk)
                # Validate timing before creating chunk
                if current_start_time is not None and current_end_time is not None and current_end_time > current_start_time:
                    chunks.append({
                        'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                        'text': chunk_text,
                        'token_count': current_tokens,
                        'sentence_count': len(current_chunk),
                        'start_sentence': len(chunks),
                        'end_sentence': len(chunks) + len(current_chunk) - 1,
                        'start_time_ms': current_start_time,
                        'end_time_ms': current_end_time,
                        'start_time_seconds': current_start_time / 1000.0,
                        'end_time_seconds': current_end_time / 1000.0
                    })
                else:
                    logger.warning(f"Skipping chunk with invalid timing: start={current_start_time}, end={current_end_time}")
                    # Create chunk without timing
                    chunks.append({
                        'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                        'text': chunk_text,
                        'token_count': current_tokens,
                        'sentence_count': len(current_chunk),
                        'start_sentence': len(chunks),
                        'end_sentence': len(chunks) + len(current_chunk) - 1,
                        'start_time_ms': None,
                        'end_time_ms': None,
                        'start_time_seconds': None,
                        'end_time_seconds': None
                    })
                
                # Start new chunk with minimal overlap (just last sentence for continuity)
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_tokens = len(current_chunk[-1].split()) if current_chunk else 0
                if current_chunk:
                    last_sentence = current_chunk[-1]
                    last_timing = sentence_timestamps.get(last_sentence, {})
                    current_start_time = last_timing.get('start', None)
                    current_end_time = last_timing.get('end', None)
            
            # Also check for natural topic boundaries (key phrases that suggest new topics)
            if self.is_topic_boundary(sentence) and current_chunk and current_tokens > self.min_chunk_tokens:
                # Force a chunk break at topic boundaries
                chunk_text = ' '.join(current_chunk)
                # Validate timing before creating chunk
                if current_start_time is not None and current_end_time is not None and current_end_time > current_start_time:
                    chunks.append({
                        'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                        'text': chunk_text,
                        'token_count': current_tokens,
                        'sentence_count': len(current_chunk),
                        'start_sentence': len(chunks),
                        'end_sentence': len(chunks) + len(current_chunk) - 1,
                        'start_time_ms': current_start_time,
                        'end_time_ms': current_end_time,
                        'start_time_seconds': current_start_time / 1000.0,
                        'end_time_seconds': current_end_time / 1000.0
                    })
                else:
                    logger.warning(f"Skipping chunk with invalid timing: start={current_start_time}, end={current_end_time}")
                    # Create chunk without timing
                    chunks.append({
                        'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                        'text': chunk_text,
                        'token_count': current_tokens,
                        'sentence_count': len(current_chunk),
                        'start_sentence': len(chunks),
                        'end_sentence': len(chunks) + len(current_chunk) - 1,
                        'start_time_ms': None,
                        'end_time_ms': None,
                        'start_time_seconds': None,
                        'end_time_seconds': None
                    })
                
                current_chunk = []
                current_tokens = 0
                current_start_time = None
                current_end_time = None
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            
            # Update timing for the chunk
            if current_start_time is None and sentence_timing.get('start'):
                current_start_time = sentence_timing.get('start')
            if sentence_timing.get('end'):
                current_end_time = sentence_timing.get('end')
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            # Validate timing before creating chunk
            if current_start_time is not None and current_end_time is not None and current_end_time > current_start_time:
                chunks.append({
                    'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'sentence_count': len(current_chunk),
                    'start_sentence': len(chunks),
                    'end_sentence': len(chunks) + len(current_chunk) - 1,
                    'start_time_ms': current_start_time,
                    'end_time_ms': current_end_time,
                    'start_time_seconds': current_start_time / 1000.0,
                    'end_time_seconds': current_end_time / 1000.0
                })
            else:
                logger.warning(f"Skipping final chunk with invalid timing: start={current_start_time}, end={current_end_time}")
                # Create chunk without timing
                chunks.append({
                    'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'sentence_count': len(current_chunk),
                    'start_sentence': len(chunks),
                    'end_sentence': len(chunks) + len(current_chunk) - 1,
                    'start_time_ms': None,
                    'end_time_ms': None,
                    'start_time_seconds': None,
                    'end_time_seconds': None
                })
        
        logger.info(f"Created {len(chunks)} timestamped chunks for {video_id}")
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Basic sentence splitting - can be enhanced
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def map_sentences_to_timestamps(self, sentences: List[str], utterances: List[Dict]) -> Dict[str, Dict[str, int]]:
        """Map sentence text to their start/end timestamps using AssemblyAI word-level data"""
        sentence_timestamps = {}
        
        # Get words directly from the top-level 'words' field (AssemblyAI structure)
        # The 'utterances' parameter is actually the words list from the transcript
        all_words = utterances  # utterances is actually the words list
        
        if not all_words:
            logger.warning("No word-level timing data found")
            return sentence_timestamps
        
        # Sort words by start time
        all_words.sort(key=lambda x: x.get('start', 0))
        
        # Instead of trying to find sentences in text, build sentences from words
        # and assign timestamps directly
        current_sentence = ""
        current_sentence_words = []
        sentence_count = 0
        
        for word_data in all_words:
            word_text = word_data.get('text', '').strip()
            if not word_text:
                continue
            
            # Add word to current sentence
            current_sentence += word_text + " "
            current_sentence_words.append(word_data)
            
            # Check if we've reached a sentence boundary
            # Look for sentence-ending punctuation or natural breaks
            if (word_text.endswith('.') or word_text.endswith('!') or word_text.endswith('?') or
                word_text.endswith('...') or len(current_sentence.strip()) > 200):
                
                # Finalize current sentence
                sentence_text = current_sentence.strip()
                
                if sentence_text and current_sentence_words:
                    # Get timing from first and last word
                    sentence_start = current_sentence_words[0]['start']
                    sentence_end = current_sentence_words[-1]['end']
                    
                    sentence_timestamps[sentence_text] = {
                        'start': sentence_start,
                        'end': sentence_end
                    }
                    
                    logger.debug(f"Sentence {sentence_count}: {sentence_start}ms - {sentence_end}ms ({len(current_sentence_words)} words)")
                    sentence_count += 1
                
                # Reset for next sentence
                current_sentence = ""
                current_sentence_words = []
        
        # Handle the last sentence if it exists
        if current_sentence.strip() and current_sentence_words:
            sentence_text = current_sentence.strip()
            sentence_start = current_sentence_words[0]['start']
            sentence_end = current_sentence_words[-1]['end']
            
            sentence_timestamps[sentence_text] = {
                'start': sentence_start,
                'end': sentence_end
            }
            
            logger.debug(f"Final sentence: {sentence_start}ms - {sentence_end}ms ({len(current_sentence_words)} words)")
            sentence_count += 1
        
        logger.info(f"Mapped {len(sentence_timestamps)} sentences to timestamps")
        if sentence_timestamps:
            # Show a few examples
            sample_sentences = list(sentence_timestamps.items())[:3]
            for sentence, timing in sample_sentences:
                logger.info(f"Sample mapping: '{sentence[:50]}...' -> {timing['start']}ms - {timing['end']}ms")
        return sentence_timestamps
    
    def is_topic_boundary(self, sentence: str) -> bool:
        """Detect if a sentence suggests a new topic or concept"""
        # Key phrases that often indicate topic shifts in Michael Levin's talks
        topic_indicators = [
            'so', 'now', 'but', 'however', 'meanwhile', 'on the other hand',
            'first', 'second', 'third', 'finally', 'in conclusion',
            'let me', 'let me show you', 'let me give you',
            'what I want to', 'what I\'m going to', 'what I do',
            'the key point is', 'the important thing is', 'the crucial thing is',
            'this is', 'this means', 'this suggests',
            'for example', 'as an example', 'consider',
            'in other words', 'that is to say', 'specifically',
            'the question is', 'the problem is', 'the challenge is',
            'we can see', 'we observe', 'we notice',
            'this leads to', 'this results in', 'this causes',
            'the reason is', 'the explanation is', 'the answer is'
        ]
        
        sentence_lower = sentence.lower().strip()
        
        # Check if sentence starts with topic indicators
        for indicator in topic_indicators:
            if sentence_lower.startswith(indicator):
                return True
        
        # Check if sentence contains strong topic transition words
        transition_words = ['therefore', 'thus', 'consequently', 'as a result', 'in summary']
        for word in transition_words:
            if word in sentence_lower:
                return True
        
        return False
    
    def enhance_chunks_with_llm(self, chunks: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
        """Enhance chunks using LLM analysis with rate limiting"""
        enhanced_chunks = []
        total_chunks = len(chunks)
        
        logger.info(f"Starting LLM enhancement of {total_chunks} chunks...")
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Enhancing chunk {i+1}/{total_chunks}: {chunk['chunk_id']}")
                enhanced_chunk = self.enhance_single_chunk(chunk, video_id)
                enhanced_chunks.append(enhanced_chunk)
                
                # Add delay between requests to respect rate limits
                if i < total_chunks - 1:  # Don't delay after the last chunk
                    delay_seconds = 1.0  # 1 second delay between requests
                    logger.info(f"Waiting {delay_seconds}s before next request...")
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                logger.warning(f"Failed to enhance chunk {chunk['chunk_id']}: {e}")
                # Use basic chunk if enhancement fails
                enhanced_chunks.append(chunk)
        
        logger.info(f"LLM enhancement completed: {len(enhanced_chunks)} chunks processed")
        return enhanced_chunks
    
    def enhance_single_chunk(self, chunk: Dict[str, Any], video_id: str) -> Dict[str, Any]:
        """Enhance a single chunk with LLM analysis"""
        try:
            # Create prompt for LLM analysis
            prompt = f"""
            Analyze this transcript chunk from a scientific presentation by Michael Levin and provide:

            1. **Primary Topics** (3-5 main scientific concepts)
            2. **Secondary Topics** (5-8 related concepts)
            3. **Key Terms** (important scientific terminology)
            4. **Content Summary** (2-3 sentence summary)
            5. **Scientific Domain** (e.g., developmental biology, bioelectricity, etc.)

            Transcript chunk:
            {chunk['text'][:2000]}...

            Respond in JSON format:
            {{
                "primary_topics": ["topic1", "topic2"],
                "secondary_topics": ["topic1", "topic2"],
                "key_terms": ["term1", "term2"],
                "content_summary": "Brief summary",
                "scientific_domain": "Domain name"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a scientific content analyst specializing in developmental biology and bioelectricity research."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse LLM response
            llm_content = response.choices[0].message.content
            enhanced_data = self.parse_llm_response(llm_content)
            
            # Merge with original chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk.update(enhanced_data)
            enhanced_chunk['enhanced'] = True
            
            return enhanced_chunk
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed for chunk {chunk['chunk_id']}: {e}")
            # Return basic chunk if enhancement fails
            return chunk
    
    def parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return {
                    'primary_topics': data.get('primary_topics', []),
                    'secondary_topics': data.get('secondary_topics', []),
                    'key_terms': data.get('key_terms', []),
                    'content_summary': data.get('content_summary', ''),
                    'scientific_domain': data.get('scientific_domain', '')
                }
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        # Fallback to basic extraction
        return {
            'primary_topics': [],
            'secondary_topics': [],
            'key_terms': [],
            'content_summary': '',
            'scientific_domain': ''
        }
    
    def save_chunks(self, chunks: List[Dict[str, Any]], video_id: str, 
                   video_metadata: Dict[str, Any], transcript_metadata: Dict[str, Any]):
        """Save enhanced chunks to file"""
        try:
            # Create chunks summary
            chunks_summary = {
                'video_id': video_id,
                'video_metadata': video_metadata,
                'transcript_metadata': transcript_metadata,
                'chunking_parameters': {
                    'max_chunk_tokens': self.max_chunk_tokens,
                    'min_chunk_tokens': self.min_chunk_tokens,
                    'overlap_tokens': self.overlap_tokens
                },
                'total_chunks': len(chunks),
                'total_tokens': sum(chunk.get('token_count', 0) for chunk in chunks),
                'chunks': chunks,
                'processing_timestamp': transcript_metadata.get('created_at', '')
            }
            
            # Save to file
            output_file = self.output_dir / f"{video_id}_chunks.json"
            with open(output_file, 'w') as f:
                json.dump(chunks_summary, f, indent=2)
            
            logger.info(f"Chunks saved: {output_file}")
            
            # Also save individual chunk files for easier processing
            chunks_dir = self.output_dir / f"{video_id}_chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            for chunk in chunks:
                chunk_file = chunks_dir / f"{chunk['chunk_id']}.json"
                with open(chunk_file, 'w') as f:
                    json.dump(chunk, f, indent=2)
            
            logger.info(f"Individual chunk files saved in: {chunks_dir}")
            
            # Update progress queue
            if self.progress_queue:
                self.progress_queue.update_video_step_status(
                    video_id,
                    'step_04_semantic_chunking',
                    'completed',
                    metadata={
                        'chunks_file': str(output_file),
                        'chunks_dir': str(chunks_dir),
                        'total_chunks': len(chunks),
                        'total_tokens': sum(chunk.get('token_count', 0) for chunk in chunks),
                        'completed_at': datetime.now().isoformat()
                    }
                )
                logger.info(f"📊 Progress queue updated: step 4 completed for {video_id}")
            
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")

def main():
    """Main execution function"""
    try:
        # Initialize progress queue
        progress_queue = get_progress_queue()
        logger.info("✅ Progress queue initialized")
        
        chunker = SemanticChunker(progress_queue)
        chunker.process_all_transcripts()
        logger.info("Chunking step completed successfully")
    except Exception as e:
        logger.error(f"Chunking step failed: {e}")
        raise

if __name__ == "__main__":
    main()
