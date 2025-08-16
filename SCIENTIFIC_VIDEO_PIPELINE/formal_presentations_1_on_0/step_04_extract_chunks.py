#!/usr/bin/env python3
"""
Step 04: Extract Semantic Chunks
Creates semantically meaningful chunks from transcripts using LLM-based parsing.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chunking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SemanticChunker:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        self.input_dir = Path("step_03_transcription")
        self.output_dir = Path("step_04_extract_chunks")
        self.output_dir.mkdir(exist_ok=True)
        
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
        
        for transcript_file in transcript_files:
            try:
                self.process_single_transcript(transcript_file)
            except Exception as e:
                logger.error(f"Failed to process {transcript_file.name}: {e}")
    
    def process_single_transcript(self, transcript_file: Path):
        """Process a single transcript file"""
        video_id = transcript_file.stem.replace('_transcript', '')
        logger.info(f"Processing transcript: {video_id}")
        
        # Load transcript
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        
        # Extract text and metadata
        transcript_text = transcript_data.get('transcript', '')
        video_metadata = transcript_data.get('video_metadata', {})
        transcript_metadata = transcript_data.get('transcript_metadata', {})
        
        if not transcript_text:
            logger.warning(f"No transcript text found for {video_id}")
            return
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(transcript_text, video_id)
        
        # Enhance chunks with LLM analysis
        enhanced_chunks = self.enhance_chunks_with_llm(chunks, video_id)
        
        # Save chunks
        self.save_chunks(enhanced_chunks, video_id, video_metadata, transcript_metadata)
        
        logger.info(f"Successfully processed transcript: {video_id}")
    
    def create_semantic_chunks(self, text: str, video_id: str) -> List[Dict[str, Any]]:
        """Create semantic chunks optimized for Michael Levin's information-dense style"""
        chunks = []
        
        # Split by sentences first
        sentences = self.split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(sentence.split())  # Rough token estimation
            
            # Check if this sentence would exceed max size
            if current_tokens + sentence_tokens > self.max_chunk_tokens and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'sentence_count': len(current_chunk),
                    'start_sentence': len(chunks),
                    'end_sentence': len(chunks) + len(current_chunk) - 1
                })
                
                # Start new chunk with minimal overlap (just last sentence for continuity)
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_tokens = len(current_chunk[-1].split()) if current_chunk else 0
            
            # Also check for natural topic boundaries (key phrases that suggest new topics)
            if self.is_topic_boundary(sentence) and current_chunk and current_tokens > self.min_chunk_tokens:
                # Force a chunk break at topic boundaries
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                    'text': chunk_text,
                    'token_count': current_tokens,
                    'sentence_count': len(current_chunk),
                    'start_sentence': len(chunks),
                    'end_sentence': len(chunks) + len(current_chunk) - 1
                })
                
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'chunk_id': f"{video_id}_chunk_{len(chunks):03d}",
                'text': chunk_text,
                'token_count': current_tokens,
                'sentence_count': len(current_chunk),
                'start_sentence': len(chunks),
                'end_sentence': len(chunks) + len(current_chunk) - 1
            })
        
        logger.info(f"Created {len(chunks)} optimized chunks for {video_id}")
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Basic sentence splitting - can be enhanced
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
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
        """Enhance chunks using LLM analysis"""
        enhanced_chunks = []
        
        for chunk in chunks:
            try:
                enhanced_chunk = self.enhance_single_chunk(chunk, video_id)
                enhanced_chunks.append(enhanced_chunk)
            except Exception as e:
                logger.warning(f"Failed to enhance chunk {chunk['chunk_id']}: {e}")
                # Use basic chunk if enhancement fails
                enhanced_chunks.append(chunk)
        
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
            
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")

def main():
    """Main execution function"""
    try:
        chunker = SemanticChunker()
        chunker.process_all_transcripts()
        logger.info("Chunking step completed successfully")
    except Exception as e:
        logger.error(f"Chunking step failed: {e}")
        raise

if __name__ == "__main__":
    main()
