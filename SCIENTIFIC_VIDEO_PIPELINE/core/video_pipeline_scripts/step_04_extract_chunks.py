#!/usr/bin/env python3
"""
Step 04: Extract Semantic Chunks (Unified)
Creates semantically meaningful chunks from transcripts using LLM-based parsing.
Supports both formal presentations and multi-speaker conversations.
"""

import os
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class UnifiedChunkExtractor:
    """Unified chunk extractor supporting multiple pipeline types"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the unified chunk extractor
        
        Args:
            pipeline_config: Pipeline-specific configuration
            progress_queue: Progress tracking queue
        """
        self.config = pipeline_config
        self.progress_queue = progress_queue
        
        # Pipeline metadata
        self.pipeline_type = pipeline_config.get('pipeline_type', 'unknown')
        self.speaker_count = pipeline_config.get('speaker_count', 1)
        self.chunking_config = pipeline_config.get('chunking_config', {})
        self.llm_config = pipeline_config.get('llm_config', {})
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Initialize OpenAI client if LLM enhancement is enabled
        self.openai_client = None
        if pipeline_config.get('llm_enhancement', False):
            self._initialize_openai_client()
        
        # Input/output directories
        self.input_dir = Path("step_03_transcription")
        self.output_dir = Path("step_04_extract_chunks")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "finetune_data").mkdir(exist_ok=True)
        (self.output_dir / "enhanced_chunks").mkdir(exist_ok=True)
        
        # Speaker mappings for multi-speaker pipelines
        self.speaker_mappings = {}
        self.speaker_mappings_file = self.output_dir / "speaker_mappings.json"
        self.load_speaker_mappings()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'chunks_created': 0,
            'llm_enhanced': 0,
            'qa_pairs_generated': 0
        }
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client for LLM enhancement"""
        try:
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
            return setup_logging(f'{self.pipeline_type}_chunking')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_chunking')
    
    def load_speaker_mappings(self):
        """Load existing speaker mappings"""
        if self.speaker_mappings_file.exists():
            try:
                with open(self.speaker_mappings_file, 'r') as f:
                    self.speaker_mappings = json.load(f)
                self.logger.info(f"Loaded speaker mappings for {len(self.speaker_mappings)} videos")
            except Exception as e:
                self.logger.warning(f"Failed to load speaker mappings: {e}")
                self.speaker_mappings = {}
    
    def save_speaker_mappings(self):
        """Save speaker mappings to file"""
        try:
            with open(self.speaker_mappings_file, 'w') as f:
                json.dump(self.speaker_mappings, f, indent=2)
            self.logger.info("Speaker mappings saved")
        except Exception as e:
            self.logger.error(f"Failed to save speaker mappings: {e}")
    
    def process_all_transcripts(self):
        """Process all transcripts in the input directory"""
        if not self.input_dir.exists():
            self.logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find all transcript files
        transcript_files = list(self.input_dir.glob("*_transcript.json"))
        self.logger.info(f"Found {len(transcript_files)} transcript files")
        
        for transcript_file in transcript_files:
            try:
                self.process_single_transcript(transcript_file)
            except Exception as e:
                self.logger.error(f"Failed to process {transcript_file.name}: {e}")
                self.stats['failed'] += 1
        
        self.log_processing_summary()
    
    def process_single_transcript(self, transcript_file: Path):
        """Process a single transcript file"""
        video_id = transcript_file.stem.replace('_transcript', '')
        self.logger.info(f"Processing transcript: {video_id}")
        
        # Check progress
        if self.progress_queue and self.progress_queue.get_video_status(video_id, 'step_04_extract_chunks') == 'completed':
            self.logger.info(f"Chunks already completed for {video_id}, skipping")
            self.stats['skipped'] += 1
            return
        
        try:
            # Load transcript data
            with open(transcript_file, 'r') as f:
                transcript_data = json.load(f)
            
            # Process based on pipeline type
            if self.pipeline_type == "formal_presentations":
                result = self._process_formal_presentation(transcript_data, video_id)
            elif self.pipeline_type.startswith("conversations"):
                result = self._process_conversation(transcript_data, video_id)
            else:
                self.logger.error(f"Unknown pipeline type: {self.pipeline_type}")
                return
            
            if result:
                self.stats['successful'] += 1
                self.stats['chunks_created'] += len(result.get('chunks', []))
                
                # Update progress
                if self.progress_queue:
                    self.progress_queue.update_video_step(video_id, 'step_04_extract_chunks', 'completed', {
                        'chunks_created': len(result.get('chunks', [])),
                        'qa_pairs': len(result.get('qa_pairs', [])),
                        'enhanced': result.get('enhanced', False)
                    })
                
                self.logger.info(f"✅ Successfully processed {video_id}")
            else:
                self.stats['failed'] += 1
                self.logger.error(f"❌ Failed to process {video_id}")
                
        except Exception as e:
            self.logger.error(f"Error processing {transcript_file.name}: {e}")
            self.stats['failed'] += 1
    
    def _process_formal_presentation(self, transcript_data: Dict, video_id: str) -> Dict[str, Any]:
        """Process formal presentation transcript (single speaker)"""
        # Extract utterances
        utterances = transcript_data.get('utterances', [])
        if not utterances:
            self.logger.warning(f"No utterances found in {video_id}")
            return None
        
        # Create semantic chunks
        chunks = self._create_semantic_chunks(utterances, video_id, is_formal=True)
        
        # Enhance chunks with LLM if enabled
        if self.openai_client and self.config.get('llm_enhancement', False):
            chunks = self._enhance_chunks_with_llm(chunks, video_id)
        
        # Save outputs
        self._save_outputs(video_id, {
            'chunks': chunks,
            'qa_pairs': [],
            'enhanced': bool(self.openai_client)
        })
        
        return {'chunks': chunks, 'enhanced': bool(self.openai_client)}
    
    def _process_conversation(self, transcript_data: Dict, video_id: str) -> Dict[str, Any]:
        """Process conversation transcript (multiple speakers)"""
        # Extract speaker information
        speakers = list(set(item.get('speaker', '') for item in transcript_data.get('utterances', [])))
        speakers = [s for s in speakers if s]
        
        if len(speakers) < 2:
            self.logger.warning(f"Video {video_id} has {len(speakers)} speakers, need at least 2")
            return None
        
        # Check if speakers are already identified
        if video_id not in self.speaker_mappings:
            self.logger.info(f"Identifying speakers for video {video_id}")
            speaker_info = self._identify_speakers_interactive(video_id, speakers)
            self.speaker_mappings[video_id] = speaker_info
            self.save_speaker_mappings()
        else:
            speaker_info = self.speaker_mappings[video_id]
            self.logger.info(f"Using existing speaker mappings for video {video_id}")
        
        # Create semantic chunks with speaker context
        chunks = self._create_conversation_chunks(transcript_data, speaker_info, video_id)
        
        # Generate Q&A pairs
        qa_pairs = self._generate_qa_pairs(transcript_data, speaker_info, video_id)
        
        # Enhance chunks with LLM if enabled
        if self.openai_client and self.config.get('llm_enhancement', False):
            chunks = self._enhance_chunks_with_llm(chunks, video_id)
        
        # Save outputs
        self._save_outputs(video_id, {
            'chunks': chunks,
            'qa_pairs': qa_pairs,
            'enhanced': bool(self.openai_client)
        })
        
        return {'chunks': chunks, 'qa_pairs': qa_pairs, 'enhanced': bool(self.openai_client)}
    
    def _create_semantic_chunks(self, utterances: List[Dict], video_id: str, is_formal: bool = False) -> List[Dict]:
        """Create semantic chunks from utterances"""
        chunks = []
        current_chunk = []
        chunk_id = 0
        
        for utterance in utterances:
            text = utterance.get('text', '')
            start = utterance.get('start', 0)
            end = utterance.get('end', 0)
            
            if not text:
                continue
            
            # Create chunk entry
            chunk_entry = {
                "text": text,
                "start_time": start,
                "end_time": end,
                "timestamp": f"{start:.2f}s - {end:.2f}s"
            }
            
            current_chunk.append(chunk_entry)
            
            # Create chunk when we have enough content
            if len(' '.join([c['text'] for c in current_chunk])) > self.chunking_config.get('max_chunk_tokens', 300):
                chunk_data = self._create_chunk_data(current_chunk, chunk_id, video_id, is_formal=is_formal)
                chunks.append(chunk_data)
                current_chunk = []
                chunk_id += 1
        
        # Add remaining content as final chunk
        if current_chunk:
            chunk_data = self._create_chunk_data(current_chunk, chunk_id, video_id, is_formal=is_formal)
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_conversation_chunks(self, transcript_data: Dict, speaker_info: Dict, video_id: str) -> List[Dict]:
        """Create conversation chunks with speaker context"""
        chunks = []
        utterances = transcript_data.get('utterances', [])
        
        # Group utterances by speaker and create chunks
        current_chunk = []
        chunk_id = 0
        
        for utterance in utterances:
            speaker_id = utterance.get('speaker', '')
            text = utterance.get('text', '')
            start = utterance.get('start', 0)
            end = utterance.get('end', 0)
            
            if not speaker_id or not text:
                continue
            
            # Get speaker context
            speaker_context = speaker_info.get(speaker_id, {})
            
            # Create chunk entry
            chunk_entry = {
                "speaker_id": speaker_id,
                "speaker_name": speaker_context.get('name', 'Unknown'),
                "speaker_role": speaker_context.get('role', 'Unknown'),
                "is_levin": speaker_context.get('is_levin', False),
                "speaker_order": speaker_context.get('speaker_order', 0),
                "text": text,
                "start_time": start,
                "end_time": end,
                "timestamp": f"{start:.2f}s - {end:.2f}s"
            }
            
            current_chunk.append(chunk_entry)
            
            # Create chunk when we have enough content or speaker changes
            if len(current_chunk) >= 3 or (len(current_chunk) > 0 and len(' '.join([c['text'] for c in current_chunk])) > 500):
                chunk_data = self._create_chunk_data(current_chunk, chunk_id, video_id, speaker_info=speaker_info)
                chunks.append(chunk_data)
                current_chunk = []
                chunk_id += 1
        
        # Add remaining content as final chunk
        if current_chunk:
            chunk_data = self._create_chunk_data(current_chunk, chunk_id, video_id, speaker_info=speaker_info)
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk_data(self, chunk_entries: List[Dict], chunk_id: int, video_id: str, 
                          is_formal: bool = False, speaker_info: Dict = None) -> Dict[str, Any]:
        """Create chunk data structure"""
        # Combine text from all entries
        combined_text = ' '.join([entry['text'] for entry in chunk_entries])
        
        # Create base chunk data
        chunk_data = {
            "chunk_id": f"{video_id}_chunk_{chunk_id:03d}",
            "video_id": video_id,
            "semantic_text": combined_text,
            "start_time": min(entry['start_time'] for entry in chunk_entries),
            "end_time": max(entry['start_time'] for entry in chunk_entries),
            "timestamp": f"{min(entry['start_time'] for entry in chunk_entries):.2f}s - {max(entry['start_time'] for entry in chunk_entries):.2f}s",
            "pipeline_type": self.pipeline_type,
            "created_at": datetime.now().isoformat()
        }
        
        # Add speaker-specific data for conversations
        if not is_formal and speaker_info:
            speakers_in_chunk = list(set(entry['speaker_id'] for entry in chunk_entries))
            chunk_data["conversation_context"] = {
                "speaker_context": {
                    "speakers_involved": speakers_in_chunk,
                    "speaker_names": [speaker_info.get(s, {}).get('name', s) for s in speakers_in_chunk],
                    "levin_present": any(entry.get('is_levin', False) for entry in chunk_entries)
                },
                "temporal_context": {
                    "duration": chunk_data["end_time"] - chunk_data["start_time"],
                    "utterance_count": len(chunk_entries)
                }
            }
        
        # Initialize topic fields (will be populated by LLM enhancement)
        chunk_data.update({
            "primary_topics": [],
            "secondary_topics": [],
            "key_terms": [],
            "content_summary": "",
            "scientific_domain": ""
        })
        
        return chunk_data
    
    def _enhance_chunks_with_llm(self, chunks: List[Dict], video_id: str) -> List[Dict]:
        """Enhance chunks using LLM analysis"""
        if not self.openai_client:
            return chunks
        
        enhanced_chunks = []
        total_chunks = len(chunks)
        
        self.logger.info(f"Starting LLM enhancement of {total_chunks} chunks...")
        
        for i, chunk in enumerate(chunks):
            try:
                self.logger.info(f"Enhancing chunk {i+1}/{total_chunks}: {chunk['chunk_id']}")
                enhanced_chunk = self._enhance_single_chunk(chunk, video_id)
                enhanced_chunks.append(enhanced_chunk)
                
                # Add delay between requests to respect rate limits
                if i < total_chunks - 1:
                    delay_seconds = self.config.get('processing_config', {}).get('llm_delay_seconds', 1.0)
                    self.logger.info(f"Waiting {delay_seconds}s before next request...")
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                self.logger.warning(f"Failed to enhance chunk {chunk['chunk_id']}: {e}")
                enhanced_chunks.append(chunk)
        
        self.logger.info(f"LLM enhancement completed: {len(enhanced_chunks)} chunks processed")
        self.stats['llm_enhanced'] += len(enhanced_chunks)
        return enhanced_chunks
    
    def _enhance_single_chunk(self, chunk: Dict[str, Any], video_id: str) -> Dict[str, Any]:
        """Enhance a single chunk with LLM analysis"""
        try:
            # Create prompt for LLM analysis
            prompt = f"""
            Analyze this transcript chunk from a scientific conversation and provide:

            1. **Primary Topics** (3-5 main scientific concepts)
            2. **Secondary Topics** (5-8 related concepts)
            3. **Key Terms** (important scientific terminology)
            4. **Content Summary** (2-3 sentence summary)
            5. **Scientific Domain** (e.g., developmental biology, bioelectricity, psychiatry, neuroscience, etc.)

            Transcript chunk:
            {chunk['semantic_text'][:2000]}...

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
                model=self.llm_config.get('model', 'gpt-4'),
                messages=[
                    {"role": "system", "content": self.llm_config.get('system_prompt', 'You are a scientific content analyst.')},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config.get('temperature', 0.3)
            )
            
            # Parse LLM response
            llm_content = response.choices[0].message.content
            enhanced_data = self._parse_llm_response(llm_content)
            
            # Merge with original chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk.update(enhanced_data)
            enhanced_chunk['enhanced'] = True
            
            return enhanced_chunk
            
        except Exception as e:
            self.logger.warning(f"LLM enhancement failed for chunk {chunk['chunk_id']}: {e}")
            return chunk
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
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
            self.logger.warning(f"Failed to parse LLM response: {e}")
        
        # Fallback to empty fields
        return {
            'primary_topics': [],
            'secondary_topics': [],
            'key_terms': [],
            'content_summary': '',
            'scientific_domain': ''
        }
    
    def _identify_speakers_interactive(self, video_id: str, speakers: List[str]) -> Dict[str, Dict]:
        """Identify speakers interactively (placeholder for now)"""
        # This would be implemented with interactive user input
        # For now, return a default mapping
        speaker_info = {}
        for i, speaker_id in enumerate(speakers):
            if i == 0:
                speaker_info[speaker_id] = {
                    'name': 'Michael Levin',
                    'role': 'michael_levin',
                    'is_levin': True,
                    'speaker_order': 0
                }
            else:
                speaker_info[speaker_id] = {
                    'name': f'Speaker {i}',
                    'role': f'speaker_{i}',
                    'is_levin': False,
                    'speaker_order': i
                }
        
        return speaker_info
    
    def _generate_qa_pairs(self, transcript_data: Dict, speaker_info: Dict, video_id: str) -> List[Dict]:
        """Generate Q&A pairs from conversation (placeholder)"""
        # This would implement Q&A generation logic
        # For now, return empty list
        return []
    
    def _save_outputs(self, video_id: str, processed_data: Dict):
        """Save all outputs for a video"""
        try:
            # Save chunks
            chunks_file = self.output_dir / f"{video_id}_chunks.json"
            with open(chunks_file, 'w') as f:
                json.dump(processed_data['chunks'], f, indent=2)
            
            # Save enhanced chunks if LLM enhancement was used
            if processed_data.get('enhanced', False):
                enhanced_file = self.output_dir / "enhanced_chunks" / f"{video_id}_enhanced_rag_chunks.json"
                with open(enhanced_file, 'w') as f:
                    json.dump(processed_data['chunks'], f, indent=2)
                self.logger.info(f"Enhanced RAG chunks saved: {enhanced_file}")
            
            # Save Q&A pairs if they exist
            if processed_data.get('qa_pairs'):
                qa_file = self.output_dir / f"{video_id}_qa_pairs.json"
                with open(qa_file, 'w') as f:
                    json.dump(processed_data['qa_pairs'], f, indent=2)
                self.stats['qa_pairs_generated'] += len(processed_data['qa_pairs'])
            
            self.logger.info(f"All outputs saved for video {video_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving outputs for video {video_id}: {e}")
            raise
    
    def log_processing_summary(self):
        """Log a summary of processing statistics"""
        self.logger.info(f"🎉 {self.pipeline_type.upper()} Chunk Extraction Summary:")
        self.logger.info(f"  Total processed: {self.stats['total_processed']}")
        self.logger.info(f"  Successful: {self.stats['successful']}")
        self.logger.info(f"  Failed: {self.stats['failed']}")
        self.logger.info(f"  Skipped: {self.stats['skipped']}")
        self.logger.info(f"  Chunks created: {self.stats['chunks_created']}")
        self.logger.info(f"  LLM enhanced: {self.stats['llm_enhanced']}")
        self.logger.info(f"  Q&A pairs: {self.stats['qa_pairs_generated']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
