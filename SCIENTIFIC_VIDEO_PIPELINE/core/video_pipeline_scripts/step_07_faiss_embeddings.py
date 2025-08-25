#!/usr/bin/env python3
"""
Step 07: FAISS Embeddings (Unified)
Generates FAISS embeddings for RAG-ready content with support for LLM-enhanced summaries.
Supports both formal presentations and multi-speaker conversations.
"""

import os
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UnifiedFAISSEmbeddingGenerator:
    """Unified FAISS embedding generator supporting multiple pipeline types"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the unified FAISS embedding generator
        
        Args:
            pipeline_config: Pipeline-specific configuration
            progress_queue: Progress tracking queue
        """
        self.config = pipeline_config
        self.progress_queue = progress_queue
        
        # Pipeline metadata
        self.pipeline_type = pipeline_config.get('pipeline_type', 'unknown')
        self.embedding_config = pipeline_config.get('embedding_config', {})
        self.llm_config = pipeline_config.get('llm_config', {})
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Initialize OpenAI client if LLM enhancement is enabled
        self.openai_client = None
        if pipeline_config.get('llm_enhancement', False):
            self._initialize_openai_client()
        
        # Input/output directories
        self.input_dir = Path("step_06_frame_chunk_alignment")
        self.output_dir = Path("step_07_faiss_embeddings")
        self.output_dir.mkdir(exist_ok=True)
        
        # Embedding parameters
        self.embedding_model = self.embedding_config.get('model', 'text-embedding-3-large')
        self.chunk_size = self.embedding_config.get('chunk_size', 1000)
        self.include_metadata = self.embedding_config.get('include_metadata', True)
        self.include_content_summary = self.embedding_config.get('include_content_summary', True)
        
        # Processing statistics
        self.stats = {
            'total_videos': 0,
            'total_chunks': 0,
            'chunks_processed': 0,
            'embeddings_generated': 0,
            'failed_embeddings': 0,
            'total_tokens': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        try:
            # Try to import centralized logging
            from logging_config import setup_logging
            return setup_logging(f'{self.pipeline_type}_embeddings')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_embeddings')
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client for embedding generation"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                self.logger.info("✅ OpenAI client initialized for embedding generation")
            else:
                self.logger.warning("⚠️ OPENAI_API_KEY not found, embedding generation disabled")
        except ImportError:
            self.logger.warning("⚠️ OpenAI package not available, embedding generation disabled")
        except Exception as e:
            self.logger.warning(f"⚠️ OpenAI client initialization failed: {e}")
    
    def process_all_videos(self):
        """Process all videos in the input directory"""
        if not self.input_dir.exists():
            self.logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find all RAG-ready content files
        rag_files = list(self.input_dir.glob("*_rag_ready_aligned_content.json"))
        self.logger.info(f"Found {len(rag_files)} RAG-ready content files")
        
        self.stats['total_videos'] = len(rag_files)
        
        for rag_file in rag_files:
            try:
                self.process_single_video(rag_file)
            except Exception as e:
                self.logger.error(f"Failed to process {rag_file.name}: {e}")
        
        self.log_processing_summary()
    
    def process_single_video(self, rag_file: Path):
        """Process a single video's RAG-ready content"""
        video_id = rag_file.stem.replace('_rag_ready_aligned_content', '')
        self.logger.info(f"Processing video: {video_id}")
        
        # Check progress
        if self.progress_queue:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_07_faiss_embeddings') == 'completed':
                self.logger.info(f"Embeddings already completed for {video_id}, skipping")
                return
        
        try:
            # Load RAG-ready content
            with open(rag_file, 'r') as f:
                rag_content = json.load(f)
            
            # Extract text chunks and metadata
            text_chunks, metadata = self._extract_text_and_metadata(rag_content)
            
            if not text_chunks:
                self.logger.warning(f"No text chunks found for {video_id}")
                return
            
            # Generate embeddings
            embeddings = self._generate_embeddings(text_chunks)
            
            if not embeddings:
                self.logger.error(f"Failed to generate embeddings for {video_id}")
                return
            
            # Save embeddings and metadata
            self._save_embeddings(video_id, embeddings, metadata, rag_content)
            
            # Update progress
            if self.progress_queue:
                self.progress_queue.update_video_step(video_id, 'step_07_faiss_embeddings', 'completed', {
                    'chunks_processed': len(text_chunks),
                    'embeddings_generated': len(embeddings),
                    'embedding_model': self.embedding_model
                })
            
            self.logger.info(f"✅ Successfully processed {video_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing {video_id}: {e}")
    
    def _extract_text_and_metadata(self, rag_content: Dict) -> Tuple[List[str], List[Dict]]:
        """Extract text chunks and metadata from RAG content"""
        text_chunks = []
        metadata = []
        
        content = rag_content.get('content', [])
        self.stats['total_chunks'] += len(content)
        
        for chunk in content:
            try:
                # Extract text content
                text_content = chunk.get('text_content', {})
                text = text_content.get('text', '')
                
                if not text:
                    continue
                
                # Extract content summary for enhanced embedding
                content_summary = text_content.get('metadata', {}).get('content_summary', '')
                
                # Combine text and summary for better semantic search
                if content_summary and self.include_content_summary:
                    # Prepend summary to text for better context in embeddings
                    enhanced_text = f"{content_summary}\n\n{text}"
                    text_chunks.append(enhanced_text)
                    self.logger.debug(f"Including content summary ({len(content_summary)} chars) for enhanced embedding")
                else:
                    text_chunks.append(text)
                    self.logger.debug("No content summary available, using text only")
                
                # Create comprehensive metadata entry
                meta_entry = {
                    'chunk_id': chunk.get('chunk_id'),
                    'video_id': chunk.get('video_id'),
                    'pipeline_type': chunk.get('pipeline_type'),
                    'text': text,
                    'text_length': len(text),
                    'content_summary': content_summary,
                    'start_time': text_content.get('metadata', {}).get('start_time', 0),
                    'end_time': text_content.get('metadata', {}).get('end_time', 0),
                    'timestamp': text_content.get('metadata', {}).get('timestamp', ''),
                    'primary_topics': text_content.get('metadata', {}).get('primary_topics', []),
                    'secondary_topics': text_content.get('metadata', {}).get('secondary_topics', []),
                    'key_terms': text_content.get('metadata', {}).get('key_terms', []),
                    'visual_content': chunk.get('visual_content', {}),
                    'temporal_info': chunk.get('temporal_info', {}),
                    'conversation_context': chunk.get('conversation_context', {}),
                    'created_at': chunk.get('created_at', '')
                }
                
                metadata.append(meta_entry)
                
                # Update statistics
                self.stats['total_tokens'] += len(text.split())
                
            except Exception as e:
                self.logger.warning(f"Failed to extract text from chunk: {e}")
                continue
        
        self.stats['chunks_processed'] += len(text_chunks)
        return text_chunks, metadata
    
    def _generate_embeddings(self, text_chunks: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings using OpenAI API"""
        if not self.openai_client:
            self.logger.error("OpenAI client not available for embedding generation")
            return None
        
        try:
            self.logger.info(f"Generating embeddings for {len(text_chunks)} text chunks...")
            
            # Generate embeddings in batches
            batch_size = 100  # OpenAI API limit
            all_embeddings = []
            
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(text_chunks) + batch_size - 1)//batch_size}")
                
                try:
                    response = self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                        encoding_format="float"
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    self.logger.debug(f"Generated {len(batch_embeddings)} embeddings for batch")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                    # Continue with remaining batches
                    continue
            
            if not all_embeddings:
                self.logger.error("No embeddings generated")
                return None
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            self.stats['embeddings_generated'] += len(embeddings_array)
            
            self.logger.info(f"✅ Successfully generated {len(embeddings_array)} embeddings")
            return embeddings_array
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def _save_embeddings(self, video_id: str, embeddings: np.ndarray, metadata: List[Dict], rag_content: Dict):
        """Save embeddings and metadata"""
        try:
            # Create timestamped output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            output_dir = self.output_dir / f"run_{timestamp}"
            output_dir.mkdir(exist_ok=True)
            
            # Save embeddings as numpy array
            embeddings_file = output_dir / "chunks_embeddings.npy"
            np.save(embeddings_file, embeddings)
            
            # Save metadata as pickle
            metadata_file = output_dir / "chunks_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save processing summary
            summary = {
                'video_id': video_id,
                'pipeline_type': self.pipeline_type,
                'embedding_model': self.embedding_model,
                'total_chunks': len(metadata),
                'embeddings_shape': embeddings.shape,
                'embedding_dimension': embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                'processing_timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'statistics': {
                    'total_tokens': self.stats['total_tokens'],
                    'chunks_processed': self.stats['chunks_processed'],
                    'embeddings_generated': self.stats['embeddings_generated']
                }
            }
            
            summary_file = output_dir / "processing_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Create symlink to latest
            latest_link = self.output_dir / "latest"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(output_dir.name)
            
            self.logger.info(f"Embeddings saved to {output_dir}")
            self.logger.info(f"  - Embeddings: {embeddings_file}")
            self.logger.info(f"  - Metadata: {metadata_file}")
            self.logger.info(f"  - Summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings for {video_id}: {e}")
            raise
    
    def log_processing_summary(self):
        """Log a summary of processing statistics"""
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info(f"🎉 {self.pipeline_type.upper()} FAISS Embedding Generation Summary:")
        self.logger.info(f"  Total videos: {self.stats['total_videos']}")
        self.logger.info(f"  Total chunks: {self.stats['total_chunks']}")
        self.logger.info(f"  Chunks processed: {self.stats['chunks_processed']}")
        self.logger.info(f"  Embeddings generated: {self.stats['embeddings_generated']}")
        self.logger.info(f"  Failed: {self.stats['failed_embeddings']}")
        self.logger.info(f"  Total tokens: {self.stats['total_tokens']}")
        self.logger.info(f"  Duration: {duration}")
        
        if self.stats['total_chunks'] > 0:
            success_rate = (self.stats['embeddings_generated'] / self.stats['total_chunks']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        if self.stats['embeddings_generated'] > 0:
            avg_tokens_per_chunk = self.stats['total_tokens'] / self.stats['embeddings_generated']
            self.logger.info(f"  Average tokens per chunk: {avg_tokens_per_chunk:.1f}")
