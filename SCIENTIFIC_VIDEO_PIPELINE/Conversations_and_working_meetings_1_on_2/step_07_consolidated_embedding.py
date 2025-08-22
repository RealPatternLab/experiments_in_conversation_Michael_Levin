#!/usr/bin/env python3
"""
Step 07: Consolidated Embedding for Conversations Pipeline
Creates embeddings for RAG-ready aligned content, builds FAISS indices, and provides search capabilities.
"""

import os
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import openai
import faiss
from datetime import datetime
from pipeline_progress_queue import get_progress_queue

# Load environment variables
load_dotenv()

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('consolidated_embedding')

class ConversationsConsolidatedEmbedding:
    def __init__(self, progress_queue=None):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        self.input_dir = Path("step_06_frame_chunk_alignment")
        self.output_dir = Path("step_07_faiss_embeddings")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = progress_queue
        
        # Embedding parameters
        self.embedding_model_name = "text-embedding-3-large"  # OpenAI model
        self.text_embedding_dim = 3072  # text-embedding-3-large dimension
        
        # FAISS parameters
        self.index_type = "IndexFlatIP"  # Inner product for cosine similarity
        
        logger.info("Conversations consolidated embedding initialized successfully")
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp for file naming"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    def calculate_content_hash(self, rag_files: List[Path]) -> str:
        """Calculate a hash of the content to detect changes"""
        import hashlib
        
        # Sort files for consistent hashing
        sorted_files = sorted(rag_files, key=lambda x: x.name)
        
        # Create a combined hash of file names, sizes, and modification times
        hash_input = ""
        for rag_file in sorted_files:
            stat = rag_file.stat()
            hash_input += f"{rag_file.name}:{stat.st_size}:{stat.st_mtime}:"
        
        # Also hash the actual content of each file
        for rag_file in sorted_files:
            try:
                with open(rag_file, 'rb') as f:
                    content_hash = hashlib.md5(f.read()).hexdigest()
                    hash_input += f"{content_hash}:"
            except Exception as e:
                logger.warning(f"Could not hash content of {rag_file.name}: {e}")
                # Use file size and modification time as fallback
                stat = rag_file.stat()
                hash_input += f"{stat.st_size}:{stat.st_mtime}:"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def should_skip_processing(self, rag_files: List[Path]) -> bool:
        """Check if we can skip processing based on content changes"""
        # First check progress queue for all videos
        if self.progress_queue:
            all_videos_completed = True
            for rag_file in rag_files:
                video_id = rag_file.stem.replace('_rag_ready_aligned_content', '')
                video_status = self.progress_queue.get_video_status(video_id)
                if not video_status or video_status.get('step_07_consolidated_embedding') != 'completed':
                    all_videos_completed = False
                    break
            
            if all_videos_completed:
                logger.info("✅ All videos already completed in step 7 (progress queue), skipping")
                return True
        
        # Check if we can skip processing based on content changes
        # Look for existing embeddings
        run_dirs = list(self.output_dir.glob("run_*"))
        if not run_dirs:
            logger.info("No existing embeddings found, will create new ones")
            return False
        
        # Get latest run directory
        latest_run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
        info_file = latest_run_dir / "embedding_info.json"
        
        if not info_file.exists():
            logger.info("No embedding info file found, will create new embeddings")
            return False
        
        try:
            with open(info_file, 'r') as f:
                info_data = json.load(f)
            
            previous_hash = info_data.get('content_hash')
            if not previous_hash:
                logger.info("No content hash in info file, will create new embeddings")
                return False
            
            # Calculate current content hash
            current_hash = self.calculate_content_hash(rag_files)
            
            if current_hash == previous_hash:
                logger.info(f"Content unchanged since last run (hash: {current_hash[:8]}...), skipping embedding creation")
                logger.info(f"Using existing embeddings from: {latest_run_dir.name}")
                return True
            else:
                logger.info(f"Content has changed (previous: {previous_hash[:8]}..., current: {current_hash[:8]}...), will create new embeddings")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking content hash: {e}, will create new embeddings")
            return False
    
    def process_all_content(self):
        """Process all RAG-ready aligned content in the input directory"""
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find all RAG-ready files
        rag_files = list(self.input_dir.glob("*_rag_ready_aligned_content.json"))
        logger.info(f"Found {len(rag_files)} RAG-ready files")
        
        if not rag_files:
            logger.error("No RAG-ready files found. Run step 6 first.")
            return
        
        # Check if we can skip processing based on content changes
        if self.should_skip_processing(rag_files):
            logger.info("✅ Skipping embedding creation - content unchanged")
            return
        
        # Process each file
        all_text_chunks = []
        all_metadata = []
        
        for rag_file in rag_files:
            try:
                text_chunks, metadata = self.process_single_file(rag_file)
                all_text_chunks.extend(text_chunks)
                all_metadata.extend(metadata)
            except Exception as e:
                logger.error(f"Failed to process {rag_file.name}: {e}")
        
        if not all_text_chunks:
            logger.error("No content found to embed")
            return
        
        # Generate timestamp for new embeddings
        timestamp = self.get_current_timestamp()
        logger.info(f"Creating new embeddings with timestamp: {timestamp}")
        
        # Create embeddings
        logger.info(f"Creating embeddings for {len(all_text_chunks)} text chunks...")
        text_embeddings = self.create_text_embeddings(all_text_chunks)
        
        # Create FAISS indices
        self.create_faiss_indices(text_embeddings, all_metadata, timestamp)
        
        # Save content hash for future change detection
        self.save_content_hash_info(timestamp, rag_files, all_metadata)
        
        # Update progress queue for all processed videos
        if self.progress_queue:
            for rag_file in rag_files:
                video_id = rag_file.stem.replace('_rag_ready_aligned_content', '')
                self.progress_queue.update_video_step_status(
                    video_id,
                    'step_07_consolidated_embedding',
                    'completed',
                    metadata={
                        'timestamp': timestamp,
                        'content_hash': self.calculate_content_hash(rag_files),
                        'embedding_files': {
                            'run_directory': f"run_{timestamp}",
                            'text_index': f"run_{timestamp}/text_index.faiss",
                            'embeddings': f"run_{timestamp}/embeddings.npy",
                            'metadata': f"run_{timestamp}/metadata.pkl",
                            'consolidated_dir': f"run_{timestamp}/consolidated/"
                        },
                        'completed_at': datetime.now().isoformat()
                    }
                )
                logger.info(f"📊 Progress queue updated: step 7 completed for {video_id}")
        
        logger.info("Consolidated embedding step completed successfully")
    
    def process_single_file(self, rag_file: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process a single RAG-ready aligned content file"""
        with open(rag_file, 'r') as f:
            rag_data = json.load(f)
        
        text_chunks = []
        metadata = []
        
        # Handle new standardized structure (matches formal presentations pipeline)
        if isinstance(rag_data, dict) and 'aligned_content' in rag_data:
            content_list = rag_data['aligned_content']
        else:
            # Fallback to old structure
            content_list = rag_data if isinstance(rag_data, list) else []
        
        for content in content_list:
            # Extract text content for embedding (new structure)
            text = content.get('text_content', {}).get('text', '')
            if not text:
                # Fallback to old structure
                text = content.get('text', '')
            
            # Extract content summary for enhanced embedding
            content_summary = content.get('text_content', {}).get('metadata', {}).get('content_summary', '')
            
            # Combine text and summary for better semantic search
            if content_summary:
                # Prepend summary to text for better context in embeddings
                enhanced_text = f"{content_summary}\n\n{text}"
                text_chunks.append(enhanced_text)
                logger.debug(f"Including content summary ({len(content_summary)} chars) for enhanced embedding")
            else:
                text_chunks.append(text)
                logger.debug("No content summary available, using text only")
                
                # Create comprehensive metadata entry
                meta_entry = {
                    'id': content.get('content_id', content.get('id', '')),
                    'video_id': content.get('metadata', {}).get('video_id', ''),
                    'text': text,
                    'text_length': len(text),
                    'content_summary': content_summary,  # Include the LLM-generated summary
                    # Extract from conversation_context if available
                    'question': content.get('conversation_context', {}).get('question', ''),
                    'answer': content.get('conversation_context', {}).get('answer', ''),
                    'questioner': content.get('conversation_context', {}).get('questioner', ''),
                    'answerer': content.get('conversation_context', {}).get('answerer', ''),
                    'topics': content.get('text_content', {}).get('metadata', {}).get('primary_topics', []),
                    'timing': {
                        'start_time_seconds': content.get('text_content', {}).get('metadata', {}).get('start_time_seconds', 0),
                        'end_time_seconds': content.get('text_content', {}).get('metadata', {}).get('end_time_seconds', 0)
                    },
                    'youtube_link': content.get('conversation_context', {}).get('youtube_link', ''),
                    'visual_context': content.get('visual_content', {}),
                    'source_file': rag_file.name,
                    'pipeline_type': 'conversations_1_on_2',
                    'created_at': content.get('metadata', {}).get('created_at', '')
                }
                metadata.append(meta_entry)
        
        return text_chunks, metadata
    
    def create_text_embeddings(self, text_chunks: List[str]) -> np.ndarray:
        """Create text embeddings using OpenAI"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not loaded")
        
        embeddings = []
        failed_chunks = []
        
        for i, text in enumerate(text_chunks):
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=text,
                    encoding_format="float"
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(text_chunks)} chunks...")
                    
            except Exception as e:
                logger.warning(f"Failed to create embedding for chunk {i}: {e}")
                failed_chunks.append(i)
                # Add zero vector for failed chunks
                embeddings.append(np.zeros(self.text_embedding_dim))
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Filter out failed chunks
        if failed_chunks:
            valid_mask = np.any(embeddings != 0, axis=1)
            embeddings = embeddings[valid_mask]
            logger.info(f"Filtered out {len(failed_chunks)} failed chunks")
        
        return embeddings
    
    def create_faiss_indices(self, text_embeddings: np.ndarray, metadata: List[Dict[str, Any]], timestamp: str):
        """Create FAISS indices for text embeddings"""
        try:
            # Create text index using inner product for cosine similarity
            text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
            text_index.add(text_embeddings)
            
            # Create clean timestamped directory for this run
            run_dir = self.output_dir / f"run_{timestamp}"
            run_dir.mkdir(exist_ok=True)
            
            # Save all files in the timestamped directory
            # Text index
            text_index_path = run_dir / "text_index.faiss"
            faiss.write_index(text_index, str(text_index_path))
            
            # Embeddings
            embeddings_path = run_dir / "embeddings.npy"
            np.save(str(embeddings_path), text_embeddings)
            
            # Metadata
            metadata_path = run_dir / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Create consolidated subdirectory for compatibility
            consolidated_dir = run_dir / "consolidated"
            consolidated_dir.mkdir(exist_ok=True)
            
            # Save consolidated files with expected names
            chunks_index_path = consolidated_dir / "chunks.index"
            faiss.write_index(text_index, str(chunks_index_path))
            
            chunks_embeddings_path = consolidated_dir / "chunks_embeddings.npy"
            np.save(str(chunks_embeddings_path), text_embeddings)
            
            chunks_metadata_path = consolidated_dir / "chunks_metadata.pkl"
            with open(chunks_metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Create comprehensive info.json
            info_data = {
                'timestamp': timestamp,
                'run_directory': str(run_dir),
                'total_chunks': len(metadata),
                'total_frames': sum(1 for meta in metadata if meta.get('visual_context', {}).get('frame_count', 0) > 0),
                'videos_processed': list(set(meta.get('video_id', '') for meta in metadata if meta.get('video_id'))),
                'embedding_dimensions': text_embeddings.shape[1],
                'embedding_model': self.embedding_model_name,
                'index_type': self.index_type,
                'created_at': datetime.now().isoformat(),
                'pipeline_type': 'conversations_1_on_2',
                'files': {
                    'text_index': 'text_index.faiss',
                    'embeddings': 'embeddings.npy',
                    'metadata': 'metadata.pkl',
                    'consolidated_dir': 'consolidated/',
                    'chunks_index': 'consolidated/chunks.index',
                    'chunks_embeddings': 'consolidated/chunks_embeddings.npy',
                    'chunks_metadata': 'consolidated/chunks_metadata.pkl'
                }
            }
            
            info_path = run_dir / "info.json"
            with open(info_path, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            # Save embedding info
            embedding_info = {
                'timestamp': timestamp,
                'run_directory': str(run_dir),
                'text_embeddings_shape': text_embeddings.shape,
                'total_content_items': len(metadata),
                'embedding_model': self.embedding_model_name,
                'index_type': self.index_type,
                'content_hash': self.calculate_content_hash([Path(f"step_06_frame_chunk_alignment/{meta.get('source_file', '')}") for meta in metadata if meta.get('source_file')]),
                'files': {
                    'text_index': 'text_index.faiss',
                    'embeddings': 'embeddings.npy',
                    'metadata': 'metadata.pkl',
                    'consolidated_dir': 'consolidated/',
                    'chunks_index': 'consolidated/chunks.index',
                    'chunks_embeddings': 'consolidated/chunks_embeddings.npy',
                    'chunks_metadata': 'consolidated/chunks_metadata.pkl',
                    'info_json': 'info.json'
                },
                'note': 'Conversations pipeline embeddings with visual context',
                'organization': 'Clean timestamped directory structure with consolidated compatibility'
            }
            
            embedding_info_path = run_dir / "embedding_info.json"
            with open(embedding_info_path, 'w') as f:
                json.dump(embedding_info, f, indent=2)
            
            logger.info(f"FAISS indices saved with timestamp: {timestamp}")
            logger.info(f"✅ Created clean run directory: {run_dir}")
            logger.info(f"✅ All files organized in timestamped directory")
            logger.info(f"✅ Consolidated compatibility files saved in: {consolidated_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS indices: {e}")
            raise
    
    def save_content_hash_info(self, timestamp: str, rag_files: List[Path], metadata: List[Dict[str, Any]]):
        """Save content hash information for future change detection"""
        try:
            content_hash = self.calculate_content_hash(rag_files)
            
            hash_info = {
                'timestamp': timestamp,
                'content_hash': content_hash,
                'input_files': [str(f.name) for f in rag_files],
                'file_count': len(rag_files),
                'total_chunks': len(metadata),
                'created_at': datetime.now().isoformat(),
                'pipeline_step': 'step_07_consolidated_embedding',
                'pipeline_type': 'conversations_1_on_2'
            }
            
            # Save in run directory
            run_dir = self.output_dir / f"run_{timestamp}"
            hash_file = run_dir / "content_hash_info.json"
                
            with open(hash_file, 'w') as f:
                json.dump(hash_info, f, indent=2)
            
            logger.info(f"Content hash info saved: {hash_file.name}")
            logger.info(f"Content hash: {content_hash[:8]}...")
            
        except Exception as e:
            logger.warning(f"Failed to save content hash info: {e}")
    
    def analyze_embedding_quality(self):
        """Analyze the quality of created embeddings"""
        try:
            # Find latest embedding info in run directories
            run_dirs = list(self.output_dir.glob("run_*"))
            if not run_dirs:
                logger.warning("No embedding run directories found")
                return
            
            # Use latest run directory
            latest_run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
            info_path = latest_run_dir / "embedding_info.json"
            
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                logger.info("=== Conversations Embedding Quality Analysis ===")
                logger.info(f"Run directory: {info.get('run_directory', 'N/A')}")
                logger.info(f"Total content items: {info['total_content_items']}")
                logger.info(f"Text embeddings: {info['text_embeddings_shape']}")
                logger.info(f"Embedding model: {info['embedding_model']}")
                logger.info(f"Index type: {info['index_type']}")
                logger.info(f"Processing timestamp: {info['timestamp']}")
                logger.info(f"Pipeline type: {info.get('pipeline_type', 'N/A')}")
                
                # Show video breakdown
                videos = info.get('videos_processed', [])
                if videos:
                    logger.info(f"Videos processed: {', '.join(videos)}")
                
                # Show frame coverage
                total_frames = info.get('total_frames', 0)
                if total_frames > 0:
                    logger.info(f"Total frames with visual context: {total_frames}")
                    coverage_percentage = (total_frames / info['total_content_items']) * 100
                    logger.info(f"Visual context coverage: {coverage_percentage:.1f}%")
                
                return
            
            logger.warning("No embedding info file found in latest run directory")
            
        except Exception as e:
            logger.error(f"Failed to analyze embedding quality: {e}")

def main():
    """Main execution function"""
    try:
        # Initialize progress queue
        progress_queue = get_progress_queue()
        logger.info("✅ Progress queue initialized")
        
        embedder = ConversationsConsolidatedEmbedding(progress_queue)
        embedder.process_all_content()
        embedder.analyze_embedding_quality()
        logger.info("Conversations consolidated embedding step completed successfully")
    except Exception as e:
        logger.error(f"Conversations consolidated embedding step failed: {e}")
        raise

if __name__ == "__main__":
    main()
