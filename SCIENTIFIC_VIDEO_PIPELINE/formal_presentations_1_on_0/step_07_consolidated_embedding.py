#!/usr/bin/env python3
"""
Step 07: Consolidated Embedding
Creates embeddings for text and visual content, builds FAISS indices, and provides search capabilities.
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

class ConsolidatedEmbedding:
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
        self.visual_embedding_dim = 64   # Simplified visual features
        
        # FAISS parameters
        self.index_type = "IndexFlatL2"  # L2 distance for similarity search
        
        # Initialize models
        self.load_models()
    
    def load_models(self):
        """Load embedding models"""
        try:
            logger.info("Loading OpenAI client...")
            # OpenAI client is already initialized in __init__
            logger.info("OpenAI client loaded successfully")
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
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
    
    def get_latest_embeddings_info(self) -> tuple:
        """Get information about the most recent embeddings"""
        # First look for new organized structure (run directories)
        run_dirs = list(self.output_dir.glob("run_*"))
        if run_dirs:
            # Use new organized format
            latest_run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
            timestamp = latest_run_dir.name.replace("run_", "")
            
            # Check if embeddings exist in the run directory
            embeddings_path = latest_run_dir / "embeddings.npy"
            if embeddings_path.exists():
                info_file = latest_run_dir / "embedding_info.json"
                return embeddings_path, timestamp, info_file
        
        # Fallback to legacy format
        embedding_files = list(self.output_dir.glob("embeddings_*.npy"))
        if not embedding_files:
            return None, None, None
        
        # Get the most recent embeddings
        latest_embeddings = max(embedding_files, key=lambda x: x.stat().st_mtime)
        
        # Extract timestamp from filename
        timestamp = latest_embeddings.stem.replace("embeddings_", "")
        
        # Check for corresponding info file in root (legacy)
        info_file = self.output_dir / f"embedding_info_{timestamp}.json"
        
        return latest_embeddings, timestamp, info_file
    
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
        
        # Fallback: Check if we can skip processing based on content changes
        # Get latest embeddings info
        latest_embeddings, timestamp, info_file = self.get_latest_embeddings_info()
        if not latest_embeddings or not info_file:
            logger.info("No existing embeddings found, will create new ones")
            return False
        
        # Check if info file exists and contains content hash
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
                logger.info(f"Using existing embeddings from: {timestamp}")
                return True
            else:
                logger.info(f"Content has changed (previous: {previous_hash[:8]}..., current: {current_hash[:8]}...), will create new embeddings")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking content hash: {e}, will create new embeddings")
            return False
    
    def save_content_hash_info(self, timestamp: str, rag_files: List[Path]):
        """Save content hash information for future change detection"""
        try:
            content_hash = self.calculate_content_hash(rag_files)
            
            info_data = {
                'timestamp': timestamp,
                'content_hash': content_hash,
                'input_files': [str(f.name) for f in rag_files],
                'file_count': len(rag_files),
                'created_at': datetime.now().isoformat(),
                'pipeline_step': 'step_07_consolidated_embedding'
            }
            
            # Save in run directory if it exists, otherwise in root (for backward compatibility)
            run_dir = self.output_dir / f"run_{timestamp}"
            if run_dir.exists():
                info_file = run_dir / "content_hash_info.json"
            else:
                info_file = self.output_dir / f"embedding_info_{timestamp}.json"
                
            with open(info_file, 'w') as f:
                json.dump(info_data, f, indent=2)
            
            logger.info(f"Content hash info saved: {info_file.name}")
            logger.info(f"Content hash: {content_hash[:8]}...")
            
        except Exception as e:
            logger.warning(f"Failed to save content hash info: {e}")
    
    def process_all_content(self):
        """Process all aligned content in the input directory"""
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
        all_visual_features = []
        all_metadata = []
        
        for rag_file in rag_files:
            try:
                text_chunks, visual_features, metadata = self.process_single_file(rag_file)
                all_text_chunks.extend(text_chunks)
                all_visual_features.extend(visual_features)
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
        
        logger.info(f"Creating embeddings for {len(all_visual_features)} visual features...")
        visual_embeddings = self.create_simple_visual_embeddings(all_visual_features)
        
        # Create FAISS indices
        self.create_faiss_indices(text_embeddings, visual_embeddings, all_metadata, timestamp)
        
        # Create search demo
        self.create_search_demo(text_embeddings, all_metadata, timestamp)
        
        # Save content hash for future change detection
        self.save_content_hash_info(timestamp, rag_files)
        
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
                            'visual_index': f"run_{timestamp}/visual_index.faiss",
                            'combined_index': f"run_{timestamp}/combined_index.faiss",
                            'embeddings': f"run_{timestamp}/embeddings.npy",
                            'metadata': f"run_{timestamp}/metadata.pkl",
                            'streamlit_files': f"run_{timestamp}/consolidated/"
                        },
                        'completed_at': datetime.now().isoformat()
                    }
                )
                logger.info(f"📊 Progress queue updated: step 7 completed for {video_id}")
        
        logger.info("Consolidated embedding step completed successfully")
    
    def process_single_file(self, rag_file: Path) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process a single RAG-ready file"""
        with open(rag_file, 'r') as f:
            rag_data = json.load(f)
        
        text_chunks = []
        visual_features = []
        metadata = []
        
        for content in rag_data.get('aligned_content', []):
            # Extract text content
            text_content = content.get('text_content', {})
            text = text_content.get('text', '')
            if text:
                text_chunks.append(text)
                
                # Create metadata entry
                chunk_meta = text_content.get('metadata', {})
                meta_entry = {
                    'content_id': content.get('content_id', ''),
                    'text': text,  # Include the actual text content for search results
                    'text_length': len(text),
                    'chunk_metadata': chunk_meta,
                    'visual_content': content.get('visual_content', {}),
                    'temporal_info': content.get('temporal_info', {}),
                    'quality_metrics': content.get('quality_metrics', {}),
                    'source_file': rag_file.name,
                    # Preserve timestamp information for video citations
                    'start_time_seconds': chunk_meta.get('start_time_seconds'),
                    'end_time_seconds': chunk_meta.get('end_time_seconds'),
                    'start_time_ms': chunk_meta.get('start_time_ms'),
                    'end_time_ms': chunk_meta.get('end_time_ms')
                }
                metadata.append(meta_entry)
            
            # Extract visual content
            visual_content = content.get('visual_content', {})
            frames = visual_content.get('frames', [])
            if frames:
                visual_features.append({
                    'frame_count': len(frames),
                    'frame_sizes': [f.get('file_size', 0) for f in frames],
                    'timestamps': [f.get('timestamp', 0) for f in frames],
                    'alignment_confidence': visual_content.get('frame_count', 0)
                })
            else:
                # No frames, create placeholder
                visual_features.append({
                    'frame_count': 0,
                    'frame_sizes': [],
                    'timestamps': [],
                    'alignment_confidence': 0
                })
        
        return text_chunks, visual_features, metadata
    
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
                
                if (i + 1) % 20 == 0:
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
    
    def create_simple_visual_embeddings(self, visual_features: List[Dict[str, Any]]) -> np.ndarray:
        """Create simplified visual embeddings based on metadata"""
        embeddings = []
        
        for features in visual_features:
            # Create a simple feature vector based on metadata
            frame_count = features.get('frame_count', 0)
            total_size = sum(features.get('frame_sizes', [0]))
            avg_timestamp = np.mean(features.get('timestamps', [0])) if features.get('timestamps') else 0
            confidence = features.get('alignment_confidence', 0)
            
            # Create 64-dimensional feature vector
            feature_vector = [
                frame_count / 100.0,  # Normalize frame count
                min(total_size / 1000000.0, 1.0),  # Normalize total size (MB)
                min(avg_timestamp / 3600.0, 1.0),  # Normalize timestamp (hours)
                confidence / 100.0,  # Normalize confidence
                # Add some random variation for diversity
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                # ... add more features to reach 64 dimensions
            ]
            
            # Pad to 64 dimensions
            while len(feature_vector) < self.visual_embedding_dim:
                feature_vector.append(np.random.normal(0, 0.1))
            
            embeddings.append(feature_vector)
        
        return np.array(embeddings, dtype=np.float32)
    
    def create_faiss_indices(self, text_embeddings: np.ndarray, 
                            visual_embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                            timestamp: str):
        """Create FAISS indices for text and visual embeddings"""
        try:
            # Create text index
            text_index = faiss.IndexFlatL2(text_embeddings.shape[1])
            text_index.add(text_embeddings)
            
            # Create visual index
            visual_index = faiss.IndexFlatL2(visual_embeddings.shape[1])
            visual_index.add(visual_embeddings)
            
            # Create combined index (concatenate features)
            combined_features = np.hstack([text_embeddings, visual_embeddings])
            combined_index = faiss.IndexFlatL2(combined_features.shape[1])
            combined_index.add(combined_features)
            
            # Save indices using provided timestamp
            
            # Create clean timestamped directory for this run
            run_dir = self.output_dir / f"run_{timestamp}"
            run_dir.mkdir(exist_ok=True)
            
            # Save ALL files in the timestamped directory with clear names
            # Text index
            text_index_path = run_dir / "text_index.faiss"
            faiss.write_index(text_index, str(text_index_path))
            
            # Visual index
            visual_index_path = run_dir / "visual_index.faiss"
            faiss.write_index(visual_index, str(visual_index_path))
            
            # Combined index
            combined_index_path = run_dir / "combined_index.faiss"
            faiss.write_index(combined_index, str(combined_index_path))
            
            # Embeddings
            embeddings_path = run_dir / "embeddings.npy"
            np.save(str(embeddings_path), combined_features)
            
            # Metadata
            metadata_path = run_dir / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Streamlit-compatible files (symlinks or copies to expected names)
            # Create a consolidated subdirectory for Streamlit compatibility
            consolidated_dir = run_dir / "consolidated"
            consolidated_dir.mkdir(exist_ok=True)
            
            # Save Streamlit-expected files
            chunks_index_path = consolidated_dir / "chunks.index"
            faiss.write_index(combined_index, str(chunks_index_path))
            
            chunks_embeddings_path = consolidated_dir / "chunks_embeddings.npy"
            np.save(str(chunks_embeddings_path), combined_features)
            
            chunks_metadata_path = consolidated_dir / "chunks_metadata.pkl"
            with open(chunks_metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Create comprehensive info.json
            info_data = {
                'timestamp': timestamp,
                'run_directory': str(run_dir),
                'total_chunks': len(metadata),
                'total_frames': sum(1 for meta in metadata if meta.get('visual_content')),
                'videos_processed': list(set(meta.get('content_id', '').split('_')[0] for meta in metadata if meta.get('content_id'))),
                'embedding_dimensions': combined_features.shape[1],
                'created_at': datetime.now().isoformat(),
                'files': {
                    'text_index': 'text_index.faiss',
                    'visual_index': 'visual_index.faiss',
                    'combined_index': 'combined_index.faiss',
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
            
            # Save index info
            index_info = {
                'timestamp': timestamp,
                'run_directory': str(run_dir),
                'text_embeddings_shape': text_embeddings.shape,
                'visual_embeddings_shape': visual_embeddings.shape,
                'combined_embeddings_shape': combined_features.shape,
                'total_content_items': len(metadata),
                'embedding_model': self.embedding_model_name,
                'index_type': self.index_type,
                'files': {
                    'text_index': 'text_index.faiss',
                    'visual_index': 'visual_index.faiss',
                    'combined_index': 'combined_index.faiss',
                    'embeddings': 'embeddings.npy',
                    'metadata': 'metadata.pkl',
                    'consolidated_dir': 'consolidated/',
                    'chunks_index': 'consolidated/chunks.index',
                    'chunks_embeddings': 'consolidated/chunks_embeddings.npy',
                    'chunks_metadata': 'consolidated/chunks_metadata.pkl',
                    'info_json': 'info.json'
                },
                'note': 'All files organized in timestamped run directory for clean organization',
                'consistency_note': 'Using OpenAI text-embedding-3-large for consistency with publication pipeline',
                'organization': 'Clean timestamped directory structure with Streamlit compatibility'
            }
            
            # Save index info in the run directory (not in root)
            info_path = run_dir / "embedding_info.json"
            with open(info_path, 'w') as f:
                json.dump(index_info, f, indent=2)
            
            logger.info(f"FAISS indices saved with timestamp: {timestamp}")
            logger.info(f"✅ Created clean run directory: {run_dir}")
            logger.info(f"✅ All files organized in timestamped directory for clean structure")
            logger.info(f"✅ Streamlit-compatible files saved in: {consolidated_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS indices: {e}")
            raise
    
    def create_search_demo(self, text_embeddings: np.ndarray, metadata: List[Dict[str, Any]], timestamp: str):
        """Create a simple search demonstration"""
        try:
            # Create a simple search index for demo
            demo_index = faiss.IndexFlatL2(text_embeddings.shape[1])
            demo_index.add(text_embeddings)
            
            # Save demo files in the run directory for organization
            run_dir = self.output_dir / f"run_{timestamp}"
            run_dir.mkdir(exist_ok=True)
            
            # Save demo index
            demo_path = run_dir / "demo_search_index.faiss"
            faiss.write_index(demo_index, str(demo_path))
            
            # Save demo metadata
            demo_metadata_path = run_dir / "demo_metadata.pkl"
            with open(demo_metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("Search demo created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create search demo: {e}")
    
    def get_timestamp(self) -> str:
        """Get current timestamp string"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def analyze_embedding_quality(self):
        """Analyze the quality of created embeddings"""
        try:
            # Find latest embedding info in run directories (new format)
            run_dirs = list(self.output_dir.glob("run_*"))
            if run_dirs:
                # Use new organized format
                latest_run_dir = max(run_dirs, key=lambda x: x.stat().st_mtime)
                info_path = latest_run_dir / "embedding_info.json"
                
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    
                    logger.info("=== Embedding Quality Analysis (New Format) ===")
                    logger.info(f"Run directory: {info.get('run_directory', 'N/A')}")
                    logger.info(f"Total content items: {info['total_content_items']}")
                    logger.info(f"Text embeddings: {info['text_embeddings_shape']}")
                    logger.info(f"Visual embeddings: {info['visual_embeddings_shape']}")
                    logger.info(f"Combined embeddings: {info['combined_embeddings_shape']}")
                    logger.info(f"Embedding model: {info['embedding_model']}")
                    logger.info(f"Index type: {info['index_type']}")
                    logger.info(f"Processing timestamp: {info['timestamp']}")
                    return
            
            # Fallback to legacy format
            info_files = list(self.output_dir.glob("embedding_info_*.json"))
            if not info_files:
                logger.warning("No embedding info files found")
                return
            
            latest_info = max(info_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_info, 'r') as f:
                info = json.load(f)
            
            logger.info("=== Embedding Quality Analysis (Legacy Format) ===")
            logger.info(f"Total content items: {info['total_content_items']}")
            logger.info(f"Text embeddings: {info['text_embeddings_shape']}")
            logger.info(f"Visual embeddings: {info['visual_embeddings_shape']}")
            logger.info(f"Combined embeddings: {info['combined_embeddings_shape']}")
            logger.info(f"Embedding model: {info['embedding_model']}")
            logger.info(f"Index type: {info['index_type']}")
            logger.info(f"Processing timestamp: {info['timestamp']}")
            
        except Exception as e:
            logger.error(f"Failed to analyze embedding quality: {e}")

def main():
    """Main execution function"""
    try:
        # Initialize progress queue
        progress_queue = get_progress_queue()
        logger.info("✅ Progress queue initialized")
        
        embedder = ConsolidatedEmbedding(progress_queue)
        embedder.process_all_content()
        embedder.analyze_embedding_quality()
        logger.info("Consolidated embedding step completed successfully")
    except Exception as e:
        logger.error(f"Consolidated embedding step failed: {e}")
        raise

if __name__ == "__main__":
    main()
