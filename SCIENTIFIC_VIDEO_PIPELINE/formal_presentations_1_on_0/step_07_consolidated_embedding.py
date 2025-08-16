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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('consolidated_embedding.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConsolidatedEmbedding:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        self.input_dir = Path("step_06_frame_chunk_alignment")
        self.output_dir = Path("step_07_faiss_embeddings")
        self.output_dir.mkdir(exist_ok=True)
        
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
        
        # Create embeddings
        logger.info(f"Creating embeddings for {len(all_text_chunks)} text chunks...")
        text_embeddings = self.create_text_embeddings(all_text_chunks)
        
        logger.info(f"Creating embeddings for {len(all_visual_features)} visual features...")
        visual_embeddings = self.create_simple_visual_embeddings(all_visual_features)
        
        # Create FAISS indices
        self.create_faiss_indices(text_embeddings, visual_embeddings, all_metadata)
        
        # Create search demo
        self.create_search_demo(text_embeddings, all_metadata)
        
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
                meta_entry = {
                    'content_id': content.get('content_id', ''),
                    'text_length': len(text),
                    'chunk_metadata': text_content.get('metadata', {}),
                    'visual_content': content.get('visual_content', {}),
                    'temporal_info': content.get('temporal_info', {}),
                    'quality_metrics': content.get('quality_metrics', {}),
                    'source_file': rag_file.name
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
                            visual_embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
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
            
            # Save indices
            timestamp = self.get_timestamp()
            
            # Save text index
            text_index_path = self.output_dir / f"text_index_{timestamp}.faiss"
            faiss.write_index(text_index, str(text_index_path))
            
            # Save visual index
            visual_index_path = self.output_dir / f"visual_index_{timestamp}.faiss"
            faiss.write_index(visual_index, str(visual_index_path))
            
            # Save combined index
            combined_index_path = self.output_dir / f"combined_index_{timestamp}.faiss"
            faiss.write_index(combined_index, str(combined_index_path))
            
            # Save embeddings and metadata
            embeddings_path = self.output_dir / f"embeddings_{timestamp}.npy"
            np.save(str(embeddings_path), {
                'text': text_embeddings,
                'visual': visual_embeddings,
                'combined': combined_features
            })
            
            metadata_path = self.output_dir / f"metadata_{timestamp}.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save index info
            index_info = {
                'timestamp': timestamp,
                'text_embeddings_shape': text_embeddings.shape,
                'visual_embeddings_shape': visual_embeddings.shape,
                'combined_embeddings_shape': combined_features.shape,
                'total_content_items': len(metadata),
                'embedding_model': self.embedding_model_name,
                'index_type': self.index_type,
                'files': {
                    'text_index': str(text_index_path),
                    'visual_index': str(visual_index_path),
                    'combined_index': str(combined_index_path),
                    'embeddings': str(embeddings_path),
                    'metadata': str(metadata_path)
                },
                'note': 'Visual embeddings are simplified features based on metadata',
                'consistency_note': 'Using OpenAI text-embedding-3-large for consistency with publication pipeline'
            }
            
            info_path = self.output_dir / f"embedding_info_{timestamp}.json"
            with open(info_path, 'w') as f:
                json.dump(index_info, f, indent=2)
            
            logger.info(f"FAISS indices saved with timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS indices: {e}")
            raise
    
    def create_search_demo(self, text_embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Create a simple search demonstration"""
        try:
            # Create a simple search index for demo
            demo_index = faiss.IndexFlatL2(text_embeddings.shape[1])
            demo_index.add(text_embeddings)
            
            # Save demo index
            demo_path = self.output_dir / "demo_search_index.faiss"
            faiss.write_index(demo_index, str(demo_path))
            
            # Save demo metadata
            demo_metadata_path = self.output_dir / "demo_metadata.pkl"
            with open(demo_metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("Search demo created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create search demo: {e}")
    
    def get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def analyze_embedding_quality(self):
        """Analyze the quality of created embeddings"""
        try:
            # Find latest embedding info
            info_files = list(self.output_dir.glob("embedding_info_*.json"))
            if not info_files:
                logger.warning("No embedding info files found")
                return
            
            latest_info = max(info_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_info, 'r') as f:
                info = json.load(f)
            
            logger.info("=== Embedding Quality Analysis ===")
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
        embedder = ConsolidatedEmbedding()
        embedder.process_all_content()
        embedder.analyze_embedding_quality()
        logger.info("Consolidated embedding step completed successfully")
    except Exception as e:
        logger.error(f"Consolidated embedding step failed: {e}")
        raise

if __name__ == "__main__":
    main()
