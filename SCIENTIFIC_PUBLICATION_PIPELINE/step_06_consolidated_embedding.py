#!/usr/bin/env python3
"""
Step 6: Consolidated FAISS Vector Embedding Generator

This script generates a SINGLE, comprehensive FAISS vector index from ALL semantic chunks
across ALL documents. It creates one consolidated index instead of fragmented timestamped directories.

Key improvements:
- Processes ALL chunks from ALL documents in one run
- Creates a single FAISS index with comprehensive metadata
- Includes the entire corpus history
- Optimized for RAG system performance

Usage:
    uv run python step_06_consolidated_embedding.py
"""

import logging
import pickle
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sys
import json

# FAISS and OpenAI imports
try:
    import numpy as np
    import faiss
    from openai import OpenAI
    FAISS_AVAILABLE = True
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: uv add faiss-cpu openai numpy")
    FAISS_AVAILABLE = False
    OPENAI_AVAILABLE = False

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False
    logging.warning("python-dotenv not available. Environment variables may not load properly.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('consolidated_embedding.log')
    ]
)
logger = logging.getLogger(__name__)

class ConsolidatedEmbeddingGenerator:
    """Generates a single, comprehensive FAISS index from all semantic chunks."""
    
    def __init__(self):
        """Initialize the consolidated embedding generator."""
        self.output_dir = Path("step_06_faiss_embeddings")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE and ENV_AVAILABLE:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    logger.error("OPENAI_API_KEY not found in environment variables")
                else:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Initialize FAISS
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return
        
        logger.info("Consolidated embedding generator initialized successfully")
    
    def find_all_semantic_chunks(self) -> List[Tuple[Path, Dict, Dict]]:
        """Find all semantic chunks across all documents."""
        chunks_data = []
        
        # Look for semantic chunks in step_05_semantic_chunks
        chunks_dir = Path("step_05_semantic_chunks")
        if not chunks_dir.exists():
            logger.error("Semantic chunks directory not found")
            return []
        
        # Find all chunk files
        chunk_files = list(chunks_dir.glob("*.json"))
        logger.info(f"Found {len(chunk_files)} chunk files")
        
        # Look for metadata in step_02_metadata
        metadata_dir = Path("step_02_metadata")
        
        for chunk_file in chunk_files:
            try:
                # Load chunks
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                
                # Extract the actual chunks list and metadata
                chunks = chunk_data.get('chunks', [])
                file_metadata = chunk_data.get('metadata', {})
                
                # Find corresponding metadata from step_02
                doc_id = chunk_file.stem.replace('_chunks', '')
                metadata_file = metadata_dir / f"{doc_id}_metadata.json"
                
                doc_metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        doc_metadata = json.load(f)
                
                # Merge metadata (file metadata takes precedence)
                doc_metadata.update(file_metadata)
                
                # Add each chunk with its metadata
                for chunk in chunks:
                    chunks_data.append((chunk_file, chunk, doc_metadata))
                    
            except Exception as e:
                logger.error(f"Error loading {chunk_file}: {e}")
                continue
        
        logger.info(f"Total chunks found: {len(chunks_data)}")
        return chunks_data
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text chunk using OpenAI."""
        if not self.openai_client:
            logger.error("OpenAI client not available")
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def create_enhanced_metadata(self, chunk: Dict, doc_metadata: Dict, chunk_id: int, doc_id: str) -> Dict[str, Any]:
        """Create enhanced metadata for a chunk."""
        text = chunk.get('text', '')
        word_count = len(text.split()) if text else 0
        
        return {
            'chunk_id': f"{doc_id}_{chunk_id}",
            'chunk_index': chunk_id,
            'text': text,
            'section': chunk.get('section', ''),
            'primary_topic': chunk.get('primary_topic', ''),
            'secondary_topics': chunk.get('secondary_topics', []),
            'chunk_summary': chunk.get('chunk_summary', ''),
            'word_count': word_count,
            'character_count': len(text) if text else 0,
            
            # Document-level metadata
            'document_id': doc_id,
            'pdf_filename': doc_metadata.get('new_filename', ''),
            'title': doc_metadata.get('title', ''),
            'authors': doc_metadata.get('authors', []),
            'publication_year': doc_metadata.get('publication_year', ''),
            'journal': doc_metadata.get('journal', ''),
            'doi': doc_metadata.get('doi', ''),
            'abstract': doc_metadata.get('abstract', ''),
            
            # Pipeline metadata
            'pipeline_step': 'step_06_consolidated_embedding',
            'created_at': datetime.now().isoformat(),
            'embedding_model': 'text-embedding-3-large'
        }
    
    def generate_consolidated_embeddings(self) -> Dict[str, Any]:
        """Generate consolidated embeddings for all chunks."""
        logger.info("🚀 Starting consolidated embedding generation...")
        
        # Find all chunks
        chunks_data = self.find_all_semantic_chunks()
        if not chunks_data:
            return {
                'success': False,
                'message': 'No chunks found',
                'total_chunks': 0,
                'embeddings_generated': 0
            }
        
        # Generate embeddings for all chunks
        all_embeddings = []
        all_metadata = []
        successful = 0
        failed = 0
        
        logger.info(f"🔮 Generating embeddings for {len(chunks_data)} chunks...")
        
        for i, (chunk_file, chunk, doc_metadata) in enumerate(chunks_data):
            try:
                text = chunk.get('text', '')
                if not text:
                    logger.warning(f"Empty text for chunk {i}")
                    failed += 1
                    continue
                
                # Generate embedding
                embedding = self.generate_embedding(text)
                if embedding:
                    all_embeddings.append(embedding)
                    
                    # Create enhanced metadata
                    doc_id = chunk_file.stem.replace('_chunks', '')
                    chunk_metadata = self.create_enhanced_metadata(chunk, doc_metadata, i, doc_id)
                    all_metadata.append(chunk_metadata)
                    
                    successful += 1
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"✅ Processed {i + 1}/{len(chunks_data)} chunks")
                        
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                failed += 1
                continue
        
        logger.info(f"📊 Embedding generation complete:")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total: {len(chunks_data)}")
        
        if not all_embeddings:
            return {
                'success': False,
                'message': 'No embeddings generated',
                'total_chunks': len(chunks_data),
                'embeddings_generated': 0
            }
        
        # Convert to numpy arrays
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add vectors to index
        index.add(embeddings_array)
        
        # Save consolidated index and metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"consolidated_{timestamp}"
        output_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(index, str(output_path / "chunks.index"))
        
        # Save embeddings
        np.save(str(output_path / "chunks_embeddings.npy"), embeddings_array)
        
        # Save metadata
        with open(output_path / "chunks_metadata.pkl", 'wb') as f:
            pickle.dump(all_metadata, f)
        
        # Save summary info
        summary = {
            'timestamp': timestamp,
            'total_chunks': len(chunks_data),
            'embeddings_generated': successful,
            'failed': failed,
            'dimension': dimension,
            'index_type': 'IndexFlatIP',
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Saved consolidated embeddings to: {output_path}")
        logger.info(f"📊 Index contains {successful} embeddings with {dimension} dimensions")
        
        return {
            'success': True,
            'message': f'Generated {successful} embeddings',
            'total_chunks': len(chunks_data),
            'embeddings_generated': successful,
            'output_path': str(output_path)
        }

def main():
    """Main execution function."""
    try:
        # Initialize generator
        generator = ConsolidatedEmbeddingGenerator()
        
        # Generate consolidated embeddings
        result = generator.generate_consolidated_embeddings()
        
        print("\n" + "=" * 60)
        print("STEP 6: CONSOLIDATED EMBEDDING GENERATION COMPLETE")
        print("=" * 60)
        print(f"Message: {result['message']}")
        print(f"Total chunks: {result['total_chunks']}")
        print(f"Embeddings generated: {result['embeddings_generated']}")
        print("=" * 60)
        
        if result['success']:
            print("✅ Consolidated embedding generation completed successfully!")
            print(f"📁 Output saved to: {result['output_path']}")
        else:
            print("❌ Consolidated embedding generation failed")
            
    except Exception as e:
        logger.error(f"Consolidated embedding generation failed: {e}")
        print(f"❌ Consolidated embedding generation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
