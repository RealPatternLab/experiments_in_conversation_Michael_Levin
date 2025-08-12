#!/usr/bin/env python3
"""
Convert embeddings pickle file to FAISS index format

This script converts the output from step09_generate_vector_embeddings_for_chunks.py
to the FAISS index format expected by the Streamlit app.
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_pickle_to_faiss(pickle_file: Path, output_dir: Path):
    """Convert pickle file to FAISS index format."""
    
    logger.info(f"🔍 Loading embeddings from: {pickle_file}")
    
    # Load the pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"📊 Loaded data with keys: {list(data.keys())}")
    
    # Extract embeddings and metadata
    embeddings = np.array(data['embeddings'], dtype=np.float32)
    chunk_info = data['chunk_info']
    
    logger.info(f"📈 Embeddings shape: {embeddings.shape}")
    logger.info(f"📋 Number of chunks: {len(chunk_info)}")
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    logger.info(f"🔧 Creating FAISS index with dimension: {dimension}")
    
    # Use IndexFlatIP for inner product (cosine similarity when vectors are normalized)
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to the index
    index.add(embeddings)
    
    logger.info(f"✅ FAISS index created with {index.ntotal} vectors")
    
    # Prepare metadata for the app
    metadata = []
    for i, chunk in enumerate(chunk_info):
        metadata.append({
            'chunk_id': i,
            'text': chunk.get('text', ''),
            'section': chunk.get('section', ''),
            'topic': chunk.get('topic', ''),
            'chunk_summary': chunk.get('chunk_summary', ''),
            'position_in_section': chunk.get('position_in_section', ''),
            'certainty_level': chunk.get('certainty_level', ''),
            'citation_context': chunk.get('citation_context', ''),
            'pdf_filename': chunk.get('pdf_filename', ''),
            'original_filename': chunk.get('original_filename', ''),
            'authors': chunk.get('authors', ''),
            'year': chunk.get('year', ''),
            'journal': chunk.get('journal', ''),
            'doi': chunk.get('doi', '')
        })
    
    # Save files in the format expected by the Streamlit app
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = output_dir / "chunks.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"💾 Saved FAISS index to: {index_path}")
    
    # Save embeddings as numpy array
    embeddings_path = output_dir / "chunks_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"💾 Saved embeddings to: {embeddings_path}")
    
    # Save metadata
    metadata_path = output_dir / "chunks_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"💾 Saved metadata to: {metadata_path}")
    
    logger.info(f"🎉 Conversion complete! Created {len(metadata)} chunks in FAISS format")
    logger.info(f"📁 Output directory: {output_dir}")
    
    return True

def main():
    """Main function."""
    # Find the pickle file
    vector_embeddings_dir = Path("data/transformed_data/vector_embeddings")
    pickle_files = list(vector_embeddings_dir.glob("*_embeddings.pkl"))
    
    if not pickle_files:
        logger.error("❌ No embeddings pickle files found!")
        return False
    
    # Use the first pickle file found
    pickle_file = pickle_files[0]
    logger.info(f"📄 Found pickle file: {pickle_file}")
    
    # Convert to FAISS format
    success = convert_pickle_to_faiss(pickle_file, vector_embeddings_dir)
    
    if success:
        logger.info("✅ Successfully converted to FAISS format!")
        logger.info("🚀 You can now test the Streamlit app!")
    else:
        logger.error("❌ Conversion failed!")
    
    return success

if __name__ == "__main__":
    main() 