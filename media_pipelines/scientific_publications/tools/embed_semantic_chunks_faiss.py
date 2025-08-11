#!/usr/bin/env python3
"""
FAISS Semantic Chunks Embedding Tool

This tool creates FAISS indices from semantically chunked text data.
It embeds text chunks using OpenAI's text-embedding-3-large and stores them in FAISS format.
Supports incremental processing to avoid re-embedding already processed chunks.

Usage:
    python tools/embed_semantic_chunks_faiss.py --input-dir data/semantically-chunked --output-dir data/faiss
    python tools/embed_semantic_chunks_faiss.py --input-dir data/semantically-chunked --output-dir data/faiss --use-sqlite
"""

import argparse
import json
import pickle
import sys
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import logging
import numpy as np
import faiss
from tqdm import tqdm

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: pip install openai faiss-cpu python-dotenv tqdm")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
import os
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / 'embed_semantic_chunks_faiss.log')
    ]
)
logger = logging.getLogger(__name__)

def compute_chunk_hash(chunk: Dict[str, Any]) -> str:
    """Compute a hash for a chunk to track if it has been processed."""
    # Create a unique identifier based on key chunk properties
    chunk_key = f"{chunk.get('original_filename', '')}_{chunk.get('chunk_index', '')}_{chunk.get('text', '')[:100]}"
    return hashlib.md5(chunk_key.encode()).hexdigest()

def load_processed_chunks_sqlite(output_dir: Path) -> Tuple[Set[str], List[Dict], np.ndarray]:
    """Load already processed chunks using SQLite database."""
    processed_hashes = set()
    existing_metadata = []
    existing_embeddings = None
    
    # SQLite database path
    db_path = output_dir / "processed_chunks.db"
    
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_chunks (
                    chunk_hash TEXT PRIMARY KEY,
                    original_filename TEXT,
                    chunk_index INTEGER,
                    title TEXT,
                    authors TEXT,
                    section TEXT,
                    document_type TEXT,
                    text_preview TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Get all processed hashes
            cursor.execute("SELECT chunk_hash FROM processed_chunks")
            processed_hashes = {row[0] for row in cursor.fetchall()}
            
            conn.close()
            logger.info(f"📋 Loaded {len(processed_hashes)} already processed chunks from SQLite")
            
        except Exception as e:
            logger.warning(f"Could not load processed chunks from SQLite: {e}")
    
    # Load existing metadata and embeddings
    metadata_file = output_dir / "chunks_metadata.pkl"
    embeddings_file = output_dir / "chunks_embeddings.npy"
    
    if metadata_file.exists() and embeddings_file.exists():
        try:
            with open(metadata_file, 'rb') as f:
                existing_metadata = pickle.load(f)
            existing_embeddings = np.load(str(embeddings_file))
            logger.info(f"📊 Loaded existing index with {len(existing_metadata)} chunks")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    
    return processed_hashes, existing_metadata, existing_embeddings

def save_processed_chunks_sqlite(output_dir: Path, new_chunks: List[Dict], new_hashes: List[str]):
    """Save the set of processed chunk hashes to SQLite database."""
    db_path = output_dir / "processed_chunks.db"
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_chunks (
                chunk_hash TEXT PRIMARY KEY,
                original_filename TEXT,
                chunk_index INTEGER,
                title TEXT,
                authors TEXT,
                section TEXT,
                document_type TEXT,
                text_preview TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert new chunks
        for chunk, chunk_hash in zip(new_chunks, new_hashes):
            cursor.execute("""
                INSERT OR REPLACE INTO processed_chunks 
                (chunk_hash, original_filename, chunk_index, title, authors, section, document_type, text_preview)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk_hash,
                chunk.get('original_filename', 'Unknown'),
                chunk.get('chunk_index', 0),
                chunk.get('title', 'Unknown'),
                chunk.get('authors', 'Unknown'),
                chunk.get('section', 'Unknown'),
                chunk.get('document_type', 'unknown'),
                chunk.get('text', '')[:100]  # First 100 chars as preview
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"💾 Saved {len(new_chunks)} processed chunk hashes to SQLite")
        
    except Exception as e:
        logger.error(f"Failed to save processed chunks to SQLite: {e}")

def load_processed_chunks_file(output_dir: Path) -> Tuple[Set[str], List[Dict], np.ndarray]:
    """Load already processed chunks using JSON file (original method)."""
    processed_hashes = set()
    existing_metadata = []
    existing_embeddings = None
    
    # Load processed hashes
    hash_file = output_dir / "processed_chunks.json"
    if hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                processed_hashes = set(json.load(f))
            logger.info(f"📋 Loaded {len(processed_hashes)} already processed chunks from JSON")
        except Exception as e:
            logger.warning(f"Could not load processed chunks file: {e}")
    
    # Load existing metadata and embeddings
    metadata_file = output_dir / "chunks_metadata.pkl"
    embeddings_file = output_dir / "chunks_embeddings.npy"
    
    if metadata_file.exists() and embeddings_file.exists():
        try:
            with open(metadata_file, 'rb') as f:
                existing_metadata = pickle.load(f)
            existing_embeddings = np.load(str(embeddings_file))
            logger.info(f"📊 Loaded existing index with {len(existing_metadata)} chunks")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
    
    return processed_hashes, existing_metadata, existing_embeddings

def save_processed_chunks_file(output_dir: Path, processed_hashes: Set[str]):
    """Save the set of processed chunk hashes to JSON file (original method)."""
    hash_file = output_dir / "processed_chunks.json"
    try:
        with open(hash_file, 'w') as f:
            json.dump(list(processed_hashes), f)
        logger.info(f"💾 Saved {len(processed_hashes)} processed chunk hashes to JSON")
    except Exception as e:
        logger.error(f"Failed to save processed chunks to JSON: {e}")

def load_processed_chunks(output_dir: Path, use_sqlite: bool = False) -> Tuple[Set[str], List[Dict], np.ndarray]:
    """Load already processed chunks using the specified method."""
    if use_sqlite:
        return load_processed_chunks_sqlite(output_dir)
    else:
        return load_processed_chunks_file(output_dir)

def save_processed_chunks(output_dir: Path, processed_hashes: Set[str], new_chunks: List[Dict] = None, new_hashes: List[str] = None, use_sqlite: bool = False):
    """Save the set of processed chunk hashes using the specified method."""
    if use_sqlite and new_chunks and new_hashes:
        save_processed_chunks_sqlite(output_dir, new_chunks, new_hashes)
    else:
        save_processed_chunks_file(output_dir, processed_hashes)

def load_chunk_data(input_dir: Path) -> List[Dict[str, Any]]:
    """Load chunk data from JSON files."""
    chunks = []
    
    # Find all JSON files in the input directory
    json_files = list(input_dir.glob("**/*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return chunks
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    for json_file in tqdm(json_files, desc="Loading chunk files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                
            # Handle new format with file_metadata and chunks array
            if isinstance(file_data, dict) and 'chunks' in file_data:
                # New format: {file_metadata: {...}, chunks: [...]}
                file_chunks = file_data['chunks']
                if isinstance(file_chunks, list):
                    chunks.extend(file_chunks)
                else:
                    logger.warning(f"Unexpected chunks format in {json_file}")
            elif isinstance(file_data, list):
                # Old format: direct list of chunks
                chunks.extend(file_data)
            else:
                logger.warning(f"Unexpected format in {json_file}")
                
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(chunks)} total chunks")
    return chunks

def create_faiss_index_incremental(chunks: List[Dict[str, Any]], 
                                  model_name: str = "text-embedding-3-large",
                                  output_dir: Path = None,
                                  use_sqlite: bool = False) -> None:
    """Create FAISS index from chunks with incremental processing."""
    
    if not chunks:
        logger.error("No chunks to process")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load already processed chunks
    processed_hashes, existing_metadata, existing_embeddings = load_processed_chunks(output_dir, use_sqlite)
    
    # Separate new chunks from already processed ones
    new_chunks = []
    new_hashes = []
    
    logger.info("Checking for new chunks...")
    for chunk in tqdm(chunks, desc="Checking chunks"):
        chunk_hash = compute_chunk_hash(chunk)
        if chunk_hash not in processed_hashes:
            new_chunks.append(chunk)
            new_hashes.append(chunk_hash)
    
    logger.info(f"📊 Found {len(new_chunks)} new chunks out of {len(chunks)} total")
    
    if not new_chunks:
        logger.info("✅ All chunks already processed!")
        return
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    client = OpenAI(api_key=api_key)
    logger.info(f"🤖 Using OpenAI model: {model_name}")
    
    # Prepare new data for FAISS
    new_texts = []
    new_metadata = []
    
    logger.info("Preparing new chunks for embedding...")
    for i, chunk in enumerate(tqdm(new_chunks, desc="Preparing new chunks")):
        # Extract text content
        text = chunk.get('text', '')
        if not text.strip():
            continue
            
        new_texts.append(text)
        
        # Store metadata
        chunk_metadata = {
            'chunk_id': len(existing_metadata) + i,
            'chunk_index': chunk.get('chunk_index', len(existing_metadata) + i),
            'text': text,
            'section': chunk.get('section', 'Unknown'),
            'topic': chunk.get('topic', 'Unknown'),
            'chunk_summary': chunk.get('chunk_summary', ''),
            'position_in_section': chunk.get('position_in_section', 'Unknown'),
            'certainty_level': chunk.get('certainty_level', 'Unknown'),
            'citation_context': chunk.get('citation_context', ''),
            'word_count': chunk.get('word_count', 0),
            'character_count': chunk.get('character_count', 0),
            'original_filename': chunk.get('original_filename', 'Unknown'),
            'sanitized_filename': chunk.get('pdf_filename', 'Unknown'),  # Use pdf_filename from JSON
            'title': chunk.get('title', 'Unknown'),
            'authors': chunk.get('authors', 'Unknown'),
            'doi': chunk.get('doi', 'Unknown'),
            'publication_date': chunk.get('publication_date', 'Unknown'),
            'journal': chunk.get('journal', 'Unknown'),
            'document_type': chunk.get('document_type', 'unknown')
        }
        new_metadata.append(chunk_metadata)
    
    if not new_texts:
        logger.info("✅ No new valid texts to process")
        return
    
    logger.info(f"✅ Prepared {len(new_texts)} new texts for embedding")
    
    # Create embeddings for new chunks using OpenAI
    logger.info("Creating embeddings for new chunks...")
    new_embeddings = []
    
    # Process in batches to avoid rate limits
    batch_size = 100
    for i in tqdm(range(0, len(new_texts), batch_size), desc="Embedding chunks"):
        batch_texts = new_texts[i:i + batch_size]
        
        try:
            response = client.embeddings.create(
                input=batch_texts,
                model=model_name
            )
            
            # Extract embeddings from response
            batch_embeddings = [data.embedding for data in response.data]
            new_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            logger.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
            # Continue with next batch
            continue
    
    if not new_embeddings:
        logger.error("❌ No embeddings were created")
        return
    
    new_embeddings = np.array(new_embeddings).astype('float32')
    embedding_dimension = new_embeddings.shape[1]
    logger.info(f"✅ Created embeddings with dimension: {embedding_dimension}")
    
    # Combine with existing data
    if existing_embeddings is not None:
        logger.info("Combining with existing embeddings...")
        all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        all_metadata = existing_metadata + new_metadata
    else:
        logger.info("Creating new index...")
        all_embeddings = new_embeddings
        all_metadata = new_metadata
    
    # Create FAISS index
    logger.info("Creating FAISS index...")
    dimension = all_embeddings.shape[1]
    
    # Use IndexFlatIP for inner product (cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index
    index.add(all_embeddings)
    
    logger.info(f"✅ Created FAISS index with {index.ntotal} vectors")
    
    # Save index and metadata
    # Save FAISS index
    index_path = output_dir / "chunks.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"💾 Saved FAISS index to {index_path}")
    
    # Save embeddings as numpy file (for faster pre-search filtering)
    embeddings_path = output_dir / "chunks_embeddings.npy"
    np.save(str(embeddings_path), all_embeddings)
    logger.info(f"💾 Saved embeddings to {embeddings_path}")
    
    # Save metadata
    metadata_path = output_dir / "chunks_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_metadata, f)
    logger.info(f"💾 Saved metadata to {metadata_path}")
    
    # Update processed hashes
    processed_hashes.update(new_hashes)
    save_processed_chunks(output_dir, processed_hashes, new_chunks, new_hashes, use_sqlite)
    
    # Save some statistics
    stats = {
        'total_chunks': len(all_metadata),
        'new_chunks_processed': len(new_chunks),
        'existing_chunks': len(existing_metadata),
        'embedding_dimension': dimension,
        'model_name': model_name,
        'index_type': 'IndexFlatIP',
        'has_embeddings_file': True,
        'last_updated': str(Path.cwd()),
        'tracking_method': 'sqlite' if use_sqlite else 'json'
    }
    
    stats_path = output_dir / "index_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"💾 Saved index statistics to {stats_path}")
    
    logger.info(f"🎉 FAISS index creation complete!")
    logger.info(f"   Total chunks: {len(all_metadata)}")
    logger.info(f"   New chunks processed: {len(new_chunks)}")
    logger.info(f"   Embedding dimension: {dimension}")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Tracking method: {'SQLite' if use_sqlite else 'JSON'}")
    logger.info(f"   Files created: index, embeddings, metadata, stats, processed_hashes")

def create_faiss_index(chunks: List[Dict[str, Any]], 
                       model_name: str = "text-embedding-3-large",
                       output_dir: Path = None,
                       use_sqlite: bool = False) -> None:
    """Create FAISS index from chunks (legacy function for backward compatibility)."""
    create_faiss_index_incremental(chunks, model_name, output_dir, use_sqlite)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create FAISS index from semantically chunked text data"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/transformed_data/semantic_chunking"),
        help="Directory containing chunk JSON files (default: data/transformed_data/semantic_chunking)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/transformed_data/vector_embeddings"),
        help="Directory to save FAISS index and metadata (default: data/transformed_data/vector_embeddings)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="text-embedding-3-large",
        help="OpenAI embedding model to use for embeddings"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force re-processing of all chunks (ignore existing processed chunks)"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show statistics about the created index"
    )
    parser.add_argument(
        "--use-sqlite",
        action="store_true",
        help="Use SQLite database for tracking processed chunks instead of JSON files"
    )
    
    args = parser.parse_args()
    
    try:
        # Load chunk data
        logger.info(f"📂 Loading chunks from: {args.input_dir}")
        chunks = load_chunk_data(args.input_dir)
        
        if not chunks:
            logger.error("No chunks found. Please run the chunking tool first.")
            sys.exit(1)
        
        # Create FAISS index
        logger.info(f"🔧 Creating FAISS index...")
        if args.force_reprocess:
            logger.info("🔄 Force re-processing enabled - will process all chunks")
            # Clear existing processed chunks tracking
            processed_file = args.output_dir / "processed_chunks.json"
            if processed_file.exists():
                processed_file.unlink()
                logger.info("🗑️  Cleared existing processed chunks tracking")
        
        create_faiss_index_incremental(chunks, args.model_name, args.output_dir, args.use_sqlite)
        
        # Show stats if requested
        if args.show_stats:
            stats_path = args.output_dir / "index_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                print("\n📊 FAISS Index Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
        
    except Exception as e:
        logger.error(f"❌ Error during FAISS index creation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 