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
from typing import List, Dict, Any, Set, Tuple, Optional
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

class VectorEmbedder:
    """Handles vector embedding generation for semantic chunks."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None
    
    def _get_enriched_metadata(self, sanitized_filename: str) -> Optional[Dict[str, Any]]:
        """Look up enriched metadata for a given PDF filename."""
        try:
            # Extract base name from sanitized filename
            base_name = sanitized_filename.replace('.pdf', '')
            
            # Look for enriched metadata file
            enriched_metadata_path = Path("data/transformed_data/metadata_enrichment") / f"{base_name}_quick_metadata_enriched.json"
            
            if enriched_metadata_path.exists():
                with open(enriched_metadata_path, 'r', encoding='utf-8') as f:
                    enriched_data = json.load(f)
                return enriched_data
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to load enriched metadata for {sanitized_filename}: {e}")
            return None
    
    def get_chunk_files_to_process(self, max_files: Optional[int] = None) -> List[Path]:
        """Get list of chunk files to process."""
        chunk_files = []
        
        for file_path in self.input_dir.glob("*_chunks.json"):
            if file_path.is_file():
                chunk_files.append(file_path)
        
        if not chunk_files:
            self.logger.info("No chunk files found to process")
            return []
        
        self.logger.info(f"Found {len(chunk_files)} chunk files to process")
        
        # Sort by modification time (newest first)
        chunk_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Apply max_files limit
        if max_files:
            chunk_files = chunk_files[:max_files]
        
        # Return all chunk files - we'll create a consolidated FAISS index
        # instead of individual embedding files
        return chunk_files
    
    def process_files(self, max_files: Optional[int] = None) -> Dict:
        """Process multiple chunk files and create a consolidated FAISS index."""
        chunk_files = self.get_chunk_files_to_process(max_files)
        
        if not chunk_files:
            self.logger.info("No files to process")
            return {"total": 0, "processed": 0, "failed": 0}
        
        self.logger.info(f"Processing {len(chunk_files)} chunk files to create consolidated FAISS index...")
        
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        
        processed = 0
        failed = 0
        
        # Process each chunk file and collect all chunks and embeddings
        for chunk_file in chunk_files:
            try:
                self.logger.info(f"Loading chunks from: {chunk_file.name}")
                
                # Load chunk data
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                if 'chunks' not in chunk_data or not chunk_data['chunks']:
                    self.logger.warning(f"No chunks found in {chunk_file.name}")
                    failed += 1
                    continue
                
                chunks = chunk_data['chunks']
                self.logger.info(f"Found {len(chunks)} chunks in {chunk_file.name}")
                
                # Generate embeddings for each chunk
                file_embeddings = []
                file_metadata = []
                
                for i, chunk in enumerate(chunks):
                    text = chunk.get('text', '')
                    if text:
                        embedding = self.generate_embedding(text)
                        if embedding:
                            file_embeddings.append(embedding)
                            
                            # Calculate word and character counts
                            word_count = len(text.split()) if text else 0
                            character_count = len(text) if text else 0
                            
                            # Create sanitized filename from original filename
                            original_filename = chunk.get('original_filename', '')
                            pdf_filename = chunk.get('pdf_filename', '')
                            
                            # For PDF citations, we want the actual PDF filename, not the extracted text filename
                            if pdf_filename and pdf_filename.endswith('.pdf'):
                                sanitized_filename = pdf_filename
                            elif original_filename and original_filename.endswith('.pdf'):
                                sanitized_filename = original_filename
                            elif original_filename and original_filename.endswith('_extracted_text.txt'):
                                # Extract the base name and add .pdf extension
                                base_name = original_filename.replace('_extracted_text.txt', '')
                                sanitized_filename = f"{base_name}.pdf"
                            else:
                                # Fallback to original filename
                                sanitized_filename = original_filename if original_filename else pdf_filename
                            
                            # Look up enriched metadata for this file
                            enriched_metadata = self._get_enriched_metadata(sanitized_filename)
                            
                            # Extract title from enriched metadata or filename
                            title = enriched_metadata.get('title') if enriched_metadata else ''
                            if not title and original_filename:
                                # Try to extract title from filename
                                title = original_filename.replace('.pdf', '').replace('_', ' ').title()
                            
                            # Set document type based on available info
                            document_type = enriched_metadata.get('document_type', 'research_paper') if enriched_metadata else 'research_paper'
                            
                            metadata_entry = {
                                'chunk_id': len(all_metadata),
                                'chunk_index': len(all_metadata),  # Same as chunk_id for compatibility
                                'text': text,
                                'section': chunk.get('section', ''),
                                'topic': chunk.get('topic', ''),
                                'chunk_summary': chunk.get('chunk_summary', ''),
                                'position_in_section': chunk.get('position_in_section', ''),
                                'certainty_level': chunk.get('certainty_level', ''),
                                'citation_context': chunk.get('citation_context', ''),
                                'word_count': word_count,
                                'character_count': character_count,
                                'pdf_filename': pdf_filename,
                                'original_filename': original_filename,
                                'sanitized_filename': sanitized_filename,
                                'title': title,
                                'authors': enriched_metadata.get('authors', 'Unknown') if enriched_metadata else 'Unknown',
                                'year': enriched_metadata.get('publication_year', 'Unknown') if enriched_metadata else 'Unknown',
                                'publication_date': enriched_metadata.get('publication_year', 'Unknown') if enriched_metadata else 'Unknown',
                                'journal': enriched_metadata.get('journal', 'Unknown') if enriched_metadata else 'Unknown',
                                'doi': enriched_metadata.get('doi', 'Unknown') if enriched_metadata else 'Unknown',
                                'document_type': document_type,
                                'page_number': chunk.get('page_number', ''),
                                # Add YouTube-specific fields (empty for PDFs)
                                'youtube_url': '',
                                'start_time': None,
                                'end_time': None,
                                'frame_path': '',
                                # Add semantic topics (empty for now)
                                'semantic_topics': {}
                            }
                            
                            file_metadata.append(metadata_entry)
                        else:
                            self.logger.warning(f"Failed to generate embedding for chunk {i} in {chunk_file.name}")
                            failed += 1
                            continue
                    else:
                        self.logger.warning(f"Empty text for chunk {i} in {chunk_file.name}")
                        failed += 1
                        continue
                
                if file_embeddings:
                    all_embeddings.extend(file_embeddings)
                    all_metadata.extend(file_metadata)
                    processed += 1
                    self.logger.info(f"✅ Successfully processed {chunk_file.name} - {len(file_embeddings)} chunks")
                else:
                    self.logger.warning(f"No valid embeddings generated for {chunk_file.name}")
                    failed += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {chunk_file.name}: {e}")
                failed += 1
                continue
        
        if not all_embeddings:
            self.logger.error("No embeddings generated from any files")
            return {"total": len(chunk_files), "processed": 0, "failed": len(chunk_files)}
        
        # Create consolidated FAISS index from all embeddings
        try:
            self.logger.info(f"Creating consolidated FAISS index from {len(all_embeddings)} total chunks...")
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            self.logger.info(f"Generated {len(all_embeddings)} embeddings with shape: {embeddings_array.shape}")
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            self.logger.info(f"Creating FAISS index with dimension: {dimension}")
            
            # Use IndexFlatIP for inner product (cosine similarity when vectors are normalized)
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_array)
            
            self.logger.info(f"FAISS index created with {index.ntotal} vectors")
            
            # Save consolidated files
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = self.output_dir / "chunks.index"
            faiss.write_index(index, str(index_path))
            self.logger.info(f"Saved FAISS index to: {index_path}")
            
            # Save embeddings as numpy array
            embeddings_path = self.output_dir / "chunks_embeddings.npy"
            np.save(embeddings_path, embeddings_array)
            self.logger.info(f"Saved embeddings to: {embeddings_path}")
            
            # Save metadata
            metadata_path = self.output_dir / "chunks_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(all_metadata, f)
            self.logger.info(f"Saved metadata to: {metadata_path}")
            
            self.logger.info(f"🎉 Successfully created consolidated FAISS index with {len(all_embeddings)} chunks from {processed} files")
            
        except Exception as e:
            self.logger.error(f"Error creating consolidated FAISS index: {e}")
            return {"total": len(chunk_files), "processed": 0, "failed": len(chunk_files)}
        
        results = {
            "total": len(chunk_files),
            "processed": processed,
            "failed": failed
        }
        
        self.logger.info(f"📊 Processing complete: {processed} files successfully processed, {failed} files failed")
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate vector embeddings for semantic chunks")
    parser.add_argument("--input-dir", required=True, help="Directory containing chunk files")
    parser.add_argument("--output-dir", required=True, help="Directory to save embedding files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create embedder
    embedder = VectorEmbedder(input_dir, output_dir)
    
    # Process files
    results = embedder.process_files(args.max_files)
    
    if results["failed"] > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main() 