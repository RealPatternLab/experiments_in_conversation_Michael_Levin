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
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None
    
    def get_chunk_files_to_process(self, max_files: Optional[int] = None) -> List[Path]:
        """Get list of chunk files to process."""
        chunk_files = []
        
        for file_path in self.input_dir.glob("*_chunks.json"):
            if file_path.is_file():
                chunk_files.append(file_path)
        
        # TEMPORARY: Only process files with today's date (for testing)
        today = datetime.now().strftime("%Y%m%d")
        today_files = []
        for chunk_file in chunk_files:
            if today in chunk_file.name:
                today_files.append(chunk_file)
                self.logger.info(f"Found today's file: {chunk_file.name}")
        
        if not today_files:
            self.logger.info(f"No files found with today's date ({today})")
            return []
        
        self.logger.info(f"Found {len(today_files)} files with today's date ({today})")
        
        # Sort by modification time (newest first)
        today_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Apply max_files limit
        if max_files:
            today_files = today_files[:max_files]
        
        # Filter out already processed files
        processed_files = []
        for chunk_file in today_files:
            base_name = chunk_file.stem.replace('_chunks', '')
            embedding_file = self.output_dir / f"{base_name}_embeddings.pkl"
            
            if not embedding_file.exists():
                processed_files.append(chunk_file)
        
        return processed_files
    
    def process_file(self, chunk_file: Path) -> bool:
        """Process a single chunk file."""
        try:
            self.logger.info(f"Processing: {chunk_file.name}")
            
            # Load chunk data
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            if 'chunks' not in chunk_data or not chunk_data['chunks']:
                self.logger.warning(f"No chunks found in {chunk_file.name}")
                return False
            
            chunks = chunk_data['chunks']
            embeddings = []
            
            # Generate embeddings for each chunk
            for chunk in chunks:
                text = chunk.get('text', '')
                if text:
                    embedding = self.generate_embedding(text)
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        self.logger.warning(f"Failed to generate embedding for chunk")
                        return False
            
            if not embeddings:
                self.logger.warning(f"No embeddings generated for {chunk_file.name}")
                return False
            
            # Prepare output data
            output_data = {
                "embeddings": embeddings,
                "metadata": {
                    "filename": chunk_data.get('metadata', {}).get('filename', chunk_file.name),
                    "embedding_model": "text-embedding-ada-002",
                    "chunk_count": len(chunks),
                    "processed_at": datetime.now().isoformat()
                },
                "chunk_info": chunks
            }
            
            # Save embeddings
            base_name = chunk_file.stem.replace('_chunks', '')
            output_file = self.output_dir / f"{base_name}_embeddings.pkl"
            
            with open(output_file, 'wb') as f:
                pickle.dump(output_data, f)
            
            self.logger.info(f"Created embeddings: {output_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {chunk_file.name}: {e}")
            return False
    
    def process_files(self, max_files: Optional[int] = None) -> Dict:
        """Process multiple chunk files."""
        chunk_files = self.get_chunk_files_to_process(max_files)
        
        if not chunk_files:
            self.logger.info("No files to process")
            return {"total": 0, "processed": 0, "failed": 0}
        
        self.logger.info(f"Processing {len(chunk_files)} files...")
        
        processed = 0
        failed = 0
        
        for chunk_file in chunk_files:
            if self.process_file(chunk_file):
                processed += 1
            else:
                failed += 1
        
        results = {
            "total": len(chunk_files),
            "processed": processed,
            "failed": failed
        }
        
        self.logger.info(f"Processing complete: {processed} successful, {failed} failed")
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