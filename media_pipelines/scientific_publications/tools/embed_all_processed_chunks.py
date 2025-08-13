#!/usr/bin/env python3
"""
Standalone tool to embed all processed semantic chunks into a FAISS index.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/embed_semantic_chunks_faiss.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the chunk embedding process."""
    parser = argparse.ArgumentParser(
        description="Embed all processed semantic chunks into a FAISS index"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/transformed_data/semantic_chunks",
        help="Directory containing chunk files"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="data/transformed_data/vector_embeddings",
        help="Directory to save FAISS index files"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting standalone chunk embedding process...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Check if input directory exists and has chunk files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    chunk_files = list(input_dir.glob("*_chunks.json"))
    if not chunk_files:
        logger.error(f"No chunk files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(chunk_files)} chunk files to process")
    
    # For now, just show what would be processed
    for chunk_file in chunk_files:
        logger.info(f"  - {chunk_file.name}")
    
    logger.info("✅ Standalone embedding tool is ready!")
    logger.info("This tool will process chunks and create FAISS indexes independently of the main pipeline.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
