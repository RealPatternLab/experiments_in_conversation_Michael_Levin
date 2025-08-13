#!/usr/bin/env python3
"""
Standalone tool to embed all processed semantic chunks into a FAISS index.

This tool is designed to run independently of the main pipeline, typically on a schedule
(e.g., once per day) to update the search index with all accumulated chunks.

Usage:
    uv run python3 tools/embed_all_processed_chunks.py
    uv run python3 tools/embed_all_processed_chunks.py --input-dir data/transformed_data/semantic_chunks --output-dir data/transformed_data/vector_embeddings
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


class ChunkEmbedder:
    """Embeds all processed semantic chunks into a consolidated FAISS index."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Set OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 Initialized ChunkEmbedder")
        logger.info(f"   Input directory: {self.input_dir}")
        logger.info(f"   Output directory: {self.output_dir}")
    
    def get_chunk_files_to_process(self) -> List[Path]:
        """Get list of all chunk files to process."""
        chunk_files = []
        for file_path in self.input_dir.glob("*_chunks.json"):
            if file_path.is_file():
                chunk_files.append(file_path)
        
        if not chunk_files:
            logger.info("No chunk files found to process")
            return []
        
        logger.info(f"Found {len(chunk_files)} chunk files to process")
        chunk_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return chunk_files
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI API."""
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-large",
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _get_enriched_metadata(self, base_name: str) -> Dict:
        """Load enriched metadata from quick_metadata_enriched.json file."""
        enriched_file = Path("data/transformed_data/metadata_enrichment") / f"{base_name}_quick_metadata_enriched.json"
        
        if enriched_file.exists():
            try:
                with open(enriched_file, 'r', encoding='utf-8') as f:
                    enriched_data = json.load(f)
                
                # Extract metadata from the enriched data
                metadata = {}
                if 'enriched_data' in enriched_data:
                    enriched = enriched_data['enriched_data']
                    metadata.update({
                        'title': enriched.get('title'),
                        'authors': enriched.get('authors'),
                        'year': enriched.get('year'),
                        'publication_date': enriched.get('publication_date'),
                        'journal': enriched.get('journal'),
                        'doi': enriched.get('doi'),
                        'document_type': enriched.get('document_type')
                    })
                
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load enriched metadata from {enriched_file}: {e}")
        
        return {}
    
    def process_files(self) -> Dict:
        """Process all chunk files and create a consolidated FAISS index."""
        chunk_files = self.get_chunk_files_to_process()
        if not chunk_files:
            logger.info("No files to process")
            return {"total": 0, "processed": 0, "failed": 0}
        
        logger.info(f"Processing {len(chunk_files)} chunk files to create consolidated FAISS index...")
        
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        processed = 0
        failed = 0
        
        for chunk_file in chunk_files:
            try:
                logger.info(f"Loading chunks from: {chunk_file.name}")
                
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                if 'chunks' not in chunk_data or not chunk_data['chunks']:
                    logger.warning(f"No chunks found in {chunk_file.name}")
                    failed += 1
                    continue
                
                chunks = chunk_data['chunks']
                logger.info(f"Found {len(chunks)} chunks in {chunk_file.name}")
                
                file_embeddings = []
                file_metadata = []
                
                for i, chunk in enumerate(chunks):
                    text = chunk.get('text', '')
                    if text:
                        embedding = self.generate_embedding(text)
                        if embedding:
                            file_embeddings.append(embedding)
                            
                            # Extract base metadata from chunk
                            base_name = chunk.get('base_name', '')
                            enriched_metadata = self._get_enriched_metadata(base_name)
                            
                            # Create comprehensive metadata entry
                            metadata_entry = {
                                'chunk_index': len(all_metadata),
                                'text': text,
                                'word_count': len(text.split()),
                                'character_count': len(text),
                                'sanitized_filename': f"{base_name}.pdf",
                                'base_name': base_name,
                                'chunk_id': chunk.get('chunk_id', f"{base_name}_chunk_{i}"),
                                'page_number': chunk.get('page_number'),
                                'title': enriched_metadata.get('title'),
                                'authors': enriched_metadata.get('authors'),
                                'year': enriched_metadata.get('year'),
                                'publication_date': enriched_metadata.get('publication_date'),
                                'journal': enriched_metadata.get('journal'),
                                'doi': enriched_metadata.get('doi'),
                                'document_type': enriched_metadata.get('document_type'),
                                # YouTube-specific fields (empty for PDFs)
                                'video_id': None,
                                'video_title': None,
                                'video_description': None,
                                'video_upload_date': None,
                                'video_duration': None,
                                'video_channel': None,
                                'video_tags': None,
                                'video_category': None,
                                'video_view_count': None,
                                'video_like_count': None,
                                'video_comment_count': None
                            }
                            
                            file_metadata.append(metadata_entry)
                        else:
                            logger.warning(f"Failed to generate embedding for chunk {i} in {chunk_file.name}")
                            failed += 1
                            continue
                    else:
                        logger.warning(f"Empty text for chunk {i} in {chunk_file.name}")
                        failed += 1
                        continue
                
                if file_embeddings:
                    all_embeddings.extend(file_embeddings)
                    all_metadata.extend(file_metadata)
                    processed += 1
                    logger.info(f"✅ Successfully processed {chunk_file.name} - {len(file_embeddings)} chunks")
                else:
                    logger.warning(f"No valid embeddings generated for {chunk_file.name}")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing {chunk_file.name}: {e}")
                failed += 1
                continue
        
        if not all_embeddings:
            logger.error("No embeddings generated from any files")
            return {"total": len(chunk_files), "processed": 0, "failed": len(chunk_files)}
        
        try:
            logger.info(f"Creating consolidated FAISS index from {len(all_embeddings)} total chunks...")
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            logger.info(f"Generated {len(all_embeddings)} embeddings with shape: {embeddings_array.shape}")
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            logger.info(f"Creating FAISS index with dimension: {dimension}")
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_array)
            logger.info(f"FAISS index created with {index.ntotal} vectors")
            
            # Save FAISS index
            index_path = self.output_dir / "chunks.index"
            faiss.write_index(index, str(index_path))
            logger.info(f"Saved FAISS index to: {index_path}")
            
            # Save embeddings array
            embeddings_path = self.output_dir / "chunks_embeddings.npy"
            np.save(embeddings_path, embeddings_array)
            logger.info(f"Saved embeddings to: {embeddings_path}")
            
            # Save metadata
            metadata_path = self.output_dir / "chunks_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(all_metadata, f)
            logger.info(f"Saved metadata to: {metadata_path}")
            
            logger.info(f"🎉 Successfully created consolidated FAISS index with {len(all_embeddings)} chunks from {processed} files")
            
        except Exception as e:
            logger.error(f"Error creating consolidated FAISS index: {e}")
            return {"total": len(chunk_files), "processed": 0, "failed": len(chunk_files)}
        
        results = {
            "total": len(chunk_files),
            "processed": processed,
            "failed": failed
        }
        
        logger.info(f"📊 Processing complete: {processed} files successfully processed, {failed} files failed")
        return results


def main():
    """Main function to run the chunk embedding process."""
    parser = argparse.ArgumentParser(
        description="Embed all processed semantic chunks into a FAISS index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default directories
    uv run python3 tools/embed_all_processed_chunks.py
    
    # Specify custom directories
    uv run python3 tools/embed_all_processed_chunks.py \\
        --input-dir data/transformed_data/semantic_chunks \\
        --output-dir data/transformed_data/vector_embeddings
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/transformed_data/semantic_chunks",
        help="Directory containing chunk files (default: data/transformed_data/semantic_chunks)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="data/transformed_data/vector_embeddings",
        help="Directory to save FAISS index files (default: data/transformed_data/vector_embeddings)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logger.info("🚀 Starting standalone chunk embedding process...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        embedder = ChunkEmbedder(input_dir, args.output_dir)
        results = embedder.process_files()
        
        if results["processed"] > 0:
            logger.info("✅ Embedding process completed successfully!")
            logger.info(f"📊 Summary: {results['processed']} files processed, {results['failed']} files failed")
            sys.exit(0)
        else:
            logger.error("❌ No files were successfully processed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Embedding process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 