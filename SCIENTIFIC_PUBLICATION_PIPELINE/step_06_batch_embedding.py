#!/usr/bin/env python3
"""
Step 6: FAISS Vector Embedding Generator

This script generates FAISS vector embeddings from semantic chunks using OpenAI's text-embedding-3-large.
It reads the progress queue and processes documents ready for embedding, creating a consolidated
FAISS index with comprehensive metadata for RAG system queries.

Usage:
    uv run python step_06_batch_embedding.py                    # Process all ready documents
    uv run python step_06_batch_embedding.py --batch-size 10   # Process in batches of 10
    uv run python step_06_batch_embedding.py --force-run       # Force run even if no documents ready
    uv run python step_06_batch_embedding.py --status-only     # Show pipeline status only
"""

import argparse
import json
import logging
import pickle
import hashlib
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sys

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

# Progress queue integration
try:
    from pipeline_progress_queue import PipelineProgressQueue
    PROGRESS_QUEUE_AVAILABLE = True
except ImportError:
    PROGRESS_QUEUE_AVAILABLE = False
    logging.warning("Progress queue not available. Pipeline tracking will be limited.")

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
        logging.FileHandler('step_06_batch_embedding.log')
    ]
)
logger = logging.getLogger(__name__)

class BatchEmbeddingTrigger:
    """Generates FAISS vector embeddings from semantic chunks."""
    
    def __init__(self, output_dir: Path = Path("step_06_faiss_embeddings")):
        """Initialize the batch embedding trigger."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = None
        if PROGRESS_QUEUE_AVAILABLE:
            try:
                self.progress_queue = PipelineProgressQueue()
                logger.info("Progress queue initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize progress queue: {e}")
                self.progress_queue = None
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE and ENV_AVAILABLE:
            try:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found in environment variables")
                else:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Initialize FAISS
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available. Cannot perform embeddings.")
            raise RuntimeError("FAISS is required for embedding generation")
    
    def get_ready_documents(self) -> List[str]:
        """Get list of document IDs ready for embedding."""
        if not self.progress_queue:
            logger.error("Progress queue not available")
            return []
        
        try:
            ready_docs = self.progress_queue.get_ready_for_embedding()
            logger.info(f"Found {len(ready_docs)} documents ready for embedding")
            return ready_docs
        except Exception as e:
            logger.error(f"Failed to get ready documents: {e}")
            return []
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire pipeline status."""
        if not self.progress_queue:
            return {}
        
        try:
            return self.progress_queue.get_pipeline_summary()
        except Exception as e:
            logger.error(f"Failed to get pipeline summary: {e}")
            return {}
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI API."""
        if not self.openai_client:
            logger.error("OpenAI client not available")
            return None
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def load_chunks_for_document(self, doc_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Load semantic chunks and metadata for a specific document."""
        try:
            # Get document status
            doc_status = self.progress_queue.get_document_status(doc_id)
            if not doc_status:
                logger.error(f"Document {doc_id} not found in progress queue")
                return [], []
            
            # Find the chunks file
            new_filename = doc_status.get('new_filename', '')
            if not new_filename:
                logger.error(f"No filename found for document {doc_id}")
                return [], []
            
            # Construct paths
            base_name = new_filename.replace('.pdf', '')
            chunks_file = Path("step_05_semantic_chunks") / f"{base_name}_chunks.json"
            metadata_file = Path("step_02_metadata") / f"{base_name}_metadata.json"
            
            # Load chunks
            if not chunks_file.exists():
                logger.error(f"Chunks file not found: {chunks_file}")
                return [], []
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            chunks = chunks_data.get('chunks', [])
            if not chunks:
                logger.error(f"No chunks found in {chunks_file}")
                return [], []
            
            # Load metadata from file and combine with progress queue data
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Add progress queue data to metadata
            metadata.update({
                'new_filename': new_filename,
                'document_id': doc_id,
                'chunk_count': doc_status.get('chunk_count', 0),
                'text_length': doc_status.get('text_length', 0)
            })
            
            logger.info(f"Loaded {len(chunks)} chunks for document {doc_id}")
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Error loading chunks for document {doc_id}: {e}")
            return [], []
    
    def create_enhanced_metadata(self, chunk: Dict, doc_metadata: Dict, chunk_id: int) -> Dict[str, Any]:
        """Create enhanced metadata for a chunk with comprehensive information."""
        # Extract text statistics
        text = chunk.get('text', '')
        word_count = len(text.split()) if text else 0
        character_count = len(text) if text else 0
        
        # Create enhanced metadata
        enhanced_metadata = {
            'chunk_id': chunk_id,
            'chunk_index': chunk_id,
            'text': text,
            'section': chunk.get('section', ''),
            'primary_topic': chunk.get('primary_topic', ''),
            'secondary_topics': chunk.get('secondary_topics', []),
            'chunk_summary': chunk.get('chunk_summary', ''),
            'position_in_section': chunk.get('position_in_section', ''),
            'certainty_level': chunk.get('certainty_level', ''),
            'citation_context': chunk.get('citation_context', ''),
            'page_number': chunk.get('page_number', ''),
            'word_count': word_count,
            'character_count': character_count,
            
            # Document-level metadata
            'pdf_filename': doc_metadata.get('new_filename', ''),
            'title': doc_metadata.get('title', ''),
            'authors': doc_metadata.get('authors', []),
            'publication_year': doc_metadata.get('publication_year', ''),
            'journal': doc_metadata.get('journal', ''),
            'doi': doc_metadata.get('doi', ''),
            'abstract': doc_metadata.get('abstract', ''),
            'document_type': doc_metadata.get('document_type', 'research_paper'),
            
            # Pipeline metadata
            'pipeline_step': 'step_06_embedding',
            'created_at': datetime.now().isoformat(),
            'embedding_model': 'text-embedding-3-large',
            
            # RAG-specific fields
            'search_keywords': self._extract_search_keywords(chunk, doc_metadata),
            'semantic_context': self._create_semantic_context(chunk, doc_metadata),
            'citation_ready': self._is_citation_ready(chunk, doc_metadata)
        }
        
        return enhanced_metadata
    
    def _extract_search_keywords(self, chunk: Dict, doc_metadata: Dict) -> List[str]:
        """Extract search keywords for RAG queries."""
        keywords = []
        
        # Add primary and secondary topics
        if chunk.get('primary_topic'):
            keywords.append(chunk['primary_topic'])
        if chunk.get('secondary_topics'):
            keywords.extend(chunk['secondary_topics'])
        
        # Add section information
        if chunk.get('section'):
            keywords.append(chunk['section'])
        
        # Add author names
        authors = doc_metadata.get('authors', [])
        if authors:
            keywords.extend(authors)
        
        # Add journal name
        if doc_metadata.get('journal'):
            keywords.append(doc_metadata['journal'])
        
        return list(set(keywords))  # Remove duplicates
    
    def _create_semantic_context(self, chunk: Dict, doc_metadata: Dict) -> Dict[str, Any]:
        """Create semantic context for the chunk."""
        return {
            'research_area': doc_metadata.get('document_type', 'research_paper'),
            'methodology': chunk.get('section', ''),
            'findings': chunk.get('chunk_summary', ''),
            'confidence': chunk.get('certainty_level', ''),
            'temporal_context': doc_metadata.get('publication_year', ''),
            'geographic_context': 'N/A',  # Could be enhanced later
            'institutional_context': 'N/A'  # Could be enhanced later
        }
    
    def _is_citation_ready(self, chunk: Dict, doc_metadata: Dict) -> bool:
        """Check if chunk has sufficient information for citation."""
        required_fields = ['title', 'authors', 'publication_year', 'journal']
        return all(doc_metadata.get(field) for field in required_fields)
    
    def mark_embedding_complete(self, doc_id: str, chunk_count: int = 0, additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """Mark embedding as complete for a document."""
        if not self.progress_queue:
            return False
        
        try:
            return self.progress_queue.mark_embedding_complete(doc_id, chunk_count, additional_info)
        except Exception as e:
            logger.error(f"Failed to mark embedding complete: {e}")
            return False
    
    def process_embedding_batch(self, batch_size: int = None, force_run: bool = False) -> Dict[str, Any]:
        """Process a batch of documents ready for embedding."""
        logger.info("🚀 Starting manual batch embedding process...")
        
        # Get pipeline summary
        summary = self.get_pipeline_summary()
        if summary:
            logger.info("📊 Current Pipeline Status:")
            logger.info(f"   Total documents: {summary.get('total_documents', 0)}")
            logger.info(f"   Ready for embedding: {summary.get('ready_for_embedding', 0)}")
            logger.info(f"   Processing: {summary.get('processing', 0)}")
            logger.info(f"   Pending: {summary.get('pending', 0)}")
            logger.info(f"   Failed: {summary.get('failed', 0)}")
        
        # Get documents ready for embedding
        ready_docs = self.get_ready_documents()
        
        if not ready_docs and not force_run:
            logger.info("✅ No documents ready for embedding")
            logger.info("💡 Use --force-run to process documents anyway")
            return {
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'total_chunks': 0,
                'message': 'No documents ready for embedding'
            }
        
        # If force run is requested, get all documents regardless of status
        if force_run and not ready_docs:
            logger.info("🔄 Force run: Processing all documents regardless of status")
            all_docs = list(self.progress_queue._load_queue_data()["pipeline_progress"].keys())
            ready_docs = all_docs
        
        # Apply batch size limit if specified
        if batch_size and len(ready_docs) > batch_size:
            logger.info(f"📦 Processing batch of {batch_size} documents (out of {len(ready_docs)} ready)")
            ready_docs = ready_docs[:batch_size]
        else:
            logger.info(f"📦 Processing all {len(ready_docs)} documents")
        
        # Process documents
        processed = 0
        successful = 0
        failed = 0
        total_chunks = 0
        
        for doc_id in ready_docs:
            try:
                logger.info(f"🔄 Processing document: {doc_id}")
                
                # Get document status
                doc_status = self.progress_queue.get_document_status(doc_id)
                if not doc_status:
                    logger.error(f"❌ Failed to get status for {doc_id}")
                    failed += 1
                    continue
                
                # Load chunks and metadata for this document
                chunks, doc_metadata = self.load_chunks_for_document(doc_id)
                if not chunks:
                    logger.error(f"❌ No chunks loaded for {doc_id}")
                    failed += 1
                    continue
                
                chunk_count = len(chunks)
                logger.info(f"📄 Document has {chunk_count} chunks to embed")
                
                # Generate embeddings for all chunks
                embeddings = []
                enhanced_metadata = []
                
                logger.info(f"🔮 Generating embeddings for {chunk_count} chunks...")
                
                for i, chunk in enumerate(chunks):
                    try:
                        # Generate embedding
                        text = chunk.get('text', '')
                        if not text:
                            logger.warning(f"Empty text for chunk {i} in {doc_id}")
                            continue
                        
                        embedding = self.generate_embedding(text)
                        if embedding:
                            embeddings.append(embedding)
                            
                            # Create enhanced metadata
                            chunk_metadata = self.create_enhanced_metadata(chunk, doc_metadata, i)
                            enhanced_metadata.append(chunk_metadata)
                            
                            logger.info(f"✅ Generated embedding for chunk {i+1}/{chunk_count}")
                        else:
                            logger.error(f"❌ Failed to generate embedding for chunk {i} in {doc_id}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"❌ Error processing chunk {i} in {doc_id}: {e}")
                        continue
                
                if not embeddings:
                    logger.error(f"❌ No embeddings generated for {doc_id}")
                    failed += 1
                    continue
                
                # Save embeddings and metadata for this document
                try:
                    # Convert to numpy arrays
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    
                    # Generate timestamp for versioning
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove microseconds, keep milliseconds
                    
                    # Save document-specific files with just the timestamp as directory name
                    doc_output_dir = self.output_dir / timestamp
                    doc_output_dir.mkdir(exist_ok=True)
                    
                    # Save FAISS index for this document
                    index_path = doc_output_dir / "chunks.index"
                    dimension = embeddings_array.shape[1]
                    index = faiss.IndexFlatIP(dimension)
                    index.add(embeddings_array)
                    faiss.write_index(index, str(index_path))
                    
                    # Save embeddings as numpy array
                    embeddings_path = doc_output_dir / "chunks_embeddings.npy"
                    np.save(embeddings_path, embeddings_array)
                    
                    # Save enhanced metadata
                    metadata_path = doc_output_dir / "chunks_metadata.pkl"
                    with open(metadata_path, 'wb') as f:
                        pickle.dump(enhanced_metadata, f)
                    
                    # Save timestamp info for reference
                    timestamp_info = {
                        'embedding_timestamp': timestamp,
                        'embedding_datetime': datetime.now().isoformat(),
                        'document_id': doc_id,
                        'original_filename': doc_metadata.get('new_filename', 'Unknown'),
                        'chunk_count': chunk_count,
                        'embedding_model': 'text-embedding-3-large',
                        'faiss_index_type': 'IndexFlatIP',
                        'embedding_dimension': dimension
                    }
                    
                    timestamp_path = doc_output_dir / "embedding_info.json"
                    with open(timestamp_path, 'w', encoding='utf-8') as f:
                        json.dump(timestamp_info, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"💾 Saved embeddings for {doc_id} to {doc_output_dir} (timestamp: {timestamp})")
                    
                    # Mark as complete with timestamp info
                    if self.mark_embedding_complete(doc_id, chunk_count, {"embedding_timestamp": timestamp}):
                        successful += 1
                        total_chunks += chunk_count
                        logger.info(f"✅ Successfully processed {doc_id} ({chunk_count} chunks) with timestamp {timestamp}")
                    else:
                        failed += 1
                        logger.error(f"❌ Failed to mark {doc_id} as complete")
                        
                except Exception as e:
                    logger.error(f"❌ Error saving embeddings for {doc_id}: {e}")
                    failed += 1
                
                processed += 1
                
            except Exception as e:
                failed += 1
                logger.error(f"❌ Error processing {doc_id}: {e}")
        
        # Summary
        results = {
            'processed': processed,
            'successful': successful,
            'failed': failed,
            'total_chunks': total_chunks,
            'message': f'Processed {processed} documents'
        }
        
        logger.info("📋 Batch Embedding Summary:")
        logger.info(f"   Documents processed: {processed}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total chunks: {total_chunks}")
        
        return results

def main():
    """Main function to run the batch embedding trigger."""
    parser = argparse.ArgumentParser(description="Step 6: Manual Batch Embedding Trigger")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Process documents in batches of specified size"
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Force run even if no documents are ready for embedding"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only show pipeline status, don't process anything"
    )
    
    args = parser.parse_args()
    
    # Create trigger
    trigger = BatchEmbeddingTrigger()
    
    if args.status_only:
        # Show status only
        summary = trigger.get_pipeline_summary()
        if summary:
            print("\n" + "="*60)
            print("PIPELINE STATUS OVERVIEW")
            print("="*60)
            print(f"Total documents: {summary.get('total_documents', 0)}")
            print(f"Ready for embedding: {summary.get('ready_for_embedding', 0)}")
            print(f"Processing: {summary.get('processing', 0)}")
            print(f"Pending: {summary.get('pending', 0)}")
            print(f"Failed: {summary.get('failed', 0)}")
            print("="*60)
            
            # Show step completion
            step_completion = summary.get('step_completion', {})
            if step_completion:
                print("\nSTEP COMPLETION STATUS:")
                for step, count in step_completion.items():
                    print(f"  {step}: {count} documents")
        else:
            print("❌ Failed to get pipeline status")
        return
    
    # Process embedding batch
    results = trigger.process_embedding_batch(
        batch_size=args.batch_size,
        force_run=args.force_run
    )
    
    # Final status
    print("\n" + "="*60)
    print("STEP 6: BATCH EMBEDDING COMPLETE")
    print("="*60)
    print(f"Message: {results['message']}")
    print(f"Documents processed: {results['processed']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Total chunks: {results['total_chunks']}")
    print("="*60)
    
    if results['successful'] > 0:
        print("✅ Batch embedding completed successfully!")
    else:
        print("⚠️  No documents were processed")

if __name__ == "__main__":
    main()
