#!/usr/bin/env python3
"""
Relevant Chunks Retrieval Tool using FAISS

This tool retrieves the most relevant semantic chunks for a given query using FAISS.
Used by the Streamlit app for RAG (Retrieval-Augmented Generation).

Uses FAISS for vector storage and similarity search with efficient pre-search filtering.

Usage:
    python tools/retrieve_relevant_chunks_faiss.py --query "your question here" --top-k 5
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
import numpy as np
import faiss
from datetime import datetime

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: pip install openai faiss-cpu python-dotenv")
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
        logging.FileHandler(logs_dir / 'retrieve_relevant_chunks_faiss.log')
    ]
)
logger = logging.getLogger(__name__)

class RelevantChunksRetrieverFAISS:
    """Retrieves relevant semantic chunks for a given query using FAISS with pre-search filtering."""
    
    def __init__(self, faiss_dir: Union[str, Path], model_name: str = "text-embedding-3-large"):
        """Initialize the FAISS retriever.
        
        Args:
            faiss_dir: Directory containing FAISS index files
            model_name: Name of the OpenAI embedding model to use
        """
        # Convert string to Path if needed
        self.faiss_dir = Path(faiss_dir) if isinstance(faiss_dir, str) else faiss_dir
        self.model_name = model_name
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"🤖 Using OpenAI model: {model_name}")
        
        # Initialize index and metadata
        self.index = None
        self.metadata = []
        self.embeddings = None
        self._load_faiss_index()
        
    def _load_faiss_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            index_path = self.faiss_dir / "chunks.index"
            metadata_path = self.faiss_dir / "chunks_metadata.pkl"
            embeddings_path = self.faiss_dir / "chunks_embeddings.npy"
            
            # Add debug logging to see exactly what we're loading
            logger.info(f"🔍 DEBUG: FAISS directory: {self.faiss_dir}")
            logger.info(f"🔍 DEBUG: Absolute FAISS directory: {self.faiss_dir.absolute()}")
            logger.info(f"🔍 DEBUG: Index path: {index_path}")
            logger.info(f"🔍 DEBUG: Index path absolute: {index_path.absolute()}")
            logger.info(f"🔍 DEBUG: Index file exists: {index_path.exists()}")
            if index_path.exists():
                logger.info(f"🔍 DEBUG: Index file size: {index_path.stat().st_size} bytes")
                logger.info(f"🔍 DEBUG: Index file modification time: {datetime.fromtimestamp(index_path.stat().st_mtime)}")
            
            if not index_path.exists():
                logger.warning(f"⚠️  FAISS index not found at {index_path}")
                logger.info("Please run the embedding tool first to create the FAISS index.")
                return
            
            if not metadata_path.exists():
                logger.warning(f"⚠️  Metadata file not found at {metadata_path}")
                logger.info("Please run the embedding tool first to create the metadata file.")
                return
            
            # Load FAISS index
            logger.info(f"🗄️  Loading FAISS index from: {index_path}")
            self.index = faiss.read_index(str(index_path))
            logger.info(f"🔍 DEBUG: FAISS index loaded with {self.index.ntotal} vectors")
            
            # Load metadata
            logger.info(f"📋 Loading metadata from: {metadata_path}")
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"🔍 DEBUG: Metadata loaded with {len(self.metadata)} entries")
            
            # Load embeddings if available (for pre-search filtering)
            if embeddings_path.exists():
                logger.info(f"📊 Loading embeddings from: {embeddings_path}")
                self.embeddings = np.load(str(embeddings_path))
                logger.info(f"🔍 DEBUG: Embeddings loaded with shape: {self.embeddings.shape}")
            else:
                logger.info("⚠️  Embeddings file not found, will use FAISS index for search")
                self.embeddings = None
            
            # Validate index consistency
            self._validate_index_consistency()
            
            logger.info(f"✅ Loaded FAISS index with {len(self.metadata)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Error loading FAISS index: {e}")
            raise
    
    def _validate_index_consistency(self):
        """Validate that FAISS index and metadata are in sync."""
        if self.index is None or not self.metadata:
            return
        
        actual_index_size = self.index.ntotal
        metadata_size = len(self.metadata)
        
        if actual_index_size != metadata_size:
            logger.warning(f"⚠️  INDEX MISALIGNMENT DETECTED!")
            logger.warning(f"   FAISS index has {actual_index_size} vectors")
            logger.warning(f"   Metadata has {metadata_size} entries")
            logger.warning(f"   Difference: {abs(actual_index_size - metadata_size)} entries")
            
            if actual_index_size > metadata_size:
                logger.warning("   FAISS index has more vectors than metadata entries")
                logger.warning("   This usually means new data was added but metadata wasn't updated")
            else:
                logger.warning("   Metadata has more entries than FAISS index vectors")
                logger.warning("   This usually means metadata was updated but index wasn't rebuilt")
            
            logger.warning("   Consider running the embedding tool to rebuild the index")
        else:
            logger.info(f"✅ Index consistency validated: {actual_index_size} vectors match {metadata_size} metadata entries")
    
    def _get_filtered_indices(self, where_filter: Dict) -> List[int]:
        """Get indices of chunks that match the filter criteria."""
        if not where_filter:
            return list(range(len(self.metadata)))
        
        filtered_indices = []
        
        for i, metadata in enumerate(self.metadata):
            matches_filter = True
            
            for key, value in where_filter.items():
                if key not in metadata:
                    matches_filter = False
                    break
                
                # Handle different types of filtering
                if isinstance(value, str):
                    # Exact string match
                    if metadata[key] != value:
                        matches_filter = False
                        break
                elif isinstance(value, list):
                    # List of allowed values
                    if metadata[key] not in value:
                        matches_filter = False
                        break
                elif isinstance(value, dict):
                    # Range filtering (e.g., {'min': 2020, 'max': 2023})
                    if 'min' in value and metadata[key] < value['min']:
                        matches_filter = False
                        break
                    if 'max' in value and metadata[key] > value['max']:
                        matches_filter = False
                        break
                else:
                    # Exact match for other types
                    if metadata[key] != value:
                        matches_filter = False
                        break
            
            if matches_filter:
                filtered_indices.append(i)
        
        logger.info(f"🔍 Filter applied: {len(filtered_indices)} chunks match criteria")
        return filtered_indices
    
    def _create_filtered_index(self, filtered_indices: List[int]) -> faiss.Index:
        """Create a temporary FAISS index with only the filtered vectors."""
        if not filtered_indices:
            # Return empty index if no matches
            dimension = self.index.d
            return faiss.IndexFlatIP(dimension)
        
        # Extract filtered embeddings
        if self.embeddings is not None:
            # Use stored embeddings if available
            filtered_embeddings = self.embeddings[filtered_indices]
        else:
            # Extract from FAISS index (slower but works)
            logger.warning("⚠️  Extracting embeddings from FAISS index (slower)")
            # Use actual FAISS index size instead of metadata length to avoid misalignment
            actual_index_size = self.index.ntotal
            metadata_size = len(self.metadata)
            
            if actual_index_size != metadata_size:
                logger.warning(f"⚠️  Index size mismatch: FAISS has {actual_index_size} vectors, metadata has {metadata_size} entries")
                logger.warning("This indicates the index and metadata are out of sync. Using FAISS index size.")
            
            # Use the smaller of the two to avoid out-of-bounds errors
            safe_size = min(actual_index_size, metadata_size)
            filtered_embeddings = self.index.reconstruct_n(0, safe_size)[filtered_indices]
        
        # Create new index with filtered vectors
        dimension = filtered_embeddings.shape[1]
        filtered_index = faiss.IndexFlatIP(dimension)
        filtered_index.add(filtered_embeddings.astype('float32'))
        
        logger.info(f"🔧 Created filtered index with {len(filtered_indices)} vectors")
        return filtered_index
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed the query using the same model."""
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model_name
            )
            embedding = response.data[0].embedding
            return np.array(embedding).astype('float32')
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5, 
                               where_filter: Optional[Dict] = None) -> List[Dict]:
        """Retrieve the most relevant chunks for a given query with pre-search filtering."""
        try:
            logger.info(f"🔍 Retrieving relevant chunks for query: '{query}'")
            
            if self.index is None or len(self.metadata) == 0:
                logger.warning("⚠️  FAISS index not loaded. Run the embedding tool first.")
                return []
            
            logger.info(f"📊 Found {len(self.metadata)} chunks in FAISS index")
            
            # Apply pre-search filtering
            filtered_indices = self._get_filtered_indices(where_filter)
            
            if not filtered_indices:
                logger.warning("⚠️  No chunks match the filter criteria")
                return []
            
            # Create filtered index for search
            filtered_index = self._create_filtered_index(filtered_indices)
            
            # Embed the query
            query_embedding = self.embed_query(query)
            logger.info(f"✅ Query embedded successfully ({len(query_embedding)} dimensions)")
            
            # Search filtered FAISS index
            query_embedding = np.array(query_embedding).reshape(1, -1).astype('float32')
            distances, indices = filtered_index.search(query_embedding, min(top_k, len(filtered_indices)))
            
            # Process results
            chunks = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(filtered_indices):
                    # Map back to original metadata index
                    original_idx = filtered_indices[idx]
                    metadata = self.metadata[original_idx]
                    
                    # Debug logging to see what's in metadata
                    logger.info(f"🔍 DEBUG: Processing chunk {i}, metadata keys: {list(metadata.keys())}")
                    logger.info(f"🔍 DEBUG: chunk_index in metadata: {metadata.get('chunk_index', 'NOT_FOUND')}")
                    
                    # Convert distance to similarity score
                    similarity_score = 1.0 - distance
                    
                    chunk = {
                        'chunk_id': int(metadata['chunk_id']),
                        'chunk_index': int(metadata['chunk_index']),
                        'text': metadata['text'],
                        'section': metadata['section'],
                        'topic': metadata['topic'],
                        'chunk_summary': metadata.get('chunk_summary', ''),
                        'position_in_section': metadata['position_in_section'],
                        'certainty_level': metadata['certainty_level'],
                        'citation_context': metadata['citation_context'],
                        'word_count': int(metadata['word_count']),
                        'character_count': int(metadata['character_count']),
                        'original_filename': metadata['original_filename'],
                        'sanitized_filename': metadata['sanitized_filename'],
                        'title': metadata['title'],
                        'authors': metadata['authors'],
                        'doi': metadata['doi'],
                        'publication_date': metadata['publication_date'],
                        'journal': metadata['journal'],
                        'document_type': metadata['document_type'],
                        'source_type': metadata['document_type'],  # Add source_type for compatibility
                        'similarity_score': similarity_score,
                        'distance': float(distance),
                        # Add YouTube-specific fields
                        'youtube_url': metadata.get('youtube_url'),
                        'start_time': metadata.get('start_time'),
                        'end_time': metadata.get('end_time'),
                        'frame_path': metadata.get('frame_path'),
                        'semantic_topics': metadata.get('semantic_topics', {})
                    }
                    
                    # Debug logging to see what's in the created chunk
                    logger.info(f"🔍 DEBUG: Created chunk keys: {list(chunk.keys())}")
                    logger.info(f"🔍 DEBUG: chunk_index in created chunk: {chunk.get('chunk_index', 'NOT_FOUND')}")
                    
                    chunks.append(chunk)
            
            logger.info(f"✅ Retrieved {len(chunks)} most relevant chunks")
            
            # Log the results
            for i, chunk in enumerate(chunks):
                logger.info(f"   {i+1}. {chunk['title']} (chunk {chunk['chunk_id']}) - Score: {chunk['similarity_score']:.4f}")
                logger.info(f"      Topic: {chunk['topic']}")
                logger.info(f"      Section: {chunk['section']}")
                logger.info(f"      Summary: {chunk['chunk_summary'][:100]}...")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving relevant chunks: {e}")
            return []
    
    def format_chunks_for_rag(self, chunks: List[Dict]) -> str:
        """Format chunks for inclusion in RAG prompt."""
        if not chunks:
            return "No relevant chunks found."
        
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            formatted_chunk = f"""
CHUNK {i}:
Title: {chunk['title']}
Authors: {chunk['authors']}
Journal: {chunk['journal']}
Document Type: {chunk['document_type']}
Section: {chunk['section']}
Topic: {chunk['topic']}
Relevance Score: {chunk['similarity_score']:.4f}

Content: {chunk['text']}

Summary: {chunk['chunk_summary']}
"""
            formatted_chunks.append(formatted_chunk)
        
        return "\n".join(formatted_chunks)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the FAISS index."""
        try:
            if self.index is None:
                return {'total_chunks': 0, 'index_name': 'faiss_index', 'embedding_model': self.model_name}
            
            return {
                'total_chunks': len(self.metadata),
                'index_name': 'faiss_index',
                'embedding_model': self.model_name,
                'index_dimension': self.index.d,
                'has_embeddings_file': self.embeddings is not None
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Retrieve relevant semantic chunks for a query using FAISS"
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=Path("data/faiss"),
        help="Directory for FAISS index storage"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to find relevant chunks for"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="text-embedding-3-large",
        help="OpenAI embedding model to use for embeddings"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top relevant chunks to retrieve"
    )
    parser.add_argument(
        "--format-for-rag",
        action="store_true",
        help="Format output for RAG prompt inclusion"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show FAISS index statistics"
    )
    
    args = parser.parse_args()
    
    try:
        retriever = RelevantChunksRetrieverFAISS(args.faiss_dir, args.model_name)
        
        # Show stats if requested
        if args.show_stats:
            stats = retriever.get_collection_stats()
            print("\n📊 FAISS Index Statistics:")
            print(f"   Total chunks: {stats.get('total_chunks', 'N/A')}")
            print(f"   Index name: {stats.get('index_name', 'N/A')}")
            print(f"   Embedding model: {stats.get('embedding_model', 'N/A')}")
            if 'index_dimension' in stats:
                print(f"   Index dimension: {stats['index_dimension']}")
            return
        
        relevant_chunks = retriever.retrieve_relevant_chunks(args.query, args.top_k)
        
        if args.format_for_rag:
            formatted_output = retriever.format_chunks_for_rag(relevant_chunks)
            print("\n" + "="*80)
            print("FORMATTED CHUNKS FOR RAG:")
            print("="*80)
            print(formatted_output)
        else:
            print(f"\n✅ Retrieved {len(relevant_chunks)} relevant chunks for query: '{args.query}'")
            for i, chunk in enumerate(relevant_chunks, 1):
                print(f"\n{i}. {chunk['title']} (Score: {chunk['similarity_score']:.4f})")
                print(f"   Topic: {chunk['topic']}")
                print(f"   Section: {chunk['section']}")
                print(f"   Summary: {chunk['chunk_summary'][:100]}...")
        
    except Exception as e:
        logger.error(f"❌ Error during retrieval: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 