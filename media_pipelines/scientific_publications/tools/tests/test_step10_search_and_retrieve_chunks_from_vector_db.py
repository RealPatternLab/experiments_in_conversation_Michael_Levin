#!/usr/bin/env python3
"""
Tests for 10_search_and_retrieve_chunks_from_vector_db.py

Tests the vector database search and retrieval functionality.
"""

import pytest
import tempfile
import shutil
import json
import pickle
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step10_search_and_retrieve_chunks_from_vector_db import VectorDBSearcher


class TestVectorDBSearcher:
    """Test the VectorDBSearcher class."""
    
    def setup_method(self):
        """Set up test directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.embeddings_dir = self.temp_dir / "embeddings"
        self.output_dir = self.temp_dir / "output"
        
        self.embeddings_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test embedding files
        self.test_embeddings_file = self.embeddings_dir / "paper1_embeddings.pkl"
        
        # Create mock embeddings data
        embeddings_data = {
            "embeddings": [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 100),  # 500-dim vector
                np.array([0.6, 0.7, 0.8, 0.9, 1.0] * 100),
                np.array([1.1, 1.2, 1.3, 1.4, 1.5] * 100)
            ],
            "metadata": {
                "filename": "paper1_extracted_text.txt",
                "embedding_model": "text-embedding-ada-002",
                "chunk_count": 3
            },
            "chunk_info": [
                {
                    "chunk_id": "chunk_1",
                    "text": "The Effects of Climate Change on Marine Ecosystems",
                    "start_position": 0,
                    "end_position": 50
                },
                {
                    "chunk_id": "chunk_2",
                    "text": "This study examines the impact of climate change on marine ecosystems.",
                    "start_position": 51,
                    "end_position": 120
                },
                {
                    "chunk_id": "chunk_3",
                    "text": "Our findings indicate significant changes in marine biodiversity.",
                    "start_position": 121,
                    "end_position": 180
                }
            ]
        }
        
        with open(self.test_embeddings_file, 'wb') as f:
            pickle.dump(embeddings_data, f)
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        assert self.output_dir.exists()
        assert (Path("logs")).exists()
    
    def test_load_embeddings(self):
        """Test loading embeddings from pickle files."""
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        embeddings_data = searcher.load_embeddings()
        
        # Check that embeddings were loaded
        assert len(embeddings_data) > 0
        
        # Check that the key is the filename without extension
        assert "paper1" in embeddings_data
        
        # Check that the data structure is correct
        paper1_data = embeddings_data["paper1"]
        assert "embeddings" in paper1_data
        assert "metadata" in paper1_data
        assert "chunk_info" in paper1_data
    
    def test_build_faiss_index(self):
        """Test building FAISS index from embeddings."""
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # Load embeddings first
        embeddings_data = searcher.load_embeddings()
        
        # Build index
        index, chunk_mapping = searcher.build_faiss_index(embeddings_data)
        
        assert index is not None
        assert len(chunk_mapping) == 3  # Three chunks
        assert all("filename" in chunk for chunk in chunk_mapping)
        assert all("chunk_id" in chunk for chunk in chunk_mapping)
        assert all("text" in chunk for chunk in chunk_mapping)
    
    def test_search_similar_chunks(self):
        """Test searching for similar chunks."""
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # Load embeddings and build index
        embeddings_data = searcher.load_embeddings()
        index, chunk_mapping = searcher.build_faiss_index(embeddings_data)
        
        # Create a query embedding
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 100)
        
        # Search for similar chunks
        results = searcher.search_similar_chunks(query_embedding, index, chunk_mapping, k=2)
        
        assert len(results) == 2
        assert all("chunk_id" in result for result in results)
        assert all("text" in result for result in results)
        assert all("similarity_score" in result for result in results)
        assert all("filename" in result for result in results)
    
    def test_search_by_text_query(self):
        """Test searching by text query."""
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # Mock OpenAI API calls
        with patch('openai.Embedding.create') as mock_create:
            # Mock successful embedding generation
            mock_create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 100)]
            )
            
            # Mock successful search
            with patch.object(searcher, 'search_similar_chunks') as mock_search:
                mock_search.return_value = [
                    {
                        'chunk_id': 'chunk_1',
                        'text': 'Relevant text about climate change',
                        'similarity_score': 0.85,
                        'filename': 'paper1',
                        'section': 'Introduction'
                    }
                ]
                
                results = searcher.search_by_text_query("climate change effects")
                
                assert results is not None
                assert len(results) == 1
                assert results[0]['chunk_id'] == 'chunk_1'
                assert results[0]['similarity_score'] == 0.85
    
    def test_search_by_text_query_api_error(self):
        """Test handling of API errors during text query search."""
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # Mock OpenAI API error
        with patch('openai.Embedding.create') as mock_create:
            mock_create.side_effect = Exception("API rate limit exceeded")
            
            # Should handle error gracefully
            results = searcher.search_by_text_query("climate change effects")
            
            assert results is None
    
    def test_get_embedding_files_to_process(self):
        """Test embedding file discovery and processing logic."""
        # Create multiple embedding files
        (self.embeddings_dir / "paper1_embeddings.pkl").touch()
        (self.embeddings_dir / "paper2_embeddings.pkl").touch()
        (self.embeddings_dir / "paper3_embeddings.pkl").touch()
        
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # Test processing all files
        embedding_files = searcher.get_embedding_files_to_process()
        assert len(embedding_files) == 3
        
        # Test max_files limit
        embedding_files_limited = searcher.get_embedding_files_to_process(max_files=2)
        assert len(embedding_files_limited) == 2
    
    def test_process_files(self):
        """Test batch file processing."""
        # Create multiple embedding files
        (self.embeddings_dir / "paper1_embeddings.pkl").touch()
        (self.embeddings_dir / "paper2_embeddings.pkl").touch()
        
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # The tool processes all files at once
        results = searcher.process_files()
        
        # The tool should handle empty files gracefully
        assert results["total"] == 2
        # The actual behavior depends on how the tool handles empty files
        # Let's just check that we get a valid result structure
        assert "total" in results
        assert "processed" in results
        assert "failed" in results
    
    def test_handles_empty_embeddings_file(self):
        """Test handling of empty embedding files."""
        empty_file = self.embeddings_dir / "empty_embeddings.pkl"
        empty_file.write_text("")
        
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        embeddings_data = searcher.load_embeddings()
        assert len(embeddings_data) == 1  # Only the valid file should be loaded
    
    def test_handles_malformed_embeddings_file(self):
        """Test handling of malformed embedding files."""
        malformed_file = self.embeddings_dir / "malformed_embeddings.pkl"
        malformed_file.write_text("invalid pickle data")
        
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # Should handle gracefully and skip malformed files
        embeddings_data = searcher.load_embeddings()
        assert len(embeddings_data) == 1  # Only the valid file should be loaded
    
    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        searcher = VectorDBSearcher(empty_dir, self.output_dir)
        embedding_files = searcher.get_embedding_files_to_process()
        
        assert len(embedding_files) == 0
    
    def test_handles_nonexistent_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        # The tool should handle this gracefully and return empty list
        searcher = VectorDBSearcher(nonexistent_dir, self.output_dir)
        embedding_files = searcher.get_embedding_files_to_process()
        
        assert len(embedding_files) == 0
    
    def test_similarity_scores_are_normalized(self):
        """Test that similarity scores are properly normalized."""
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # Load embeddings and build index
        embeddings_data = searcher.load_embeddings()
        index, chunk_mapping = searcher.build_faiss_index(embeddings_data)
        
        # Create a query embedding
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 100)
        
        # Search for similar chunks
        results = searcher.search_similar_chunks(query_embedding, index, chunk_mapping, k=3)
        
        # Check that similarity scores are reasonable
        for result in results:
            assert 0.0 <= result["similarity_score"] <= 1.0
    
    def test_search_results_are_ranked(self):
        """Test that search results are properly ranked by similarity."""
        searcher = VectorDBSearcher(self.embeddings_dir, self.output_dir)
        
        # Load embeddings and build index
        embeddings_data = searcher.load_embeddings()
        index, chunk_mapping = searcher.build_faiss_index(embeddings_data)
        
        # Create a query embedding
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5] * 100)
        
        # Search for similar chunks
        results = searcher.search_similar_chunks(query_embedding, index, chunk_mapping, k=3)
        
        # Check that results are ranked by similarity (descending)
        scores = [result["similarity_score"] for result in results]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__]) 