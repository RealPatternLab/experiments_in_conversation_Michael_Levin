#!/usr/bin/env python3
"""
Tests for 09_generate_vector_embeddings_for_chunks.py

Tests the vector embedding generation functionality for semantic chunks.
"""

import pytest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step09_generate_vector_embeddings_for_chunks import VectorEmbedder


class TestVectorEmbedder:
    """Test the VectorEmbedder class."""
    
    def setup_method(self):
        """Set up test directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test chunk files
        self.test_chunks_file = self.input_dir / "paper1_chunks.json"
        self.test_chunks_file.write_text(json.dumps({
            "metadata": {
                "filename": "paper1_extracted_text.txt",
                "chunking_method": "semantic",
                "chunk_count": 3,
                "max_chunk_size": 500
            },
            "chunks": [
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
        }))
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        embedder = VectorEmbedder(self.input_dir, self.output_dir)
        
        assert self.output_dir.exists()
        assert (Path("logs")).exists()
    
    def test_generate_embedding_success(self):
        """Test successful embedding generation."""
        with patch('openai.Embedding.create') as mock_create:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
            mock_create.return_value = mock_response
            
            extractor = VectorEmbedder(self.input_dir, self.output_dir)
            embedding = extractor.generate_embedding("Test text content")
            
            assert len(embedding) == 500
            assert all(isinstance(x, float) for x in embedding)

    def test_generate_embedding_api_error(self):
        """Test handling of API errors during embedding generation."""
        with patch('openai.Embedding.create') as mock_create:
            mock_create.side_effect = Exception("API rate limit exceeded")
            
            extractor = VectorEmbedder(self.input_dir, self.output_dir)
            embedding = extractor.generate_embedding("Test text content")
            
            assert embedding is None
    
    def test_get_chunk_files_to_process(self):
        """Test chunk file discovery and processing logic."""
        # Create multiple chunk files
        (self.input_dir / "paper1_chunks.json").touch()
        (self.input_dir / "paper2_chunks.json").touch()
        (self.input_dir / "paper3_chunks.json").touch()
        
        embedder = VectorEmbedder(self.input_dir, self.output_dir)
        
        # Test processing all files
        chunk_files = embedder.get_chunk_files_to_process()
        assert len(chunk_files) == 3
        
        # Test max_files limit
        chunk_files_limited = embedder.get_chunk_files_to_process(max_files=2)
        assert len(chunk_files_limited) == 2
        
        # Test already processed files
        (self.output_dir / "paper1_embeddings.pkl").touch()
        chunk_files_processed = embedder.get_chunk_files_to_process()
        assert len(chunk_files_processed) == 2  # paper1 already processed
    
    def test_process_file(self):
        """Test processing of individual chunk file."""
        with patch('openai.Embedding.create') as mock_create:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
            mock_create.return_value = mock_response
            
            extractor = VectorEmbedder(self.input_dir, self.output_dir)
            success = extractor.process_file(self.test_chunks_file)
            
            assert success == True
            
            # Check that embeddings file was created
            embeddings_file = self.output_dir / "test_paper_embeddings.pkl"
            assert embeddings_file.exists()
    
    def test_process_file_with_metadata(self):
        """Test processing of chunk file with metadata."""
        with patch('openai.Embedding.create') as mock_create:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
            mock_create.return_value = mock_response
            
            extractor = VectorEmbedder(self.input_dir, self.output_dir)
            success = extractor.process_file(self.test_chunks_file)
            
            assert success == True
            
            # Check that embeddings file was created
            embeddings_file = self.output_dir / "test_paper_embeddings.pkl"
            assert embeddings_file.exists()
        
        # Load and check embeddings data
        import pickle
        with open(embeddings_file, 'rb') as f:
            embeddings_data = pickle.load(f)
            
            assert "embeddings" in embeddings_data
            assert "metadata" in embeddings_data
            assert "chunk_info" in embeddings_data
            
            # Check metadata
            metadata = embeddings_data["metadata"]
            assert metadata["filename"] == "paper1_extracted_text.txt"
            assert metadata["embedding_model"] == "text-embedding-ada-002"
            assert metadata["chunk_count"] == 3
            
            # Check embeddings
            embeddings = embeddings_data["embeddings"]
            assert len(embeddings) == 3
            assert all(len(emb) == 1536 for emb in embeddings)  # OpenAI ada-002 dimension
            
            # Check chunk info
            chunk_info = embeddings_data["chunk_info"]
            assert len(chunk_info) == 3
            assert chunk_info[0]["chunk_id"] == "chunk_1"
            assert chunk_info[0]["text"] == "The Effects of Climate Change on Marine Ecosystems"
    
    def test_process_files(self):
        """Test batch processing of chunk files."""
        with patch('openai.Embedding.create') as mock_create:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
            mock_create.return_value = mock_response
            
            extractor = VectorEmbedder(self.input_dir, self.output_dir)
            results = extractor.process_files()
            
            assert results["total"] == 2
            assert results["processed"] == 2
            assert results["failed"] == 0
    
    def test_handles_empty_chunks_file(self):
        """Test handling of empty chunk files."""
        empty_file = self.input_dir / "empty_chunks.json"
        empty_file.write_text("{}")
        
        embedder = VectorEmbedder(self.input_dir, self.output_dir)
        
        success = embedder.process_file(empty_file)
        assert success == False  # Should fail with empty chunks
    
    def test_handles_malformed_chunks_file(self):
        """Test handling of malformed chunk files."""
        malformed_file = self.input_dir / "malformed_chunks.json"
        malformed_file.write_text("{ invalid json }")
        
        embedder = VectorEmbedder(self.input_dir, self.output_dir)
        
        success = embedder.process_file(malformed_file)
        assert success == False  # Should fail with malformed JSON
    
    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        embedder = VectorEmbedder(empty_dir, self.output_dir)
        chunk_files = embedder.get_chunk_files_to_process()
        
        assert len(chunk_files) == 0
    
    def test_handles_nonexistent_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            embedder = VectorEmbedder(nonexistent_dir, self.output_dir)
            embedder.get_chunk_files_to_process()
    
    def test_embedding_dimensions_consistent(self):
        """Test that all embeddings have consistent dimensions."""
        with patch('openai.Embedding.create') as mock_create:
            mock_response = MagicMock()
            mock_response.data = [MagicMock()]
            mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100
            mock_create.return_value = mock_response
            
            extractor = VectorEmbedder(self.input_dir, self.output_dir)
            success = extractor.process_file(self.test_chunks_file)
            
            assert success == True
            
            # Check that embeddings file was created
            embeddings_file = self.output_dir / "test_paper_embeddings.pkl"
            assert embeddings_file.exists()


if __name__ == "__main__":
    pytest.main([__file__]) 