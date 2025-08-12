#!/usr/bin/env python3
"""
Tests for 07_create_semantic_chunks_from_text.py

Tests the semantic chunking functionality for extracted text files.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step07_create_semantic_chunks_from_text import SemanticChunker


class TestSemanticChunker:
    """Test the SemanticChunker class."""
    
    def setup_method(self):
        """Set up test directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test text files
        self.test_text_file = self.input_dir / "paper1_extracted_text.txt"
        self.test_text_file.write_text("""
        The Effects of Climate Change on Marine Ecosystems
        
        Abstract: This study examines the impact of climate change on marine ecosystems...
        
        Introduction: Climate change is a significant threat to marine biodiversity...
        
        Methods: We conducted a comprehensive analysis of marine ecosystem data...
        
        Results: Our findings indicate significant changes in marine biodiversity...
        
        Discussion: The implications of these changes are far-reaching...
        
        Conclusion: Further research is needed to understand these impacts...
        """)
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        assert self.output_dir.exists()
        assert (Path("logs")).exists()
    
    def test_create_chunks_from_text(self):
        """Test basic chunk creation from text."""
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        text = "This is a test document with multiple sentences. It should be chunked appropriately."
        chunks = chunker.create_chunks_from_text(text, max_chunk_size=100)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)
    
    def test_create_chunks_respects_max_size(self):
        """Test that chunks respect maximum size limit."""
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        text = "This is a test document. " * 50  # Long text
        chunks = chunker.create_chunks_from_text(text, max_chunk_size=200)
        
        assert all(len(chunk) <= 200 for chunk in chunks)
    
    def test_create_chunks_maintains_semantic_boundaries(self):
        """Test that chunks maintain semantic boundaries."""
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        text = "Section 1: Introduction. This is the introduction. Section 2: Methods. This describes the methods."
        chunks = chunker.create_chunks_from_text(text, max_chunk_size=50)
        
        # Should not break in the middle of sentences
        for chunk in chunks:
            assert not chunk["text"].endswith("Intro")
            assert not chunk["text"].endswith("This")
    
    def test_get_text_files_to_process(self):
        """Test text file discovery and processing logic."""
        # Create multiple text files
        (self.input_dir / "paper1_extracted_text.txt").touch()
        (self.input_dir / "paper2_extracted_text.txt").touch()
        (self.input_dir / "paper3_extracted_text.txt").touch()
        
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        # Test processing all files
        text_files = chunker.get_text_files_to_process()
        assert len(text_files) == 3
        
        # Test max_files limit
        text_files_limited = chunker.get_text_files_to_process(max_files=2)
        assert len(text_files_limited) == 2
        
        # Test already processed files
        (self.output_dir / "paper1_chunks.json").touch()
        text_files_processed = chunker.get_text_files_to_process()
        assert len(text_files_processed) == 2  # paper1 already processed
    
    def test_process_file(self):
        """Test individual file processing."""
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        success = chunker.process_file(self.test_text_file)
        assert success == True
        
        # Check that chunks file was created
        chunks_file = self.output_dir / "paper1_chunks.json"
        assert chunks_file.exists()
        
        # Check chunks content
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
            assert "chunks" in chunks_data
            assert "metadata" in chunks_data
            assert len(chunks_data["chunks"]) > 0
            assert chunks_data["metadata"]["filename"] == "paper1_extracted_text.txt"
    
    def test_process_files(self):
        """Test batch file processing."""
        # Create multiple text files with actual content
        (self.input_dir / "paper1_extracted_text.txt").write_text("This is the first paper with some content that should be chunked properly.")
        (self.input_dir / "paper2_extracted_text.txt").write_text("This is the second paper with different content that should also be chunked.")
        
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        results = chunker.process_files()
        
        assert results["total"] == 2
        assert results["processed"] == 2
        assert results["failed"] == 0
    
    def test_handles_empty_text_file(self):
        """Test handling of empty text files."""
        empty_file = self.input_dir / "empty_extracted_text.txt"
        empty_file.write_text("")
        
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        success = chunker.process_file(empty_file)
        assert success == False  # Should fail with empty file
    
    def test_handles_very_short_text(self):
        """Test handling of very short text files."""
        short_file = self.input_dir / "short_extracted_text.txt"
        short_file.write_text("Short text.")
        
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        success = chunker.process_file(short_file)
        assert success == True
        
        # Check that chunks file was created
        chunks_file = self.output_dir / "short_chunks.json"
        assert chunks_file.exists()
        
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
            assert len(chunks_data["chunks"]) == 1  # Should have one chunk
    
    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        chunker = SemanticChunker(empty_dir, self.output_dir)
        text_files = chunker.get_text_files_to_process()
        
        assert len(text_files) == 0
    
    def test_handles_nonexistent_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            chunker = SemanticChunker(nonexistent_dir, self.output_dir)
            chunker.get_text_files_to_process()
    
    def test_chunk_metadata_includes_required_fields(self):
        """Test that chunk metadata includes all required fields."""
        chunker = SemanticChunker(self.input_dir, self.output_dir)
        
        success = chunker.process_file(self.test_text_file)
        assert success == True
        
        chunks_file = self.output_dir / "paper1_chunks.json"
        with open(chunks_file, 'r') as f:
            chunks_data = json.load(f)
            
            # Check metadata fields
            metadata = chunks_data["metadata"]
            assert "filename" in metadata
            assert "chunking_method" in metadata
            assert "chunk_count" in metadata
            assert "max_chunk_size" in metadata
            assert "processed_at" in metadata
            
            # Check chunk structure
            chunks = chunks_data["chunks"]
            assert len(chunks) > 0
            
            for chunk in chunks:
                assert "text" in chunk
                assert "chunk_id" in chunk
                assert "start_position" in chunk
                assert "end_position" in chunk
                assert len(chunk["text"]) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 