#!/usr/bin/env python3
"""
Tests for 03_extract_quick_metadata_with_gemini.py

Tests the Gemini Pro-based quick metadata extraction for deduplication.
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

from step03_extract_quick_metadata_with_gemini import GeminiQuickMetadataExtractor


class TestGeminiQuickMetadataExtractor:
    """Test the GeminiQuickMetadataExtractor class."""
    
    def setup_method(self):
        """Set up test directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test PDF file
        self.test_pdf = self.input_dir / "test_paper.pdf"
        self.test_pdf.write_bytes(b"%PDF-1.4\nTest PDF content")
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
        
        assert self.output_dir.exists()
        assert (self.output_dir.parent.parent / "logs").exists()
    
    def test_gemini_import_success(self):
        """Test successful import of Gemini."""
        with patch('step03_extract_quick_metadata_with_gemini.genai') as mock_genai:
            mock_genai.configure.return_value = None
            mock_genai.GenerativeModel.return_value = MagicMock()
            
            # Should not raise an exception
            extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
            assert extractor is not None

    def test_gemini_import_failure(self):
        """Test handling of Gemini import failure."""
        with patch('step03_extract_quick_metadata_with_gemini.genai', side_effect=ImportError("No module named 'genai'")):
            # Should handle import failure gracefully
            extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
            assert extractor is not None
    
    def test_create_metadata_prompt(self):
        """Test that metadata prompt is created correctly."""
        extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
        prompt = extractor.create_metadata_prompt()
        
        assert "DOCUMENT CLASSIFICATION" in prompt
        assert "METADATA EXTRACTION" in prompt
        assert "research_paper" in prompt
        assert "title" in prompt
        assert "authors" in prompt
        assert "doi" in prompt
    
    def test_extract_metadata_with_gemini_success(self):
        """Test successful metadata extraction with Gemini."""
        with patch('step03_extract_quick_metadata_with_gemini.genai') as mock_genai:
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Title: Test Paper\nAuthors: Dr. Smith, Dr. Jones\nAbstract: This is a test abstract"
            mock_model.generate_content.return_value = mock_response
            
            mock_genai.configure.return_value = None
            mock_genai.GenerativeModel.return_value = mock_model
            
            extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
            metadata = extractor.extract_metadata_with_gemini("test content")
            
            assert metadata is not None
            assert "Title: Test Paper" in metadata
    
    def test_extract_metadata_with_gemini_fallback(self):
        """Test fallback when Gemini fails."""
        with patch('step03_extract_quick_metadata_with_gemini.genai') as mock_genai:
            mock_genai.configure.side_effect = Exception("API Error")
            
            extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
            metadata = extractor.extract_metadata_with_gemini("test content")
            
            # Should return fallback metadata
            assert metadata is not None
    
    def test_get_pdfs_to_process(self):
        """Test PDF discovery and processing logic."""
        # Create multiple PDF files
        (self.input_dir / "paper1.pdf").touch()
        (self.input_dir / "paper2.PDF").touch()
        (self.input_dir / "paper3.pdf").touch()
        
        extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
        
        # Test processing all files
        pdfs = extractor.get_pdfs_to_process()
        assert len(pdfs) == 3
        
        # Test max_files limit
        pdfs_limited = extractor.get_pdfs_to_process(max_files=2)
        assert len(pdfs_limited) == 2
        
        # Test already processed files
        (self.output_dir / "paper1_quick_metadata.json").touch()
        pdfs_processed = extractor.get_pdfs_to_process()
        assert len(pdfs_processed) == 2  # paper1 already processed
    
    def test_process_file(self):
        """Test individual file processing."""
        extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
        
        # Mock the extraction method
        with patch.object(extractor, 'extract_metadata_with_gemini') as mock_extract:
            mock_extract.return_value = {
                "document_type": "research_paper",
                "title": "Test Title",
                "extraction_confidence": "high"
            }
            
            success = extractor.process_file(self.test_pdf)
            assert success == True
            
            # Check that metadata file was created
            metadata_file = self.output_dir / "test_paper_quick_metadata.json"
            assert metadata_file.exists()
            
            # Check metadata content
            with open(metadata_file, 'r') as f:
                saved_metadata = json.load(f)
                assert saved_metadata["title"] == "Test Title"
    
    def test_process_files(self):
        """Test batch file processing."""
        # Create multiple PDF files
        (self.input_dir / "paper1.pdf").touch()
        (self.input_dir / "paper2.pdf").touch()
        
        extractor = GeminiQuickMetadataExtractor(self.input_dir, self.output_dir)
        
        # Mock the extraction method
        with patch.object(extractor, 'extract_metadata_with_gemini') as mock_extract:
            mock_extract.return_value = {
                "document_type": "research_paper",
                "title": "Test Title",
                "extraction_confidence": "high"
            }
            
            results = extractor.process_files()
            
            assert results["total"] == 2
            assert results["processed"] == 2
            assert results["failed"] == 0
    
    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        extractor = GeminiQuickMetadataExtractor(empty_dir, self.output_dir)
        pdfs = extractor.get_pdfs_to_process()
        
        assert len(pdfs) == 0
    
    def test_handles_nonexistent_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            extractor = GeminiQuickMetadataExtractor(nonexistent_dir, self.output_dir)
            extractor.get_pdfs_to_process()


if __name__ == "__main__":
    pytest.main([__file__]) 