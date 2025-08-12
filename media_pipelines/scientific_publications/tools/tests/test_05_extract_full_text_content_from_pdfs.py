#!/usr/bin/env python3
"""
Tests for 05_extract_full_text_content_from_pdfs.py

Tests the full PDF text extraction functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract_full_text_content_from_pdfs import PDFTextExtractor


class TestPDFTextExtractor:
    """Test the PDFTextExtractor class."""
    
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
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        assert self.output_dir.exists()
        assert (self.output_dir.parent.parent / "logs").exists()
    
    def test_get_pdfs_to_process(self):
        """Test PDF discovery and processing logic."""
        # Create multiple PDF files
        (self.input_dir / "paper1.pdf").touch()
        (self.input_dir / "paper2.PDF").touch()
        (self.input_dir / "paper3.pdf").touch()
        
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        # Test processing all files
        pdfs = extractor.get_pdfs_to_process()
        assert len(pdfs) == 3
        
        # Test max_files limit
        pdfs_limited = extractor.get_pdfs_to_process(max_files=2)
        assert len(pdfs_limited) == 2
        
        # Test already processed files
        (self.output_dir / "paper1_extracted_text.txt").touch()
        pdfs_processed = extractor.get_pdfs_to_process()
        assert len(pdfs_processed) == 2  # paper1 already processed
    
    @patch('extract_full_text_content_from_pdfs.fitz')
    def test_extract_text_with_pymupdf_success(self, mock_fitz):
        """Test successful text extraction with PyMuPDF."""
        # Mock PyMuPDF
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 3
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page content"
        mock_doc.__getitem__ = lambda self, idx: mock_page
        mock_fitz.open.return_value = mock_doc
        
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        text = extractor.extract_text_with_pymupdf(self.test_pdf)
        
        assert text == "Page content\nPage content\nPage content"
    
    @patch('extract_full_text_content_from_pdfs.pdfplumber')
    def test_extract_text_with_pdfplumber_success(self, mock_pdfplumber):
        """Test successful text extraction with pdfplumber."""
        # Mock pdfplumber
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock(), MagicMock()]
        for page in mock_pdf.pages:
            page.extract_text.return_value = "Page content"
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        text = extractor.extract_text_with_pdfplumber(self.test_pdf)
        
        assert text == "Page content\nPage content\nPage content"
    
    @patch('extract_full_text_content_from_pdfs.PyPDF2')
    def test_extract_text_with_pypdf2_success(self, mock_pypdf2):
        """Test successful text extraction with PyPDF2."""
        # Mock PyPDF2
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock(), MagicMock(), MagicMock()]
        for page in mock_reader.pages:
            page.extract_text.return_value = "Page content"
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        text = extractor.extract_text_with_pypdf2(self.test_pdf)
        
        assert text == "Page content\nPage content\nPage content"
    
    def test_extract_text_fallback_strategy(self):
        """Test fallback text extraction strategy."""
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        # Mock all extraction methods to fail
        with patch.object(extractor, 'extract_text_with_pymupdf', side_effect=Exception("PyMuPDF failed")), \
             patch.object(extractor, 'extract_text_with_pdfplumber', side_effect=Exception("pdfplumber failed")), \
             patch.object(extractor, 'extract_text_with_pypdf2', side_effect=Exception("PyPDF2 failed")):
            
            text = extractor.extract_text(self.test_pdf)
            
            # Should return basic text or error message
            assert isinstance(text, str)
    
    def test_process_file(self):
        """Test individual file processing."""
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        # Mock the extraction method
        with patch.object(extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = "Extracted text content"
            
            success = extractor.process_file(self.test_pdf)
            assert success == True
            
            # Check that text file was created
            text_file = self.output_dir / "test_paper_extracted_text.txt"
            assert text_file.exists()
            
            # Check text content
            with open(text_file, 'r') as f:
                content = f.read()
                assert content == "Extracted text content"
    
    def test_process_files(self):
        """Test batch file processing."""
        # Create multiple PDF files
        (self.input_dir / "paper1.pdf").touch()
        (self.input_dir / "paper2.pdf").touch()
        
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        # Mock the extraction method
        with patch.object(extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = "Extracted text content"
            
            results = extractor.process_files()
            
            assert results["total"] == 2
            assert results["processed"] == 2
            assert results["failed"] == 0
    
    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        extractor = PDFTextExtractor(empty_dir, self.output_dir)
        pdfs = extractor.get_pdfs_to_process()
        
        assert len(pdfs) == 0
    
    def test_handles_nonexistent_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            extractor = PDFTextExtractor(nonexistent_dir, self.output_dir)
            extractor.get_pdfs_to_process()
    
    def test_handles_corrupted_pdf(self):
        """Test handling of corrupted PDF files."""
        # Create a corrupted PDF file
        corrupted_pdf = self.input_dir / "corrupted.pdf"
        corrupted_pdf.write_bytes(b"Not a valid PDF")
        
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        # Mock extraction to fail
        with patch.object(extractor, 'extract_text', side_effect=Exception("PDF is corrupted")):
            success = extractor.process_file(corrupted_pdf)
            assert success == False
    
    def test_handles_large_pdf(self):
        """Test handling of large PDF files."""
        # Create a large PDF file (simulated)
        large_pdf = self.input_dir / "large.pdf"
        large_pdf.write_bytes(b"%PDF-1.4\n" + b"Large content " * 1000)
        
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        # Mock extraction to return large text
        with patch.object(extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = "Large text content " * 1000
            
            success = extractor.process_file(large_pdf)
            assert success == True
            
            # Check that large file was created
            text_file = self.output_dir / "large_extracted_text.txt"
            assert text_file.exists()
            assert text_file.stat().st_size > 1000  # File should be large


if __name__ == "__main__":
    pytest.main([__file__]) 