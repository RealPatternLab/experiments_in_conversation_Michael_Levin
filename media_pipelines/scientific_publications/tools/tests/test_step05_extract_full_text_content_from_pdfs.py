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

from step05_extract_full_text_content_from_pdfs import PDFTextExtractor


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
    
    def test_extract_text_with_pymupdf_success(self):
        """Test successful text extraction with PyMuPDF."""
        with patch('step05_extract_full_text_content_from_pdfs.fitz') as mock_fitz:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = "Test PDF content from PyMuPDF"
            mock_doc.__len__ = lambda self: 1
            mock_doc.__getitem__ = lambda self, idx: mock_page
            mock_fitz.open.return_value = mock_doc
            
            extractor = PDFTextExtractor(self.input_dir, self.output_dir)
            text = extractor.extract_text_with_pymupdf(self.test_pdf)
            
            assert "Test PDF content from PyMuPDF" in text
    
    def test_extract_text_with_pdfplumber_success(self):
        """Test successful text extraction with pdfplumber."""
        with patch('step05_extract_full_text_content_from_pdfs.pdfplumber') as mock_plumber:
            mock_pdf = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Test PDF content from pdfplumber"
            mock_pdf.pages = [mock_page]
            mock_plumber.open.return_value.__enter__.return_value = mock_pdf
            
            extractor = PDFTextExtractor(self.input_dir, self.output_dir)
            text = extractor.extract_text_with_pdfplumber(self.test_pdf)
            
            assert "Test PDF content from pdfplumber" in text
    
    def test_extract_text_with_pypdf2_success(self):
        """Test successful text extraction with PyPDF2."""
        with patch('step05_extract_full_text_content_from_pdfs.PyPDF2') as mock_pypdf2:
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Test PDF content from PyPDF2"
            mock_reader.pages = [mock_page]
            mock_pypdf2.PdfReader.return_value = mock_reader
            
            extractor = PDFTextExtractor(self.input_dir, self.output_dir)
            text = extractor.extract_text_with_pypdf2(self.test_pdf)
            
            assert "Test PDF content from PyPDF2" in text
    
    def test_extract_text_fallback_strategy(self):
        """Test fallback strategy when primary extraction fails."""
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        with patch.object(extractor, 'extract_text_from_pdf', side_effect=Exception("PyMuPDF failed")):
            # Should handle error gracefully
            success, text, error = extractor.extract_text_from_pdf(self.test_pdf, {})
            assert success == False
            assert text is None
            assert error is not None

    def test_process_file(self):
        """Test processing of individual PDF file."""
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        with patch.object(extractor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = (True, "Extracted text content", None)
            
            success = extractor.process_file(self.test_pdf)
            assert success == True
            
            # Check that text file was created
            text_file = self.output_dir / "test_paper_extracted_text.txt"
            assert text_file.exists()

    def test_process_files(self):
        """Test batch processing of PDF files."""
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        with patch.object(extractor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = (True, "Extracted text content", None)
            
            results = extractor.process_files()
            
            assert results["total"] == 1
            assert results["processed"] == 1
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
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        with patch.object(extractor, 'extract_text_from_pdf', side_effect=Exception("PDF is corrupted")):
            success = extractor.process_file(self.test_pdf)
            assert success == False

    def test_handles_large_pdf(self):
        """Test handling of large PDF files."""
        extractor = PDFTextExtractor(self.input_dir, self.output_dir)
        
        with patch.object(extractor, 'extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = (True, "Large PDF content" * 1000, None)
            
            success = extractor.process_file(self.test_pdf)
            assert success == True


if __name__ == "__main__":
    pytest.main([__file__]) 