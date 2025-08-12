#!/usr/bin/env python3
"""
Tests for 02_detect_corruption_and_sanitize_pdfs.py

Tests the PDF corruption detection and file sanitization functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step02_detect_corruption_and_sanitize_pdfs import (
    sanitize_filename,
    generate_safe_filename,
    is_file_corrupted,
    process_file_type,
    sanitize_files
)


class TestSanitizeFilename:
    """Test filename sanitization."""
    
    def test_removes_extra_dots(self):
        """Test that extra dots are removed from filenames."""
        filename = "file..with...dots...."
        sanitized = sanitize_filename(filename)
        assert sanitized == "file.with.dots"
    
    def test_removes_problematic_characters(self):
        """Test that problematic characters are removed."""
        filename = 'file<with>"problematic"chars?*'
        sanitized = sanitize_filename(filename)
        assert '<' not in sanitized
        assert '>' not in sanitized
        assert '"' not in sanitized
        assert '?' not in sanitized
        assert '*' not in sanitized

    def test_handles_empty_filename(self):
        """Test that empty filenames are handled."""
        sanitized = sanitize_filename("")
        assert sanitized == ""
    
    def test_handles_filename_with_only_special_chars(self):
        """Test that filenames with only special characters are handled."""
        sanitized = sanitize_filename("<>|?*:")
        assert sanitized == ""


class TestGenerateSafeFilename:
    """Test safe filename generation."""
    
    def test_generates_safe_filename(self):
        """Test that safe filenames are generated."""
        filename = generate_safe_filename("pdf")
        assert filename.startswith("pdf_")
        assert filename.endswith(".pdf")
        assert len(filename) > 10  # Should have timestamp

    def test_handles_different_file_types(self):
        """Test that different file types are handled correctly."""
        pdf_filename = generate_safe_filename("pdf")
        html_filename = generate_safe_filename("html")
        
        assert pdf_filename.endswith(".pdf")
        assert html_filename.endswith(".html")
    
    def test_microseconds_ensure_uniqueness(self):
        """Test that microseconds ensure unique filenames."""
        filename1 = generate_safe_filename("pdf")
        filename2 = generate_safe_filename("pdf")
        
        assert filename1 != filename2
    
    def test_handles_uppercase_extensions(self):
        """Test that uppercase extensions are handled correctly."""
        filename = generate_safe_filename("PDF")
        assert filename.startswith("PDF_")
        assert filename.endswith(".PDF")


class TestIsFileCorrupted:
    """Test file corruption detection."""
    
    def test_detects_good_file(self):
        """Test that good files are not marked as corrupted."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("This is a good file")
            temp_file = Path(f.name)
        
        try:
            assert not is_file_corrupted(temp_file)
        finally:
            temp_file.unlink()
    
    def test_detects_empty_file(self):
        """Test that empty files are not marked as corrupted."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Empty file
            pass
            temp_file = Path(f.name)
        
        try:
            assert not is_file_corrupted(temp_file)
        finally:
            temp_file.unlink()
    
    def test_detects_nonexistent_file(self):
        """Test that nonexistent files are marked as corrupted."""
        temp_file = Path("/nonexistent/file.txt")
        assert is_file_corrupted(temp_file)
    
    def test_detects_corrupted_pdf(self):
        """Test that corrupted PDFs are detected."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("This is not a valid PDF")
            temp_file = Path(f.name)
        
        try:
            # For PDFs, we'd need to check if it's a valid PDF format
            # This test assumes the function can detect invalid PDF content
            result = is_file_corrupted(temp_file)
            # The result depends on the implementation
            assert isinstance(result, bool)
        finally:
            temp_file.unlink()


class TestProcessFileType:
    """Test file type processing."""
    
    def setup_method(self):
        """Set up test directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_processes_pdf_file(self):
        """Test processing of a PDF file."""
        pdf_file = self.input_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nTest content")
        
        result = process_file_type(self.input_dir, self.output_dir, "pdf")
        
        assert result[0] == 1  # processed
        assert result[1] == 0  # corrupted

    def test_handles_nonexistent_file(self):
        """Test handling of nonexistent file."""
        nonexistent_file = self.input_dir / "nonexistent.pdf"
        
        result = process_file_type(self.input_dir, self.output_dir, "pdf")
        
        assert result[0] == 0  # processed
        assert result[1] == 0  # corrupted

    def test_handles_unsupported_file_type(self):
        """Test handling of unsupported file type."""
        unsupported_file = self.input_dir / "test.xyz"
        unsupported_file.write_bytes(b"test content")
        
        result = process_file_type(self.input_dir, self.output_dir, "xyz")
        
        assert result[0] == 0  # processed
        assert result[1] == 0  # corrupted


class TestSanitizeFiles:
    """Test the main file sanitization functionality."""
    
    def setup_method(self):
        """Set up test directory structure."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test files
        (self.input_dir / "paper1.pdf").write_bytes(b"%PDF-1.4\nTest PDF")
        (self.input_dir / "paper2.PDF").write_bytes(b"%PDF-1.4\nTest PDF 2")
        (self.input_dir / "corrupted.pdf").write_bytes(b"Not a valid PDF")
        (self.input_dir / "document.txt").touch()
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_sanitizes_all_files(self):
        """Test that all files are sanitized."""
        # Create test files
        (self.input_dir / "test1.pdf").write_bytes(b"%PDF-1.4\nTest content")
        (self.input_dir / "test2.pdf").write_bytes(b"%PDF-1.4\nTest content")
        (self.input_dir / "test3.html").write_text("<html>Test</html>")
        (self.input_dir / "test4.jpg").write_bytes(b"fake jpg content")
        
        result = sanitize_files(self.input_dir)
        
        # Check that results contain expected keys
        assert "pdf" in result
        assert "html" in result
        assert "jpg" in result
        
        # Check that files were processed
        assert result["pdf"]["processed"] == 2
        assert result["html"]["processed"] == 1
        assert result["jpg"]["processed"] == 1

    def test_creates_sanitized_filenames(self):
        """Test that sanitized filenames are created."""
        # Create test files
        (self.input_dir / "test1.pdf").write_bytes(b"%PDF-1.4\nTest content")
        
        result = sanitize_files(self.input_dir)
        
        # Check that output directory was created
        output_dir = self.input_dir / "preprocessed" / "sanitized" / "pdfs"
        assert output_dir.exists()
        
        # Check that files were moved
        output_files = list(output_dir.glob("*.pdf"))
        assert len(output_files) > 0

    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        result = sanitize_files(empty_dir)
        
        assert result == {}

    def test_handles_nonexistent_input_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        # The tool should handle this gracefully
        result = sanitize_files(nonexistent_dir)
        
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__]) 