#!/usr/bin/env python3
"""
Tests for 04_deduplicate_pdfs_and_move_to_dlq.py

Tests the PDF deduplication and DLQ movement functionality.
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

from step04_deduplicate_pdfs_and_move_to_dlq import PDFDeduplicator


class TestPDFDeduplicator:
    """Test the PDFDeduplicator class."""
    
    def setup_method(self):
        """Set up test directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.metadata_dir = self.temp_dir / "metadata"
        self.pdf_dir = self.temp_dir / "pdfs"
        self.dlq_dir = self.temp_dir / "dlq"
        
        self.metadata_dir.mkdir()
        self.pdf_dir.mkdir()
        self.dlq_dir.mkdir()
        
        # Create test PDF files
        (self.pdf_dir / "paper1.pdf").touch()
        (self.pdf_dir / "paper2.pdf").touch()
        (self.pdf_dir / "paper3.pdf").touch()
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        assert self.dlq_dir.exists()
        assert (self.dlq_dir.parent.parent / "logs").exists()
    
    def test_normalize_text(self):
        """Test text normalization for comparison."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        # Test basic normalization
        assert deduplicator.normalize_text("Hello World") == "hello world"
        assert deduplicator.normalize_text("  Extra  Spaces  ") == "extra spaces"
        assert deduplicator.normalize_text("") == ""
        assert deduplicator.normalize_text(None) == ""
    
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        # Test identical texts
        assert deduplicator.calculate_similarity("hello world", "hello world") == 1.0
        
        # Test similar texts
        similarity = deduplicator.calculate_similarity("hello world", "hello world!")
        assert similarity > 0.8
        
        # Test different texts
        similarity = deduplicator.calculate_similarity("hello world", "goodbye world")
        assert similarity < 0.8
        
        # Test empty texts
        assert deduplicator.calculate_similarity("", "hello") == 0.0
        assert deduplicator.calculate_similarity("hello", "") == 0.0
    
    def test_are_pdfs_duplicates_title_similarity(self):
        """Test duplicate detection based on title similarity."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        metadata1 = {
            "document_type": "research_paper",
            "title": "The Effects of Climate Change",
            "authors": ["Author A"],
            "extraction_confidence": "high"
        }
        
        metadata2 = {
            "document_type": "research_paper",
            "title": "The Effects of Climate Change on Ecosystems",
            "authors": ["Author B"],
            "extraction_confidence": "high"
        }
        
        is_duplicate, reason, confidence = deduplicator.are_pdfs_duplicates(metadata1, metadata2)
        
        assert is_duplicate == True
        assert "Similar titles" in reason
        assert confidence > 0.9
    
    def test_are_pdfs_duplicates_different_types(self):
        """Test that different document types are not considered duplicates."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        metadata1 = {
            "document_type": "research_paper",
            "title": "Same Title",
            "authors": ["Author A"]
        }
        
        metadata2 = {
            "document_type": "patent",
            "title": "Same Title",
            "authors": ["Author A"]
        }
        
        is_duplicate, reason, confidence = deduplicator.are_pdfs_duplicates(metadata1, metadata2)
        
        assert is_duplicate == False
        assert "Different document types" in reason
    
    def test_are_pdfs_duplicates_doi_match(self):
        """Test duplicate detection based on DOI match."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        metadata1 = {
            "document_type": "research_paper",
            "title": "Different Title 1",
            "doi": "10.1234/same.doi"
        }
        
        metadata2 = {
            "document_type": "research_paper",
            "title": "Different Title 2",
            "doi": "10.1234/same.doi"
        }
        
        is_duplicate, reason, confidence = deduplicator.are_pdfs_duplicates(metadata1, metadata2)
        
        assert is_duplicate == True
        assert "Same DOI" in reason
        assert confidence == 1.0
    
    def test_are_pdfs_duplicates_author_overlap(self):
        """Test duplicate detection based on author overlap."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        metadata1 = {
            "document_type": "research_paper",
            "title": "Similar Title",
            "authors": ["Author A", "Author B", "Author C"]
        }
        
        metadata2 = {
            "document_type": "research_paper",
            "title": "Similar Title",
            "authors": ["Author A", "Author B", "Author D"]
        }
        
        is_duplicate, reason, confidence = deduplicator.are_pdfs_duplicates(metadata1, metadata2)
        
        assert is_duplicate == True
        assert "Significant author overlap" in reason
    
    def test_load_metadata_files(self):
        """Test loading of metadata files."""
        # Create test metadata files
        metadata1 = {
            "filename": "paper1.pdf",
            "title": "Paper 1",
            "extraction_confidence": "high"
        }
        metadata2 = {
            "filename": "paper2.pdf",
            "title": "Paper 2",
            "extraction_confidence": "high"
        }
        
        (self.metadata_dir / "paper1_quick_metadata.json").write_text(json.dumps(metadata1))
        (self.metadata_dir / "paper2_quick_metadata.json").write_text(json.dumps(metadata2))
        
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        metadata_list = deduplicator.load_metadata_files()
        
        assert len(metadata_list) == 2
        assert metadata_list[0]["filename"] == "paper1.pdf"
        assert metadata_list[1]["filename"] == "paper2.pdf"
    
    def test_find_duplicates(self):
        """Test duplicate detection across multiple files."""
        # Create test metadata with duplicates
        metadata1 = {
            "filename": "paper1.pdf",
            "title": "Same Title",
            "authors": ["Author A"],
            "extraction_confidence": "high"
        }
        metadata2 = {
            "filename": "paper2.pdf",
            "title": "Same Title",
            "authors": ["Author A"],
            "extraction_confidence": "high"
        }
        metadata3 = {
            "filename": "paper3.pdf",
            "title": "Different Title",
            "authors": ["Author B"],
            "extraction_confidence": "high"
        }
        
        (self.metadata_dir / "paper1_quick_metadata.json").write_text(json.dumps(metadata1))
        (self.metadata_dir / "paper2_quick_metadata.json").write_text(json.dumps(metadata2))
        (self.metadata_dir / "paper3_quick_metadata.json").write_text(json.dumps(metadata3))
        
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        duplicates = deduplicator.find_duplicates(deduplicator.load_metadata_files())
        
        assert len(duplicates) == 1  # One duplicate pair
        assert duplicates[0][0]["filename"] == "paper1.pdf"  # Keep this one
        assert duplicates[0][1]["filename"] == "paper2.pdf"  # Remove this one
    
    def test_move_to_dlq(self):
        """Test moving files to DLQ."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        # Test successful move
        success = deduplicator.move_to_dlq("paper1.pdf", "Duplicate title")
        assert success == True
        
        # Check that file was moved
        assert not (self.pdf_dir / "paper1.pdf").exists()
        
        # Check that file is in DLQ with timestamp
        dlq_files = list(self.dlq_dir.glob("*paper1*"))
        assert len(dlq_files) == 1
        assert "duplicate" in dlq_files[0].name
    
    def test_move_to_dlq_nonexistent_file(self):
        """Test handling of nonexistent files."""
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        
        success = deduplicator.move_to_dlq("nonexistent.pdf", "Test reason")
        assert success == False
    
    def test_run_deduplication_dry_run(self):
        """Test deduplication in dry run mode."""
        # Create test metadata with duplicates
        metadata1 = {
            "filename": "paper1.pdf",
            "title": "Same Title",
            "authors": ["Author A"],
            "extraction_confidence": "high"
        }
        metadata2 = {
            "filename": "paper2.pdf",
            "title": "Same Title",
            "authors": ["Author A"],
            "extraction_confidence": "high"
        }
        
        (self.metadata_dir / "paper1_quick_metadata.json").write_text(json.dumps(metadata1))
        (self.metadata_dir / "paper2_quick_metadata.json").write_text(json.dumps(metadata2))
        
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        result = deduplicator.run_deduplication(dry_run=True)
        
        assert result["duplicates_found"] == 1
        assert result["moved_to_dlq"] == 0  # No files moved in dry run
        assert result["dry_run"] == True
        
        # Files should still be in original location
        assert (self.pdf_dir / "paper1.pdf").exists()
        assert (self.pdf_dir / "paper2.pdf").exists()
    
    def test_run_deduplication_real(self):
        """Test actual deduplication and file movement."""
        # Create test metadata with duplicates
        metadata1 = {
            "filename": "paper1.pdf",
            "title": "Same Title",
            "authors": ["Author A"],
            "extraction_confidence": "high"
        }
        metadata2 = {
            "filename": "paper2.pdf",
            "title": "Same Title",
            "authors": ["Author A"],
            "extraction_confidence": "high"
        }
        
        (self.metadata_dir / "paper1_quick_metadata.json").write_text(json.dumps(metadata1))
        (self.metadata_dir / "paper2_quick_metadata.json").write_text(json.dumps(metadata2))
        
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        result = deduplicator.run_deduplication(dry_run=False)
        
        assert result["duplicates_found"] == 1
        assert result["moved_to_dlq"] == 1
        assert result["dry_run"] == False
        
        # One file should be moved to DLQ
        assert not (self.pdf_dir / "paper2.pdf").exists()
        dlq_files = list(self.dlq_dir.glob("*paper2*"))
        assert len(dlq_files) == 1
    
    def test_handles_empty_metadata_directory(self):
        """Test handling of empty metadata directory."""
        empty_metadata_dir = self.temp_dir / "empty_metadata"
        empty_metadata_dir.mkdir()
        
        deduplicator = PDFDeduplicator(empty_metadata_dir, self.pdf_dir, self.dlq_dir)
        result = deduplicator.run_deduplication()
        
        assert "error" in result
        assert result["error"] == "No metadata files found"
    
    def test_handles_no_duplicates(self):
        """Test handling when no duplicates are found."""
        # Create test metadata with no duplicates
        metadata1 = {
            "filename": "paper1.pdf",
            "title": "Title 1",
            "authors": ["Author A"],
            "extraction_confidence": "high"
        }
        metadata2 = {
            "filename": "paper2.pdf",
            "title": "Title 2",
            "authors": ["Author B"],
            "extraction_confidence": "high"
        }
        
        (self.metadata_dir / "paper1_quick_metadata.json").write_text(json.dumps(metadata1))
        (self.metadata_dir / "paper2_quick_metadata.json").write_text(json.dumps(metadata2))
        
        deduplicator = PDFDeduplicator(self.metadata_dir, self.pdf_dir, self.dlq_dir)
        result = deduplicator.run_deduplication()
        
        assert result["duplicates_found"] == 0
        assert result["moved_to_dlq"] == 0
        assert result["remaining_pdfs"] == 2


if __name__ == "__main__":
    pytest.main([__file__]) 