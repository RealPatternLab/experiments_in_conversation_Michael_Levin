#!/usr/bin/env python3
"""
Tests for 06_extract_metadata_from_extracted_text.py

Tests the metadata extraction from extracted text files functionality.
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

from step06_extract_metadata_from_extracted_text import MetadataExtractor


class TestMetadataExtractor:
    """Test the MetadataExtractor class."""
    
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
        
        by Dr. Jane Smith, Dr. John Doe, and Dr. Alice Johnson
        
        Abstract: This study examines the impact of climate change on marine ecosystems...
        
        Introduction: Climate change is a significant threat to marine biodiversity...
        
        DOI: 10.1234/marine.2023.001
        """)
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        assert self.output_dir.exists()
        assert (Path("logs")).exists()
    
    def test_extract_title(self):
        """Test title extraction from text."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        title = extractor.extract_title(self.test_text_file.read_text())
        
        assert title == "The Effects of Climate Change on Marine Ecosystems"
    
    def test_extract_title_short_text(self):
        """Test title extraction from short text."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        short_text = "Short Title\nContent here"
        title = extractor.extract_title(short_text)
        
        assert title == "Short Title"
    
    def test_extract_title_no_title(self):
        """Test title extraction when no clear title exists."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        no_title_text = "Abstract: This is an abstract\nIntroduction: This is an introduction"
        title = extractor.extract_title(no_title_text)
        
        assert title is None
    
    def test_extract_authors(self):
        """Test author extraction from text."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        text_with_authors = """
        Title: Test Paper
        
        by Dr. Jane Smith, Dr. John Doe
        
        Abstract: This is a test abstract...
        """
        
        authors = extractor.extract_authors(text_with_authors)
        assert len(authors) > 0
        assert any("Dr. Jane Smith" in author for author in authors)

    def test_extract_authors_et_al_pattern(self):
        """Test author extraction with et al pattern."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        text_with_et_al = """
        Title: Another Test Paper
        
        Dr. Smith et al. conducted this research...
        """
        
        authors = extractor.extract_authors(text_with_et_al)
        assert len(authors) > 0
        assert any("Dr. Smith" in author for author in authors)

    def test_extract_authors_no_authors(self):
        """Test author extraction when no authors are found."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        no_authors_text = "Title\nAbstract: No authors mentioned\nContent"
        authors = extractor.extract_authors(no_authors_text)
        
        assert len(authors) == 0
    
    def test_extract_doi(self):
        """Test DOI extraction from text."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        doi = extractor.extract_doi(self.test_text_file.read_text())
        
        assert doi == "10.1234/marine.2023.001"
    
    def test_extract_doi_no_doi(self):
        """Test DOI extraction when no DOI exists."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        no_doi_text = "Title\nAbstract: No DOI in this text\nContent"
        doi = extractor.extract_doi(no_doi_text)
        
        assert doi is None
    
    def test_extract_abstract(self):
        """Test abstract extraction from text."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        abstract = extractor.extract_abstract(self.test_text_file.read_text())
        
        assert abstract is not None
        assert "This study examines the impact of climate change" in abstract
        assert len(abstract) > 50
    
    def test_extract_abstract_no_abstract(self):
        """Test abstract extraction when no abstract exists."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        no_abstract_text = "Title\nIntroduction: No abstract section\nContent"
        abstract = extractor.extract_abstract(no_abstract_text)
        
        assert abstract is None
    
    def test_extract_publication_date(self):
        """Test publication date extraction."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        text_with_date = """
        Title: Test Paper
        
        Published in 2023
        
        Abstract: This is a test abstract...
        """
        
        publication_info = extractor.extract_publication_info(text_with_date)
        assert "publication_year" in publication_info
        assert publication_info["publication_year"] == 2023

    def test_extract_publication_date_no_date(self):
        """Test publication date extraction when no date is present."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        text_without_date = """
        Title: Test Paper
        
        Abstract: This is a test abstract...
        """
        
        publication_info = extractor.extract_publication_info(text_without_date)
        assert "publication_year" not in publication_info

    def test_extract_journal(self):
        """Test journal extraction."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        text_with_journal = """
        Title: Test Paper
        
        Published in Nature Journal
        
        Abstract: This is a test abstract...
        """
        
        publication_info = extractor.extract_publication_info(text_with_journal)
        assert "journal" in publication_info
        assert "Nature Journal" in publication_info["journal"]

    def test_extract_journal_no_journal(self):
        """Test journal extraction when no journal is present."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        text_without_journal = """
        Title: Test Paper
        
        Abstract: This is a test abstract...
        """
        
        publication_info = extractor.extract_publication_info(text_without_journal)
        assert "journal" not in publication_info

    def test_extract_all_metadata(self):
        """Test comprehensive metadata extraction."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        text_with_metadata = """
        Title: Test Paper
        
        Authors: Dr. Jane Smith, Dr. John Doe
        
        Abstract: This is a test abstract about climate change research.
        
        Keywords: climate change, research, testing
        
        Published in 2023
        
        DOI: 10.1234/test.2023
        """
        
        metadata = extractor.extract_metadata(text_with_metadata, "test_paper.txt")
        
        assert metadata["filename"] == "test_paper.txt"
        assert metadata["title"] is not None
        assert len(metadata["authors"]) > 0
        assert metadata["doi"] == "10.1234/test.2023"
        assert metadata["abstract"] is not None
        assert len(metadata["keywords"]) > 0
        assert metadata["publication_year"] == 2023
    
    def test_get_text_files_to_process(self):
        """Test text file discovery and processing logic."""
        # Create multiple text files
        (self.input_dir / "paper1_extracted_text.txt").touch()
        (self.input_dir / "paper2_extracted_text.txt").touch()
        (self.input_dir / "paper3_extracted_text.txt").touch()
        
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        # Test processing all files
        text_files = extractor.get_text_files_to_process()
        assert len(text_files) == 3
        
        # Test max_files limit
        text_files_limited = extractor.get_text_files_to_process(max_files=2)
        assert len(text_files_limited) == 2
        
        # Test already processed files
        (self.output_dir / "paper1_metadata.json").touch()
        text_files_processed = extractor.get_text_files_to_process()
        assert len(text_files_processed) == 2  # paper1 already processed
    
    def test_process_file(self):
        """Test individual file processing."""
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        success = extractor.process_file(self.test_text_file)
        assert success == True
        
        # Check that metadata file was created
        metadata_file = self.output_dir / "paper1_metadata.json"
        assert metadata_file.exists()
        
        # Check metadata content
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
            assert saved_metadata["title"] == "The Effects of Climate Change on Marine Ecosystems"
            assert "Dr. Jane Smith" in saved_metadata["authors"]
            assert saved_metadata["doi"] == "10.1234/marine.2023.001"
    
    def test_process_files(self):
        """Test batch file processing."""
        # Create multiple text files
        (self.input_dir / "paper1_extracted_text.txt").touch()
        (self.input_dir / "paper2_extracted_text.txt").touch()
        
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        results = extractor.process_files()
        
        assert results["total"] == 2
        assert results["processed"] == 2
        assert results["failed"] == 0
    
    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        extractor = MetadataExtractor(empty_dir, self.output_dir)
        text_files = extractor.get_text_files_to_process()
        
        assert len(text_files) == 0
    
    def test_handles_nonexistent_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            extractor = MetadataExtractor(nonexistent_dir, self.output_dir)
            extractor.get_text_files_to_process()
    
    def test_handles_corrupted_text_file(self):
        """Test handling of corrupted text files."""
        # Create a corrupted text file
        corrupted_file = self.input_dir / "corrupted_extracted_text.txt"
        corrupted_file.write_text("")  # Empty file
        
        extractor = MetadataExtractor(self.input_dir, self.output_dir)
        
        success = extractor.process_file(corrupted_file)
        assert success == False  # Should fail with empty file


if __name__ == "__main__":
    pytest.main([__file__]) 