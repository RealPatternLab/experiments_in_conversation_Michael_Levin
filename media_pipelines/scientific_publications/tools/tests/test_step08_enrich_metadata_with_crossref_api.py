#!/usr/bin/env python3
"""
Tests for 08_enrich_metadata_with_crossref_api.py

Tests the external API enrichment functionality using CrossRef and Unpaywall APIs.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step08_enrich_metadata_with_crossref_api import (
    CrossRefEnricher, UnpaywallEnricher, MetadataEnricher
)


class TestCrossRefEnricher:
    """Test the CrossRefEnricher class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.enricher = CrossRefEnricher(email="test@example.com")
    
    def test_init_with_email(self):
        """Test CrossRefEnricher initialization with email."""
        enricher = CrossRefEnricher(email="test@example.com")
        assert enricher.email == "test@example.com"
        assert enricher.base_url == "https://api.crossref.org"
    
    def test_init_without_email(self):
        """Test CrossRefEnricher initialization without email."""
        # Clear environment variable for this test
        with patch.dict(os.environ, {'PIPELINE_EMAIL': ''}):
            enricher = CrossRefEnricher()
            assert enricher.email is None
            assert enricher.base_url == "https://api.crossref.org"
    
    def test_init_with_env_email(self):
        """Test CrossRefEnricher initialization with environment email."""
        with patch.dict(os.environ, {'PIPELINE_EMAIL': 'env@example.com'}):
            enricher = CrossRefEnricher()
            assert enricher.email == "env@example.com"
    
    @patch('step08_enrich_metadata_with_crossref_api.requests.get')
    def test_search_by_doi_success(self, mock_get):
        """Test successful DOI search."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": ["Test Paper Title"],
            "author": [{"given": "John", "family": "Doe"}],
            "abstract": "Test abstract",
            "published-print": {"date-parts": [[2023, 1, 1]]},
            "container-title": ["Test Journal"]
        }
        mock_get.return_value = mock_response
        
        result = self.enricher.search_by_doi("10.1234/test.2023.001")
        
        assert result is not None
        assert result["title"] == "Test Paper Title"
        assert result["authors"] == ["John Doe"]
        assert result["abstract"] == "Test abstract"
        assert result["publication_date"] == "2023-01-01"
        assert result["journal"] == "Test Journal"
    
    @patch('step08_enrich_metadata_with_crossref_api.requests.get')
    def test_search_by_doi_not_found(self, mock_get):
        """Test DOI search when not found."""
        # Mock not found response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.enricher.search_by_doi("10.1234/notfound.2023.001")
        
        assert result is None
    
    def test_search_by_doi_invalid_format(self):
        """Test DOI search with invalid format."""
        result = self.enricher.search_by_doi("invalid-doi-format")
        
        assert result is None
    
    @patch('step08_enrich_metadata_with_crossref_api.requests.get')
    def test_search_by_title_success(self, mock_get):
        """Test successful title search."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "items": [{
                    "title": ["Test Paper Title"],
                    "author": [{"given": "John", "family": "Doe"}],
                    "abstract": "Test abstract",
                    "published-print": {"date-parts": [[2023, 1, 1]]},
                    "container-title": ["Test Journal"]
                }]
            }
        }
        mock_get.return_value = mock_response
        
        result = self.enricher.search_by_title("Test Paper Title")
        
        assert result is not None
        assert result["title"] == "Test Paper Title"
        assert result["authors"] == ["John Doe"]
    
    def test_parse_crossref_response(self):
        """Test parsing of CrossRef API response."""
        response_data = {
            "title": ["Test Paper Title"],
            "author": [{"given": "John", "family": "Doe"}],
            "abstract": "Test abstract",
            "published-print": {"date-parts": [[2023, 1, 1]]},
            "container-title": ["Test Journal"]
        }
        
        result = self.enricher._parse_crossref_response(response_data)
        
        assert result["title"] == "Test Paper Title"
        assert result["authors"] == ["John Doe"]
        assert result["abstract"] == "Test abstract"
        assert result["publication_date"] == "2023-01-01"
        assert result["journal"] == "Test Journal"
    
    def test_parse_crossref_response_missing_fields(self):
        """Test parsing of CrossRef response with missing fields."""
        response_data = {
            "title": ["Test Paper Title"]
            # Missing other fields
        }
        
        result = self.enricher._parse_crossref_response(response_data)
        
        assert result["title"] == "Test Paper Title"
        assert result["authors"] == []
        assert result["abstract"] is None
        assert result["publication_date"] is None
        assert result["journal"] is None


class TestUnpaywallEnricher:
    """Test the UnpaywallEnricher class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.enricher = UnpaywallEnricher(email="test@example.com")
    
    def test_init_with_email(self):
        """Test UnpaywallEnricher initialization with email."""
        enricher = UnpaywallEnricher(email="test@example.com")
        assert enricher.email == "test@example.com"
        assert enricher.base_url == "https://api.unpaywall.org/v2"
    
    @patch('step08_enrich_metadata_with_crossref_api.requests.get')
    def test_search_by_doi_success(self, mock_get):
        """Test successful DOI search."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://example.com/paper.pdf",
                "url_for_landing_page": "https://example.com/paper",
                "host_type": "repository"
            },
            "publisher": "Test Publisher",
            "journal_name": "Test Journal",
            "published_date": "2023-01-01",
            "license": "CC-BY"
        }
        mock_get.return_value = mock_response
        
        result = self.enricher.search_by_doi("10.1234/test.2023.001")
        
        assert result is not None
        assert result["is_open_access"] is True
        assert result["pdf_url"] == "https://example.com/paper.pdf"
        assert result["landing_page_url"] == "https://example.com/paper"
        assert result["host_type"] == "repository"
        assert result["publisher"] == "Test Publisher"
        assert result["journal_name"] == "Test Journal"
        assert result["published_date"] == "2023-01-01"
        assert result["license"] == "CC-BY"
    
    @patch('step08_enrich_metadata_with_crossref_api.requests.get')
    def test_search_by_doi_not_found(self, mock_get):
        """Test DOI search when not found."""
        # Mock not found response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = self.enricher.search_by_doi("10.1234/notfound.2023.001")
        
        assert result is None
    
    def test_parse_unpaywall_response(self):
        """Test parsing of Unpaywall API response."""
        response_data = {
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://example.com/paper.pdf",
                "url_for_landing_page": "https://example.com/paper",
                "host_type": "repository"
            },
            "publisher": "Test Publisher",
            "journal_name": "Test Journal",
            "published_date": "2023-01-01",
            "license": "CC-BY"
        }
        
        result = self.enricher._parse_unpaywall_response(response_data)
        
        assert result["is_open_access"] is True
        assert result["pdf_url"] == "https://example.com/paper.pdf"
        assert result["landing_page_url"] == "https://example.com/paper"
        assert result["host_type"] == "repository"
        assert result["publisher"] == "Test Publisher"
        assert result["journal_name"] == "Test Journal"
        assert result["published_date"] == "2023-01-01"
        assert result["license"] == "CC-BY"


class TestMetadataEnricher:
    """Test the MetadataEnricher class."""
    
    def setup_method(self):
        """Set up test directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create test metadata files
        self.test_metadata_file = self.input_dir / "paper1_metadata.json"
        self.test_metadata_file.write_text(json.dumps({
            "filename": "paper1_extracted_text.txt",
            "title": "Test Paper Title",
            "authors": ["John Doe"],
            "doi": "10.1234/test.2023.001",
            "extraction_method": "rule_based",
            "extraction_confidence": "medium"
        }))
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates necessary directories."""
        enricher = MetadataEnricher(self.input_dir, self.output_dir)
        
        assert self.output_dir.exists()
        assert (Path("logs")).exists()
    
    def test_get_metadata_files_to_process(self):
        """Test metadata file discovery and processing logic."""
        # Create multiple metadata files
        (self.input_dir / "paper1_metadata.json").touch()
        (self.input_dir / "paper2_metadata.json").touch()
        (self.input_dir / "paper3_metadata.json").touch()
        
        enricher = MetadataEnricher(self.input_dir, self.output_dir)
        
        # Test processing all files
        metadata_files = enricher.get_metadata_files_to_process()
        assert len(metadata_files) == 3
        
        # Test max_files limit
        metadata_files_limited = enricher.get_metadata_files_to_process(max_files=2)
        assert len(metadata_files_limited) == 2
        
        # Test already processed files
        (self.output_dir / "paper1_metadata_enriched.json").touch()
        metadata_files_processed = enricher.get_metadata_files_to_process()
        assert len(metadata_files_processed) == 2  # paper1 already processed
    
    @patch('step08_enrich_metadata_with_crossref_api.CrossRefEnricher')
    @patch('step08_enrich_metadata_with_crossref_api.UnpaywallEnricher')
    def test_enrich_metadata_success(self, mock_unpaywall_class, mock_crossref_class):
        """Test successful metadata enrichment."""
        # Mock enrichers
        mock_crossref = MagicMock()
        mock_crossref.search_by_doi.return_value = {
            "title": "Enhanced Title",
            "authors": ["John Doe", "Jane Smith"],
            "abstract": "Enhanced abstract",
            "publication_date": "2023-01-01",
            "journal": "Enhanced Journal"
        }
        mock_crossref_class.return_value = mock_crossref
        
        mock_unpaywall = MagicMock()
        mock_unpaywall.search_by_doi.return_value = {
            "is_open_access": True,
            "pdf_url": "https://example.com/paper.pdf"
        }
        mock_unpaywall_class.return_value = mock_unpaywall
        
        enricher = MetadataEnricher(self.input_dir, self.output_dir)
        
        success = enricher.enrich_metadata(self.test_metadata_file)
        assert success == True
        
        # Check that enriched file was created
        enriched_file = self.output_dir / "paper1_metadata_enriched.json"
        assert enriched_file.exists()
        
        # Check enriched content
        with open(enriched_file, 'r') as f:
            enriched_data = json.load(f)
            assert enriched_data["title"] == "Enhanced Title"
            assert "Jane Smith" in enriched_data["authors"]
            assert enriched_data["is_open_access"] is True
            assert enriched_data["pdf_url"] == "https://example.com/paper.pdf"
    
    def test_enrich_metadata_no_doi(self):
        """Test metadata enrichment when no DOI exists."""
        # Create metadata without DOI
        no_doi_file = self.input_dir / "paper2_metadata.json"
        no_doi_file.write_text(json.dumps({
            "filename": "paper2_extracted_text.txt",
            "title": "Test Paper Title",
            "authors": ["John Doe"],
            "extraction_method": "rule_based"
        }))
        
        enricher = MetadataEnricher(self.input_dir, self.output_dir)
        
        success = enricher.enrich_metadata(no_doi_file)
        assert success == False  # Should fail without DOI
    
    def test_process_files(self):
        """Test batch file processing."""
        # Create multiple metadata files
        (self.input_dir / "paper1_metadata.json").touch()
        (self.input_dir / "paper2_metadata.json").touch()
        
        enricher = MetadataEnricher(self.input_dir, self.output_dir)
        
        # Mock the enrichment method
        with patch.object(enricher, 'enrich_metadata') as mock_enrich:
            mock_enrich.return_value = True
            
            results = enricher.process_files()
            
            assert results["total"] == 2
            assert results["processed"] == 2
            assert results["failed"] == 0
    
    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        enricher = MetadataEnricher(empty_dir, self.output_dir)
        metadata_files = enricher.get_metadata_files_to_process()
        
        assert len(metadata_files) == 0
    
    def test_handles_nonexistent_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        # The tool should handle this gracefully and return empty list
        enricher = MetadataEnricher(nonexistent_dir, self.output_dir)
        metadata_files = enricher.get_metadata_files_to_process()
        
        assert len(metadata_files) == 0


if __name__ == "__main__":
    pytest.main([__file__]) 