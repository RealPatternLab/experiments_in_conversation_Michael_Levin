#!/usr/bin/env python3
"""
Tests for 01_sort_and_archive_incoming_files.py

Tests the file sorting and archiving functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step01_sort_and_archive_incoming_files import (
    get_file_extension,
    discover_file_types,
    create_type_directory,
    sort_files_by_type
)


class TestGetFileExtension:
    """Test file extension extraction."""
    
    def test_lowercase_extension(self):
        """Test that extensions are converted to lowercase."""
        assert get_file_extension("file.PDF") == "pdf"
        assert get_file_extension("file.HTML") == "html"
    
    def test_removes_dot(self):
        """Test that the dot is removed from extension."""
        assert get_file_extension("file.pdf") == "pdf"
        assert get_file_extension("file.html") == "html"
    
    def test_handles_no_extension(self):
        """Test files without extensions."""
        assert get_file_extension("file") == ""
        assert get_file_extension("file.") == ""
    
    def test_handles_multiple_dots(self):
        """Test files with multiple dots."""
        assert get_file_extension("file.name.pdf") == "pdf"
        assert get_file_extension("file..pdf") == "pdf"


class TestDiscoverFileTypes:
    """Test file type discovery."""
    
    def setup_method(self):
        """Set up test directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_finds_different_file_types(self):
        """Test that different file types are discovered."""
        # Create test files
        (self.temp_dir / "file1.pdf").touch()
        (self.temp_dir / "file2.html").touch()
        (self.temp_dir / "file3.jpg").touch()
        
        file_types = discover_file_types(self.temp_dir)
        
        assert "pdf" in file_types
        assert "html" in file_types
        assert "jpg" in file_types
        assert len(file_types) == 3
    
    def test_finds_files_in_subdirectories(self):
        """Test that files in subdirectories are found."""
        # Create subdirectory and files
        subdir = self.temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.pdf").touch()
        (self.temp_dir / "file.html").touch()
        
        file_types = discover_file_types(self.temp_dir)
        
        assert "pdf" in file_types
        assert "html" in file_types
        assert len(file_types) == 2
    
    def test_ignores_files_without_extensions(self):
        """Test that files without extensions are ignored."""
        (self.temp_dir / "file1.pdf").touch()
        (self.temp_dir / "file2").touch()  # No extension
        (self.temp_dir / "file3.").touch()  # No extension
        
        file_types = discover_file_types(self.temp_dir)
        
        assert "pdf" in file_types
        assert len(file_types) == 1
    
    def test_handles_case_insensitive_extensions(self):
        """Test that case-insensitive extensions are handled."""
        (self.temp_dir / "file1.pdf").touch()
        (self.temp_dir / "file2.PDF").touch()
        (self.temp_dir / "file3.Pdf").touch()
        
        file_types = discover_file_types(self.temp_dir)
        assert "pdf" in file_types
        assert len(file_types) == 1  # All should be grouped as "pdf"


class TestCreateTypeDirectory:
    """Test directory creation for file types."""
    
    def setup_method(self):
        """Set up test directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_creates_directory_for_new_type(self):
        """Test that new type directories are created."""
        type_dir = create_type_directory(self.temp_dir, "pdf")
        
        assert type_dir.exists()
        assert type_dir.is_dir()
        assert type_dir.name == "raw_pdf"
    
    def test_returns_existing_directory(self):
        """Test that existing directories are returned."""
        # Create directory first
        existing_dir = create_type_directory(self.temp_dir, "pdf")
        
        # Try to create again
        returned_dir = create_type_directory(self.temp_dir, "pdf")
        
        assert returned_dir == existing_dir
        assert returned_dir.exists()
    
    def test_handles_different_file_types(self):
        """Test that different file types get different directories."""
        pdf_dir = create_type_directory(self.temp_dir, "pdf")
        html_dir = create_type_directory(self.temp_dir, "html")
        
        assert pdf_dir != html_dir
        assert pdf_dir.name == "raw_pdf"
        assert html_dir.name == "raw_html"


class TestSortFilesByType:
    """Test the main file sorting functionality."""
    
    def setup_method(self):
        """Set up test directory structure."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.temp_dir / "raw"
        self.base_dir = self.temp_dir
        self.archive_dir = self.temp_dir / "archive"
        self.raw_dir.mkdir()
        self.archive_dir.mkdir()
        
        # Create test files
        (self.raw_dir / "paper1.pdf").touch()
        (self.raw_dir / "paper2.PDF").touch()
        (self.raw_dir / "webpage.html").touch()
        (self.raw_dir / "image.jpg").touch()
        (self.raw_dir / "document.txt").touch()
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_sorts_files_by_type(self):
        """Test that files are sorted into correct type directories."""
        # Create test files
        (self.raw_dir / "document1.pdf").touch()
        (self.raw_dir / "document2.pdf").touch()
        (self.raw_dir / "webpage.html").touch()
        (self.raw_dir / "image.jpg").touch()
        
        result = sort_files_by_type(self.raw_dir, self.base_dir, self.archive_dir)
        
        # Check that files were moved
        assert len(result) == 4
        
        # Check that type directories were created
        assert (self.base_dir / "raw_pdf").exists()
        assert (self.base_dir / "raw_html").exists()
        assert (self.base_dir / "raw_jpg").exists()
        
        # Check that files are in correct directories
        assert (self.base_dir / "raw_pdf" / "document1.pdf").exists()
        assert (self.base_dir / "raw_pdf" / "document2.pdf").exists()
        assert (self.base_dir / "raw_html" / "webpage.html").exists()
        assert (self.base_dir / "raw_jpg" / "image.jpg").exists()
        
        # Check that files were archived
        assert (self.archive_dir / "document1.pdf").exists()
        assert (self.archive_dir / "document2.pdf").exists()
        assert (self.archive_dir / "webpage.html").exists()
        assert (self.archive_dir / "image.jpg").exists()

    def test_returns_sorting_summary(self):
        """Test that the function returns a summary of moved files."""
        # Create test files
        (self.raw_dir / "test1.pdf").touch()
        (self.raw_dir / "test2.pdf").touch()
        (self.raw_dir / "test3.html").touch()
        
        result = sort_files_by_type(self.raw_dir, self.base_dir, self.archive_dir)
        
        # Should return list of moved file paths
        assert len(result) == 3
        assert all(isinstance(path, Path) for path in result)
        
        # Check that files were actually moved
        assert not (self.raw_dir / "test1.pdf").exists()
        assert not (self.raw_dir / "test2.pdf").exists()
        assert not (self.raw_dir / "test3.html").exists()

    def test_handles_empty_directory(self):
        """Test handling of empty input directory."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        result = sort_files_by_type(empty_dir, self.base_dir, self.archive_dir)
        
        assert result == []

    def test_handles_nonexistent_directory(self):
        """Test handling of nonexistent input directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            sort_files_by_type(nonexistent_dir, self.base_dir, self.archive_dir)


if __name__ == "__main__":
    pytest.main([__file__]) 