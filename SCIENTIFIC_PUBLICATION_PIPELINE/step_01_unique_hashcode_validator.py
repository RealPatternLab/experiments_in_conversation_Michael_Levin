#!/usr/bin/env python3
"""
Unique Hashcode Validator

This script validates file uniqueness by checking hash codes against stored records.
If the file has a unique hash, it generates a new filename and proceeds.
If the file is a duplicate, it logs the match and deletes the file.
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

# Progress queue integration
try:
    from pipeline_progress_queue import PipelineProgressQueue
    PROGRESS_QUEUE_AVAILABLE = True
except ImportError:
    PROGRESS_QUEUE_AVAILABLE = False
    logging.warning("Progress queue not available. Pipeline tracking will be limited.")

# PDF validation imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2 not available. PDF validation will be limited.")


class UniqueHashcodeValidator:
    """Validates file uniqueness and manages file processing."""
    
    def __init__(self, hash_database_path: str = "hash_database.txt"):
        self.hash_database_path = Path(hash_database_path)
        self.logger = self._setup_logging()
        self._ensure_hash_database()
        
        # Initialize progress queue if available
        self.progress_queue = None
        if PROGRESS_QUEUE_AVAILABLE:
            try:
                self.progress_queue = PipelineProgressQueue()
                self.logger.info("Progress queue initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize progress queue: {e}")
                self.progress_queue = None
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler
        file_handler = logging.FileHandler('hash_validation.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _ensure_hash_database(self) -> None:
        """Ensure the hash database file exists."""
        if not self.hash_database_path.exists():
            self.hash_database_path.touch()
            self.logger.info(f"Created new hash database: {self.hash_database_path}")
    
    def validate_pdf_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate that a PDF file is readable and not corrupted.
        
        Args:
            file_path: Path to the PDF file to validate
            
        Returns:
            Tuple of (is_valid, error_message) where error_message is empty if valid
        """
        if not PDF_AVAILABLE:
            self.logger.warning("PyPDF2 not available, skipping PDF validation")
            return True, ""
        
        try:
            with open(file_path, 'rb') as f:
                # Check PDF header (should start with %PDF-)
                header = f.read(1024)
                if not header.startswith(b'%PDF-'):
                    return False, "Invalid PDF header - file does not start with %PDF-"
                
                # Reset file pointer for PyPDF2
                f.seek(0)
                
                # Try to read with PyPDF2 to check for corruption
                try:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    # Check if PDF has pages
                    if len(pdf_reader.pages) == 0:
                        return False, "PDF has no readable pages"
                    
                    # Try to extract text from first page to test readability
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    
                    # Check if text extraction produced meaningful content
                    if not text or len(text.strip()) < 10:
                        return False, "PDF text extraction failed or produced insufficient content"
                    
                    return True, ""
                    
                except Exception as e:
                    return False, f"PDF parsing failed: {str(e)}"
                    
        except Exception as e:
            return False, f"File reading error: {str(e)}"
    
    def generate_file_hash(self, file_path: str) -> str:
        """
        Generate SHA-256 hash of file content.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            SHA-256 hash string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256()
                
                # Read file in chunks to handle large files efficiently
                chunk_size = 8192  # 8KB chunks
                while chunk := f.read(chunk_size):
                    file_hash.update(chunk)
                
                hash_string = file_hash.hexdigest()
                self.logger.debug(f"Generated hash {hash_string} for file {file_path}")
                
                return hash_string
                
        except IOError as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            raise IOError(f"Failed to read file {file_path}: {e}")
    
    def check_hash_exists(self, file_hash: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a file hash already exists in the database.
        
        Args:
            file_hash: SHA-256 hash to check
            
        Returns:
            Tuple of (exists, matching_filename) where matching_filename is None if no match
        """
        try:
            with open(self.hash_database_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        stored_hash, stored_filename = line.split('|', 1)
                        if stored_hash == file_hash:
                            self.logger.info(f"Hash {file_hash} matches existing file: {stored_filename}")
                            return True, stored_filename
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error checking hash database: {e}")
            # If we can't check the database, assume hash doesn't exist to avoid data loss
            return False, None
    
    def record_hash(self, file_hash: str, filename: str) -> None:
        """
        Record a new file hash in the database.
        
        Args:
            file_hash: SHA-256 hash to record
            filename: Filename associated with the hash
        """
        try:
            with open(self.hash_database_path, 'a') as f:
                f.write(f"{file_hash}|{filename}\n")
            self.logger.info(f"Recorded hash {file_hash} for file {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to record hash {file_hash}: {e}")
            raise
    
    def generate_unique_filename(self, file_hash: str, original_filename: str) -> str:
        """
        Generate a unique filename using hash and timestamp.
        
        Args:
            file_hash: SHA-256 hash of the file
            original_filename: Original filename for extension preservation
            
        Returns:
            New unique filename in format: hash_datetime.ext
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        extension = Path(original_filename).suffix
        
        # Use first 8 characters of hash for readability
        short_hash = file_hash[:8]
        
        new_filename = f"{short_hash}_{timestamp}{extension}"
        self.logger.debug(f"Generated unique filename: {new_filename} from {original_filename}")
        
        return new_filename
    
    def process_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Process a file through the hash validation pipeline.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Tuple of (is_unique, new_filename) where new_filename is None if file was deleted
        """
        file_path = Path(file_path)
        original_filename = file_path.name
        
        self.logger.info(f"Processing file: {original_filename}")
        
        # Add document to progress queue if available
        doc_id = None
        if self.progress_queue:
            try:
                # Generate a unique document ID based on filename and timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                doc_id = f"{file_path.stem}_{timestamp}"
                
                if self.progress_queue.add_document(doc_id, original_filename):
                    self.logger.info(f"Added document {doc_id} to progress queue")
                else:
                    self.logger.warning(f"Failed to add document {doc_id} to progress queue")
                    doc_id = None
            except Exception as e:
                self.logger.warning(f"Progress queue error: {e}")
                doc_id = None
        
        try:
            # Step 1: Validate PDF file
            is_valid, error_message = self.validate_pdf_file(str(file_path))
            if not is_valid:
                self.logger.error(f"PDF validation failed for {original_filename}: {error_message}")
                
                # Update progress queue if available
                if self.progress_queue and doc_id:
                    try:
                        self.progress_queue.add_error(doc_id, "step_01_hash_validation", error_message)
                    except Exception as e:
                        self.logger.warning(f"Failed to update progress queue: {e}")
                
                # Delete corrupted/unreadable file
                os.remove(file_path)
                self.logger.info(f"Deleted corrupted file: {original_filename}")
                return False, None
            
            self.logger.info(f"PDF validation passed for {original_filename}")
            
            # Step 2: Generate hash
            file_hash = self.generate_file_hash(str(file_path))
            self.logger.info(f"Generated hash: {file_hash}")
            
            # Step 3: Check if hash exists
            hash_exists, matching_filename = self.check_hash_exists(file_hash)
            
            if hash_exists:
                # File is a duplicate - log and delete
                self.logger.warning(
                    f"File {original_filename} is a copy of previously processed file: {matching_filename}"
                )
                
                # Update progress queue if available
                if self.progress_queue and doc_id:
                    try:
                        self.progress_queue.add_error(
                            doc_id, 
                            "step_01_hash_validation", 
                            f"Duplicate file - matches {matching_filename}"
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to update progress queue: {e}")
                
                # Delete the duplicate file
                os.remove(file_path)
                self.logger.info(f"Deleted duplicate file: {original_filename}")
                
                return False, None  # File was not unique, was deleted
            else:
                # File is unique - generate new filename and record hash
                new_filename = self.generate_unique_filename(file_hash, original_filename)
                self.record_hash(file_hash, new_filename)
                
                # Move the file to processing directory with unique filename
                processing_dir = Path("processing")
                processing_dir.mkdir(exist_ok=True)
                new_file_path = processing_dir / new_filename
                file_path.rename(new_file_path)
                
                self.logger.info(
                    f"File {original_filename} is unique. Moved to processing as: {new_filename}"
                )
                
                # Update progress queue if available
                if self.progress_queue and doc_id:
                    try:
                        self.progress_queue.update_step_status(
                            doc_id, 
                            "step_01_hash_validation", 
                            "complete",
                            {"new_filename": new_filename}
                        )
                        self.logger.info(f"Updated progress queue for {doc_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to update progress queue: {e}")
                
                return True, new_filename  # File is unique, proceed to next step
                
        except Exception as e:
            self.logger.error(f"Error processing file {original_filename}: {e}")
            raise


def main():
    """Main function to run the hash validator on files in the raw directory."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unique Hashcode Validator")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="step_01_raw",
        help="Directory containing files to validate (default: step_01_raw)"
    )
    parser.add_argument(
        "--hash-db",
        type=str,
        default="hash_database.txt",
        help="Path to hash database file (default: hash_database.txt)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create validator
    validator = UniqueHashcodeValidator(args.hash_db)
    
    # Process files in input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    
    # Get list of files to process
    files_to_process = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.pdf':
            files_to_process.append(file_path)
    
    if not files_to_process:
        print(f"No PDF files found in {input_dir}")
        return 0
    
    print(f"Found {len(files_to_process)} PDF files to process")
    print("=" * 60)
    
    # Process each file
    unique_files = 0
    duplicate_files = 0
    corrupted_files = 0
    
    for file_path in files_to_process:
        try:
            is_unique, new_filename = validator.process_file(str(file_path))
            
            if is_unique:
                unique_files += 1
                print(f"✓ {file_path.name} → {new_filename} (UNIQUE)")
            elif new_filename is None:
                corrupted_files += 1
                print(f"✗ {file_path.name} → DELETED (CORRUPTED/DUPLICATE)")
            else:
                duplicate_files += 1
                print(f"✗ {file_path.name} → DELETED (DUPLICATE)")
                
        except Exception as e:
            print(f"✗ {file_path.name} → ERROR: {e}")
    
    print("=" * 60)
    print(f"Processing complete:")
    print(f"  Unique files: {unique_files}")
    print(f"  Duplicate files: {duplicate_files}")
    print(f"  Corrupted files: {corrupted_files}")
    print(f"  Total processed: {len(files_to_process)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
