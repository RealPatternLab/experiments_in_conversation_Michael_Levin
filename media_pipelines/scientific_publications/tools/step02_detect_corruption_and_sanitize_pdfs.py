#!/usr/bin/env python3
"""
PDF Corruption Detection, Sanitization, and Deduplication Tool

This tool processes PDFs from raw_pdf directories, detects corruption,
performs hash-based deduplication, and moves unique PDFs to 
preprocessed/sanitized/pdfs with sanitized names.

Updated for the new scientific publications pipeline structure with
hash-based deduplication to prevent reprocessing of identical files.
"""

import argparse
import shutil
import sys
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Set
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HashRegistry:
    """Manages persistent storage of file hashes for deduplication."""
    
    def __init__(self, registry_file: Path):
        self.registry_file = registry_file
        self.processed_hashes: Set[str] = set()
        self.load_registry()
    
    def load_registry(self):
        """Load existing hash registry from file."""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                    self.processed_hashes = set(data.get('processed_hashes', []))
                logger.info(f"📚 Loaded {len(self.processed_hashes)} existing hashes from registry")
            else:
                logger.info("🆕 Creating new hash registry")
        except Exception as e:
            logger.warning(f"⚠️ Could not load hash registry: {e}")
            self.processed_hashes = set()
    
    def save_registry(self):
        """Save current hash registry to file."""
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, 'w') as f:
                json.dump({
                    'processed_hashes': list(self.processed_hashes),
                    'last_updated': datetime.now().isoformat(),
                    'total_files': len(self.processed_hashes)
                }, f, indent=2)
            logger.debug(f"💾 Saved hash registry with {len(self.processed_hashes)} hashes")
        except Exception as e:
            logger.error(f"❌ Failed to save hash registry: {e}")
    
    def is_already_processed(self, file_hash: str) -> bool:
        """Check if a file hash has been processed before."""
        return file_hash in self.processed_hashes
    
    def add_processed_hash(self, file_hash: str):
        """Add a file hash to the processed registry."""
        self.processed_hashes.add(file_hash)
        self.save_registry()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hash registry."""
        return {
            'total_processed': len(self.processed_hashes),
            'registry_file': str(self.registry_file),
            'last_updated': datetime.now().isoformat()
        }


def get_file_hash(file_path: Path) -> str:
    """
    Generate SHA-256 hash of file content.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        SHA-256 hash string
    """
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256()
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    except Exception as e:
        logger.error(f"❌ Failed to hash file {file_path}: {e}")
        return ""


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove problematic filesystem characters
    problematic_chars = '<>:"/\\|?*'
    for char in problematic_chars:
        filename = filename.replace(char, '')
    
    # Remove extra whitespace
    filename = ' '.join(filename.split())
    
    # Remove extra dots
    filename = re.sub(r'\.+', '.', filename)
    
    return filename


def generate_safe_filename(original_name: str, file_hash: str) -> str:
    """
    Generate a safe, unique filename with hash and timestamp.
    
    Args:
        original_name: Original filename
        file_hash: File hash for uniqueness
        file_type: Type of file (e.g., 'pdf')
        
    Returns:
        Safe filename with hash and timestamp
    """
    # Extract file extension
    file_type = original_name.split('.')[-1] if '.' in original_name else 'pdf'
    
    # Create base name from original (without extension)
    base_name = original_name.rsplit('.', 1)[0] if '.' in original_name else original_name
    base_name = sanitize_filename(base_name)
    
    # Use first 8 characters of hash for uniqueness
    hash_suffix = file_hash[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    return f"{base_name}_{hash_suffix}_{timestamp}.{file_type}"


def is_file_corrupted(file_path: Path) -> bool:
    """
    Check if a file is corrupted by attempting to read it.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if file is corrupted, False otherwise
    """
    try:
        # Try to read the file
        with open(file_path, 'rb') as f:
            f.read(1024)  # Read first 1KB
        return False
    except Exception:
        return True


def process_pdfs_with_deduplication(
    raw_pdf_dir: Path, 
    target_dir: Path, 
    hash_registry: HashRegistry,
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """
    Process PDFs with hash-based deduplication.
    
    Args:
        raw_pdf_dir: Directory containing raw PDF files
        target_dir: Directory to move processed files to
        hash_registry: Hash registry for deduplication
        dry_run: If True, don't actually move files
        
    Returns:
        Tuple of (processed_count, corrupted_count, duplicate_count)
    """
    processed = 0
    corrupted = 0
    duplicates = 0
    
    if not raw_pdf_dir.exists():
        logger.error(f"❌ Raw PDF directory does not exist: {raw_pdf_dir}")
        return processed, corrupted, duplicates
    
    # Get all PDF files
    pdf_files = list(raw_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"ℹ️ No PDF files found in {raw_pdf_dir}")
        return processed, corrupted, duplicates
    
    logger.info(f"🔍 Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        try:
            logger.debug(f"📄 Processing {pdf_path.name}")
            
            # Check for corruption first
            if is_file_corrupted(pdf_path):
                logger.warning(f"⚠️ Corrupted file detected: {pdf_path.name}")
                if not dry_run:
                    pdf_path.unlink()  # Delete corrupted file
                corrupted += 1
                continue
            
            # Generate file hash for deduplication
            file_hash = get_file_hash(pdf_path)
            if not file_hash:
                logger.error(f"❌ Failed to hash file: {pdf_path.name}")
                continue
            
            # Check if this file has been processed before
            if hash_registry.is_already_processed(file_hash):
                logger.info(f"⏭️ Skipping duplicate: {pdf_path.name} (hash: {file_hash[:8]}...)")
                if not dry_run:
                    pdf_path.unlink()  # Remove duplicate file
                duplicates += 1
                continue
            
            # Process new, unique file
            if not dry_run:
                # Generate safe filename with hash
                safe_filename = generate_safe_filename(pdf_path.name, file_hash)
                target_path = target_dir / safe_filename
                
                # Ensure target directory exists
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Move file to target directory
                shutil.move(str(pdf_path), str(target_path))
                
                # Add hash to registry
                hash_registry.add_processed_hash(file_hash)
                
                logger.info(f"✅ Processed: {pdf_path.name} → {safe_filename}")
            else:
                logger.info(f"🔍 Would process: {pdf_path.name} (hash: {file_hash[:8]}...)")
            
            processed += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            continue
    
    return processed, corrupted, duplicates


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Detect corruption, sanitize, and deduplicate PDFs using hash-based comparison"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/source_data"),
        help="Base directory containing raw_pdf directory (default: data/source_data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info(f"🔍 Starting PDF corruption detection, sanitization, and deduplication")
        logger.info(f"📍 Base directory: {args.base_dir}")
        
        if args.dry_run:
            logger.info("🔍 DRY RUN MODE - No files will be moved")
        
        # Setup paths
        raw_pdf_dir = args.base_dir / "raw_pdf"
        target_dir = args.base_dir / "preprocessed" / "sanitized" / "pdfs"
        registry_file = args.base_dir / "preprocessed" / "hash_registry.json"
        
        # Initialize hash registry
        hash_registry = HashRegistry(registry_file)
        
        # Process PDFs with deduplication
        processed, corrupted, duplicates = process_pdfs_with_deduplication(
            raw_pdf_dir, target_dir, hash_registry, args.dry_run
        )
        
        # Display results
        logger.info(f"\n📊 Processing Results:")
        logger.info(f"   ✅ Processed (new): {processed}")
        logger.info(f"   ⚠️ Corrupted: {corrupted}")
        logger.info(f"   ⏭️ Duplicates (skipped): {duplicates}")
        logger.info(f"   📚 Total in registry: {hash_registry.get_stats()['total_processed']}")
        
        if processed > 0 or corrupted > 0 or duplicates > 0:
            logger.info(f"\n✅ Processing complete!")
        else:
            logger.info("ℹ️ No files to process")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 