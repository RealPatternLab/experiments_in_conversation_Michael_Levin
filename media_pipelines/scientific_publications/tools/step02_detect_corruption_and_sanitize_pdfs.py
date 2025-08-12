#!/usr/bin/env python3
"""
Sanitize Files Tool

This tool processes files from raw_* directories, detects corruption,
and moves them to preprocessed/sanitized/* directories with sanitized names.

Updated for the new scientific publications pipeline structure.
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import re


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


def generate_safe_filename(file_type: str) -> str:
    """
    Generate a safe, unique filename with timestamp.
    
    Args:
        file_type: Type of file (e.g., 'pdf', 'html')
        
    Returns:
        Safe filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{file_type}_{timestamp}.{file_type}"


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


def process_file_type(
    raw_dir: Path, 
    target_dir: Path, 
    file_type: str, 
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Process files of a specific type from raw directory to target directory.
    
    Args:
        raw_dir: Directory containing raw files
        target_dir: Directory to move processed files to
        file_type: Type of files to process
        dry_run: If True, don't actually move files
        
    Returns:
        Tuple of (processed_count, corrupted_count)
    """
    processed = 0
    corrupted = 0
    
    if not raw_dir.exists():
        print(f"    ❌ Raw directory does not exist: {raw_dir}")
        return processed, corrupted
    
    # Create target directory
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all files of this type
    pattern = f"*.{file_type}"
    files = list(raw_dir.glob(pattern.lower()))  # Check lowercase
    files.extend(list(raw_dir.glob(pattern.upper())))  # Also check uppercase
    
    if not files:
        print(f"    ℹ️  No {file_type} files found")
        return processed, corrupted
    
    print(f"    📄 Processing {len(files)} {file_type} files...")
    
    for file_path in files:
        try:
            # Check if file is corrupted
            if is_file_corrupted(file_path):
                print(f"    🚫 Corrupted file detected: {file_path.name}")
                corrupted += 1
                continue
            
            # Generate safe filename
            safe_filename = generate_safe_filename(file_type)
            target_path = target_dir / safe_filename
            
            if dry_run:
                print(f"    🔍 Would process: {file_path.name} → {safe_filename}")
                processed += 1
            else:
                # Copy file to target directory
                shutil.copy2(file_path, target_path)
                print(f"    ✅ Processed: {file_path.name} → {safe_filename}")
                
                # Delete original file
                file_path.unlink()
                
                processed += 1
            
        except Exception as e:
            print(f"    ❌ Error processing {file_path.name}: {e}")
            continue
    
    return processed, corrupted


def sanitize_files(base_dir: Path, dry_run: bool = False) -> Dict[str, Dict[str, int]]:
    """
    Sanitize all files in raw_* directories.
    
    Args:
        base_dir: Base directory containing raw_* directories
        dry_run: If True, don't actually move files
        
    Returns:
        Dictionary with results for each file type
    """
    results = {}
    
    if not base_dir.exists():
        print(f"❌ Base directory does not exist: {base_dir}")
        return results
    
    # Find all raw_* directories
    raw_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('raw_')]
    
    if not raw_dirs:
        print("ℹ️  No raw_* directories found")
        return results
    
    print(f"🔍 Found {len(raw_dirs)} raw directories to process")
    
    for raw_dir in raw_dirs:
        file_type = raw_dir.name[4:]  # Remove 'raw_' prefix
        target_dir = base_dir / "preprocessed" / "sanitized" / f"{file_type}s"
        
        print(f"\n📁 Processing {file_type} files from {raw_dir.name}...")
        
        processed, corrupted = process_file_type(
            raw_dir, target_dir, file_type, dry_run
        )
        
        results[file_type] = {
            "processed": processed,
            "corrupted": corrupted
        }
        
        if processed > 0 or corrupted > 0:
            print(f"    📊 Processed: {processed}, Corrupted: {corrupted}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Sanitize files from raw_* directories"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/source_data"),
        help="Base directory containing raw_* directories (default: data/source_data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"🧹 Sanitizing files in {args.base_dir}")
        if args.dry_run:
            print("🔍 DRY RUN MODE - No files will be moved")
        
        results = sanitize_files(args.base_dir, args.dry_run)
        
        if results:
            print(f"\n✅ Sanitization complete!")
            total_processed = sum(r["processed"] for r in results.values())
            total_corrupted = sum(r["corrupted"] for r in results.values())
            print(f"📊 Total processed: {total_processed}, Total corrupted: {total_corrupted}")
        else:
            print("ℹ️  No files to process")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 