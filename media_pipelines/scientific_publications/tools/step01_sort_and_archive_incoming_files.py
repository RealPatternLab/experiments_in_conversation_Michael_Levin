#!/usr/bin/env python3
"""
Sort Files by Type Tool

This tool sorts files from a landing directory into type-specific raw_* directories
and archives them for permanent storage.

Updated for the new scientific publications pipeline structure.
"""

import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Any
import re


def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (lowercase, without dot)
    """
    return Path(filename).suffix.lower().lstrip('.')


def discover_file_types(raw_dir: Path) -> Set[str]:
    """
    Discover all file types in the raw directory.
    
    Args:
        raw_dir: Directory to search for files
        
    Returns:
        Set of file extensions found
    """
    if not raw_dir.exists():
        return set()
    
    extensions = set()
    
    # Find all files recursively
    for file_path in raw_dir.rglob('*'):
        if file_path.is_file():
            ext = get_file_extension(file_path.name)
            if ext:  # Only include files with extensions
                extensions.add(ext)
    
    return extensions


def create_type_directory(base_dir: Path, file_type: str, dry_run: bool = False) -> Path:
    """
    Create a directory for a specific file type.
    
    Args:
        base_dir: Base directory to create the type directory in
        file_type: Type of file (e.g., 'pdf', 'html')
        dry_run: If True, don't actually create the directory
        
    Returns:
        Path to the created directory
    """
    type_dir = base_dir / f"raw_{file_type}"
    
    if not dry_run:
        type_dir.mkdir(parents=True, exist_ok=True)
    
    return type_dir


def archive_file(file_path: Path, archive_dir: Path, dry_run: bool = False) -> Path:
    """
    Archive a file with timestamp suffix if duplicate exists.
    
    Args:
        file_path: Path to the file to archive
        archive_dir: Directory to archive the file in
        dry_run: If True, don't actually copy the file
        
    Returns:
        Path to the archived file
    """
    archive_path = archive_dir / file_path.name
    
    # If file already exists, add timestamp suffix
    if archive_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name_parts = file_path.stem, timestamp, file_path.suffix
        archive_path = archive_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
    
    if not dry_run:
        shutil.copy2(file_path, archive_path)
    
    return archive_path


def sort_files_by_type(
    raw_dir: Path, 
    base_dir: Path, 
    archive_dir: Path,
    dry_run: bool = False
) -> List[Path]:
    """
    Sort files by type into raw_* directories and archive them.
    
    Args:
        raw_dir: Directory containing files to sort
        base_dir: Base directory to create raw_* directories in
        archive_dir: Directory to archive files in
        dry_run: If True, don't actually move files
        
    Returns:
        List of paths to moved files
    """
    if not raw_dir.exists():
        print(f"❌ Raw directory does not exist: {raw_dir}")
        return []
    
    # Discover all file types
    file_types = discover_file_types(raw_dir)
    if not file_types:
        print("ℹ️  No files found to sort")
        return []
    
    print(f"🔍 Found file types: {', '.join(sorted(file_types))}")
    
    # Create archive directory
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
    
    moved_files = []
    current_time = datetime.now()
    
    # Process each file type
    for file_type in sorted(file_types):
        print(f"\n📁 Processing {file_type} files...")
        
        # Create type-specific directory
        type_dir = create_type_directory(base_dir, file_type, dry_run)
        
        # Find all files of this type (case-insensitive)
        pattern = f"*.{file_type}"
        files = list(raw_dir.glob(pattern.lower()))  # Check lowercase
        files.extend(list(raw_dir.glob(pattern.upper())))  # Also check uppercase
        
        if not files:
            print(f"    ℹ️  No {file_type} files found")
            continue
        
        print(f"    📄 Found {len(files)} {file_type} files")
        
        for file_path in files:
            try:
                # Archive the file
                archive_path = archive_file(file_path, archive_dir, dry_run)
                
                # Move to type-specific directory
                target_path = type_dir / file_path.name
                
                if dry_run:
                    print(f"    🔍 Would move: {file_path.name} → {target_path}")
                    moved_files.append(target_path)
                else:
                    shutil.move(str(file_path), str(target_path))
                    print(f"    ✅ Moved successfully: {file_path.name}")
                    moved_files.append(target_path)
                    
            except Exception as e:
                print(f"    ❌ Failed to move: {e}")
    
    return moved_files


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Sort files by type into raw_* directories"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/source_data/raw"),
        help="Directory containing files to sort (default: data/source_data/raw)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/source_data"),
        help="Base directory to create raw_* directories in (default: data/source_data)"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/source_data/archive"),
        help="Directory to archive files in (default: data/source_data/archive)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    try:
        print(f"📁 Sorting files from {args.raw_dir}")
        if args.dry_run:
            print("🔍 DRY RUN MODE - No files will be moved")
        
        moved_files = sort_files_by_type(
            args.raw_dir, args.base_dir, args.archive_dir, args.dry_run
        )
        
        if moved_files:
            print(f"\n✅ Sorting complete!")
            print(f"📊 Moved {len(moved_files)} files")
        else:
            print("ℹ️  No files to sort")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 