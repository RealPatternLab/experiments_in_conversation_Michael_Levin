#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def restore_final_half():
    temp_dir = Path("../archive_final_half")
    archive_dir = Path("media_pipelines/scientific_publications/data/source_data/archive")
    
    # Get all PDF files from temp directory
    pdf_files = list(temp_dir.glob("*.pdf"))
    total_pdfs = len(pdf_files)
    
    print(f"Total PDFs in temp directory: {total_pdfs}")
    print(f"Moving all PDFs back to archive...")
    
    # Move all files back
    moved_count = 0
    for pdf_file in pdf_files:
        try:
            shutil.move(str(pdf_file), str(archive_dir / pdf_file.name))
            moved_count += 1
            if moved_count % 50 == 0:
                print(f"Moved {moved_count} files...")
        except Exception as e:
            print(f"Error moving {pdf_file.name}: {e}")
    
    print(f"Successfully moved {moved_count} PDFs back to archive")
    print(f"Total PDFs now in archive: {len(list(archive_dir.glob('*.pdf')))}")
    
    # Remove the empty temp directory
    try:
        temp_dir.rmdir()
        print(f"Removed empty temp directory: {temp_dir}")
    except Exception as e:
        print(f"Could not remove temp directory: {e}")

if __name__ == "__main__":
    restore_final_half() 