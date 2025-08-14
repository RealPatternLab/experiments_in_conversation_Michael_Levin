#!/usr/bin/env python3
"""
Script to restore the pipeline progress queue to reflect the current state
of all processed documents.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from pipeline_progress_queue import PipelineProgressQueue

def restore_progress_queue():
    """Restore the progress queue with all processed documents."""
    
    # Initialize progress queue
    progress_queue = PipelineProgressQueue()
    
    # Get all PDF files in processing directory
    processing_dir = Path("processing")
    pdf_files = list(processing_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in processing directory")
    
    # Get all chunk files
    chunks_dir = Path("step_05_semantic_chunks")
    chunk_files = list(chunks_dir.glob("*_chunks.json"))
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Get all metadata files
    metadata_dir = Path("step_02_metadata")
    metadata_files = list(metadata_dir.glob("*_metadata.json"))
    
    print(f"Found {len(metadata_files)} metadata files")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        base_name = pdf_file.stem
        print(f"Processing: {base_name}")
        
        # Find corresponding metadata and chunk files
        metadata_file = metadata_dir / f"{base_name}_metadata.json"
        chunk_file = chunks_dir / f"{base_name}_chunks.json"
        
        if not metadata_file.exists():
            print(f"  ❌ No metadata file found for {base_name}")
            continue
            
        if not chunk_file.exists():
            print(f"  ❌ No chunk file found for {base_name}")
            continue
        
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"  ❌ Error loading metadata: {e}")
            continue
        
        # Load chunks to get count
        try:
            with open(chunk_file, 'r') as f:
                chunks_data = json.load(f)
                chunk_count = len(chunks_data.get('chunks', []))
        except Exception as e:
            print(f"  ❌ Error loading chunks: {e}")
            chunk_count = 0
        
        # Create document ID (using base_name as identifier)
        doc_id = base_name
        
        # Add document to progress queue
        try:
            # Add initial document
            progress_queue.add_document(
                doc_id=doc_id,
                filename=pdf_file.name
            )
            
            # Mark all steps as complete
            progress_queue.update_step_status(doc_id, "step_01_hash_validation", "complete")
            progress_queue.update_step_status(doc_id, "step_02_metadata_extraction", "complete")
            progress_queue.update_step_status(doc_id, "step_03_text_extraction", "complete")
            progress_queue.update_step_status(doc_id, "step_04_metadata_enrichment", "complete")
            progress_queue.update_step_status(doc_id, "step_05_semantic_chunking", "complete")
            
            # Update with additional info
            progress_queue.update_step_status(doc_id, "step_05_semantic_chunking", "complete", {
                "chunk_count": chunk_count,
                "text_length": metadata.get("text_length", 0),
                "new_filename": pdf_file.name,
                "current_status": "ready_for_embedding"
            })
            
            print(f"  ✅ Added {base_name} with {chunk_count} chunks")
            
        except Exception as e:
            print(f"  ❌ Error adding to progress queue: {e}")
    
    print("\nProgress queue restoration complete!")
    
    # Show final status
    summary = progress_queue.get_pipeline_summary()
    if summary:
        print(f"\nPipeline Summary:")
        print(f"  Total documents: {summary.get('total_documents', 0)}")
        print(f"  Ready for embedding: {summary.get('ready_for_embedding', 0)}")
        print(f"  Processing: {summary.get('processing', 0)}")
        print(f"  Pending: {summary.get('pending', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")

if __name__ == "__main__":
    restore_progress_queue()
