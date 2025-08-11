#!/usr/bin/env python3
"""
Scientific Publications Pipeline

This script orchestrates the complete pipeline for processing scientific publications (PDFs).
It processes PDFs from raw input through text extraction, metadata extraction, chunking, and embedding.

Pipeline Overview:
1. File Organization & Archiving
2. File Sanitization
3. Text Extraction
4. Metadata Extraction & Enrichment
5. Semantic Chunking
6. Vector Embedding

Usage:
    python process_scientific_publications.py [--max-files N] [--file FILE] [--reset]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import logging
import shutil

def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).parent.parent
    dirs = [
        base_dir / "data" / "source_data" / "raw",
        base_dir / "data" / "source_data" / "archive",
        base_dir / "data" / "source_data" / "DLQ",
        base_dir / "data" / "source_data" / "preprocessed" / "sanitized" / "pdfs",
        base_dir / "data" / "transformed_data" / "metadata_extraction",
        base_dir / "data" / "transformed_data" / "semantic_chunking",
        base_dir / "data" / "transformed_data" / "text_extraction",
        base_dir / "data" / "transformed_data" / "vector_embeddings",
        base_dir / "pipeline" / "logs"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def setup_logging():
    """Configure logging after directories are created."""
    base_dir = Path(__file__).parent.parent
    logs_dir = base_dir / "pipeline" / "logs"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / 'scientific_publications_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def run_command(command: str, description: str, logger) -> bool:
    """Run a command and log the result."""
    logger.info(f"🔄 {description}")
    logger.info(f"📝 Command: {command}")
    
    try:
        # Run command with real-time output instead of capturing
        result = subprocess.run(command, shell=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCCESS")
            return True
        else:
            logger.error(f"❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"❌ {description} - EXCEPTION: {e}")
        return False

def archive_raw_files(logger):
    """Archive all files from raw directory to archive directory."""
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "source_data" / "raw"
    archive_dir = base_dir / "data" / "source_data" / "archive"
    
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        logger.info("📁 Raw directory is empty, nothing to archive")
        return True
    
    logger.info("📦 Archiving raw files...")
    
    # Copy all files to archive
    for file_path in raw_dir.iterdir():
        if file_path.is_file():
            archive_path = archive_dir / file_path.name
            if archive_path.exists():
                # Add timestamp if file already exists
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                name_parts = file_path.stem, timestamp, file_path.suffix
                archive_path = archive_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
            
            shutil.copy2(file_path, archive_path)
            logger.info(f"📦 Archived: {file_path.name}")
    
    return True

def move_non_pdfs_to_dlq(logger):
    """Move all non-PDF files to DLQ directory."""
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "source_data" / "raw"
    dlq_dir = base_dir / "data" / "source_data" / "DLQ"
    
    if not raw_dir.exists():
        logger.info("📁 Raw directory doesn't exist")
        return True
    
    logger.info("🚫 Moving non-PDF files to DLQ...")
    
    for file_path in raw_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() != '.pdf':
            dlq_path = dlq_dir / file_path.name
            if dlq_path.exists():
                # Add timestamp if file already exists
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                name_parts = file_path.stem, timestamp, file_path.suffix
                dlq_path = dlq_dir / f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
            
            shutil.move(file_path, dlq_path)
            logger.info(f"🚫 Moved to DLQ: {file_path.name}")
    
    return True

def main():
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(description="Scientific Publications Pipeline")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--file", type=str, help="Process specific file only")
    parser.add_argument("--reset", action="store_true", help="Reset pipeline (clear all processed data)")
    
    args = parser.parse_args()
    
    # Setup directories first
    setup_directories()
    
    # Setup logging after directories are created
    logger = setup_logging()
    
    logger.info("🚀 Starting Scientific Publications Pipeline")
    logger.info("=" * 60)
    
    # Track overall success
    pipeline_success = True
    
    # ============================================================================
    # PHASE 1: FILE ORGANIZATION & ARCHIVING
    # ============================================================================
    
    logger.info("📋 PHASE 1: File Organization & Archiving")
    logger.info("-" * 50)
    
    # Step 1: Archive all raw files
    success = archive_raw_files(logger)
    if not success:
        pipeline_success = False
    
    # Step 2: Move non-PDFs to DLQ
    success = move_non_pdfs_to_dlq(logger)
    if not success:
        pipeline_success = False
    
    # ============================================================================
    # PHASE 2: FILE SANITIZATION
    # ============================================================================
    
    logger.info("📋 PHASE 2: File Sanitization")
    logger.info("-" * 50)
    
    # Step 3: Sanitize PDF files
    tools_dir = Path(__file__).parent.parent / "tools"
    base_dir = Path(__file__).parent.parent
    success = run_command(
        f"python {tools_dir}/sanitize_files.py --base-dir {base_dir}/data/source_data",
        "Sanitize PDF files and move to preprocessed directory",
        logger
    )
    if not success:
        pipeline_success = False
    
    # ============================================================================
    # PHASE 3: CONTENT PROCESSING
    # ============================================================================
    
    logger.info("📋 PHASE 3: Content Processing")
    logger.info("-" * 50)
    
    # Step 4: Extract text from PDFs
    success = run_command(
        f"python {tools_dir}/extract_text_from_pdf.py --ingested-dir {base_dir}/data/source_data/preprocessed/sanitized/pdfs --extracted-text-dir {base_dir}/data/transformed_data/text_extraction",
        "Extract text content from sanitized PDFs",
        logger
    )
    if not success:
        pipeline_success = False
    
    # Step 5: Extract metadata from PDFs
    success = run_command(
        f"python {tools_dir}/extract_metadata_from_pdf.py --ingested-dir {base_dir}/data/source_data/preprocessed/sanitized/pdfs --output-dir {base_dir}/data/transformed_data/metadata_extraction",
        "Extract metadata from PDFs using multiple strategies",
        logger
    )
    if not success:
        pipeline_success = False
    
    # Step 6: Enrich metadata with Gemini
    success = run_command(
        f"python {tools_dir}/enrich_metadata_with_gemini.py --input-dir {base_dir}/data/transformed_data/metadata_extraction --output-dir {base_dir}/data/transformed_data/metadata_extraction",
        "Enrich low-confidence metadata using Gemini Pro",
        logger
    )
    if not success:
        pipeline_success = False
    
    # Step 7: Enrich metadata with CrossRef
    success = run_command(
        f"python {tools_dir}/enrich_metadata_with_crossref.py --input-dir {base_dir}/data/transformed_data/metadata_extraction --output-dir {base_dir}/data/transformed_data/metadata_extraction",
        "Enrich metadata using CrossRef and Unpaywall APIs",
        logger
    )
    if not success:
        pipeline_success = False
    
    # ============================================================================
    # PHASE 4: CHUNKING & EMBEDDING
    # ============================================================================
    
    logger.info("📋 PHASE 4: Chunking & Embedding")
    logger.info("-" * 50)
    
    # Step 8: Generate chunking prompt
    # COMMENTED OUT: We already have the chunking prompt file
    # success = run_command(
    #     f"python {tools_dir}/generate_chunking_prompt.py",
    #     "Generate optimal prompt for semantic chunking",
    #     logger
    # )
    # if not success:
    #     pipeline_success = False

    # Step 9: Generate semantic chunks (using existing prompt)
    success = run_command(
        f"python {tools_dir}/chunk_extracted_text.py --input-dir {base_dir}/data/transformed_data/text_extraction --output-dir {base_dir}/data/transformed_data/semantic_chunking --chunking-prompt {base_dir}/data/transformed_data/chunking_prompt.txt",
        "Generate semantic chunks from extracted text using existing prompt",
        logger
    )
    if not success:
        pipeline_success = False

    # Step 10: Create vector embeddings
    success = run_command(
        f"python {tools_dir}/embed_semantic_chunks_faiss.py --input-dir {base_dir}/data/transformed_data/semantic_chunking --output-dir {base_dir}/data/transformed_data/vector_embedding",
        "Create vector embeddings for semantic chunks",
        logger
    )
    if not success:
        pipeline_success = False
    
    # ============================================================================
    # PIPELINE SUMMARY
    # ============================================================================
    
    logger.info("📋 PIPELINE SUMMARY")
    logger.info("-" * 50)
    
    if pipeline_success:
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("📊 All phases completed. Scientific publications are ready for RAG applications.")
    else:
        logger.error("❌ PIPELINE COMPLETED WITH ERRORS!")
        logger.error("🔍 Check the logs above for specific failure points.")
    
    logger.info("=" * 60)
    return pipeline_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 