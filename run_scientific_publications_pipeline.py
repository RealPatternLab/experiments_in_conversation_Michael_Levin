#!/usr/bin/env python3
"""
Scientific Publications Pipeline Orchestrator

This script orchestrates the processing pipeline for scientific publications (steps 1-8).
The embedding step (step 9) is now handled separately by embed_all_processed_chunks.py
for better operational flexibility and scheduling.

Pipeline Steps:
1. Sort and archive incoming files
2. Detect corruption and sanitize PDFs  
3. Extract quick metadata with Gemini
4. Deduplicate PDFs and move to DLQ
5. Extract full text content from PDFs
6. Extract metadata from extracted text
7. Create semantic chunks from text
8. Enrich metadata with Crossref API

Usage:
    # Run entire pipeline (steps 1-8)
    uv run python3 run_scientific_publications_pipeline.py
    
    # Run from a specific step
    uv run python3 run_scientific_publications_pipeline.py --start-from-step 5
    
    # Run with custom base directory
    uv run python3 run_scientific_publications_pipeline.py --base-dir /path/to/data

Note: After running the pipeline, use embed_all_processed_chunks.py separately to create
the FAISS search index from all processed chunks.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ScientificPublicationsPipelineOrchestrator:
    """Orchestrates the scientific publications processing pipeline (steps 1-8)."""
    
    def __init__(self, pipeline_root: Path):
        self.pipeline_root = pipeline_root
        self.tools_dir = pipeline_root / "tools"
        self.data_dir = pipeline_root / "data"
        
        # Create logs directory
        logs_dir = pipeline_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Define pipeline steps (1-8 only)
        self.pipeline_steps = [
            {
                "name": "Step 1: Sort and Archive Incoming Files",
                "script": "step01_sort_and_archive_incoming_files.py",
                "command": ["uv", "run", "python3", "tools/step01_sort_and_archive_incoming_files.py", "--base-dir", "data"],
                "required_inputs": ["data/source_data/raw"],
                "outputs": ["data/source_data/raw_pdf", "data/source_data/archive"],
                "description": "Sort incoming files, archive them, and route PDFs to raw_pdf/"
            },
            {
                "name": "Step 2: Detect Corruption, Sanitize, and Deduplicate PDFs",
                "script": "step02_detect_corruption_and_sanitize_pdfs.py",
                "command": ["uv", "run", "python3", "tools/step02_detect_corruption_and_sanitize_pdfs.py", "--base-dir", "data/source_data"],
                "required_inputs": ["data/source_data/raw_pdf"],
                "outputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "description": "Detect corruption, sanitize PDFs, and perform hash-based deduplication to prevent reprocessing"
            },
            {
                "name": "Step 3: Extract Quick Metadata with Gemini",
                "script": "step03_extract_quick_metadata_with_gemini.py",
                "command": ["uv", "run", "python3", "tools/step03_extract_quick_metadata_with_gemini.py", "--input-dir", "data/source_data/preprocessed/sanitized/pdfs", "--output-dir", "data/transformed_data/quick_metadata"],
                "required_inputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/transformed_data/quick_metadata"],
                "description": "Extract metadata using Gemini Pro/Flash"
            },
            {
                "name": "Step 4: Deduplicate PDFs and Move to DLQ",
                "script": "step04_deduplicate_pdfs_and_move_to_dlq.py",
                "command": ["uv", "run", "python3", "tools/step04_deduplicate_pdfs_and_move_to_dlq.py", "--metadata-dir", "data/transformed_data/quick_metadata", "--pdf-dir", "data/source_data/preprocessed/sanitized/pdfs", "--dlq-dir", "data/source_data/DLQ"],
                "required_inputs": ["data/transformed_data/quick_metadata", "data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/source_data/DLQ"],
                "description": "Remove duplicate PDFs based on metadata"
            },
            {
                "name": "Step 5: Extract Full Text Content from PDFs",
                "script": "step05_extract_full_text_content_from_pdfs.py",
                "command": ["uv", "run", "python3", "tools/step05_extract_full_text_content_from_pdfs.py", "--ingested-dir", "data/source_data/preprocessed/sanitized/pdfs", "--extracted-text-dir", "data/transformed_data/extracted_text"],
                "required_inputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/transformed_data/extracted_text"],
                "description": "Extract text content from PDFs"
            },
            {
                "name": "Step 6: Extract Metadata from Extracted Text",
                "script": "step06_extract_metadata_from_extracted_text.py",
                "command": ["uv", "run", "python3", "tools/step06_extract_metadata_from_extracted_text.py", "--input-dir", "data/transformed_data/extracted_text", "--output-dir", "data/transformed_data/metadata_extraction"],
                "required_inputs": ["data/transformed_data/extracted_text"],
                "outputs": ["data/transformed_data/metadata_extraction"],
                "description": "Extract metadata using rule-based methods"
            },
            {
                "name": "Step 7: Create Semantic Chunks from Text",
                "script": "step07_create_semantic_chunks_from_text.py",
                "command": ["uv", "run", "python3", "tools/step07_create_semantic_chunks_from_text.py", "--input-dir", "data/transformed_data/extracted_text", "--output-dir", "data/transformed_data/semantic_chunks"],
                "required_inputs": ["data/transformed_data/extracted_text"],
                "outputs": ["data/transformed_data/semantic_chunks"],
                "description": "Create semantic chunks using Gemini"
            },
            {
                "name": "Step 8: Enrich Metadata with Crossref API",
                "script": "step08_enrich_metadata_with_crossref_api.py",
                "command": ["uv", "run", "python3", "tools/step08_enrich_metadata_with_crossref_api.py", "--input-dir", "data/transformed_data/quick_metadata", "--output-dir", "data/transformed_data/metadata_enrichment"],
                "required_inputs": ["data/transformed_data/quick_metadata"],
                "outputs": ["data/transformed_data/metadata_enrichment"],
                "description": "Enrich metadata using Crossref and Unpaywall APIs"
            }
        ]
    
    def create_directory_structure(self):
        """Create all necessary directories for the pipeline."""
        logger.info("📁 Creating directory structure...")
        
        directories = [
            "data/source_data/raw",
            "data/source_data/raw_pdf", 
            "data/source_data/archive",
            "data/source_data/DLQ",
            "data/source_data/preprocessed/sanitized/pdfs",
            "data/transformed_data/quick_metadata",
            "data/transformed_data/extracted_text",
            "data/transformed_data/metadata_extraction",
            "data/transformed_data/semantic_chunks",
            "data/transformed_data/metadata_enrichment",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.pipeline_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"   Created: {directory}")
        
        logger.info("✅ Directory structure created")
    
    def check_prerequisites(self) -> bool:
        """Check if all required tools and dependencies are available."""
        logger.info("🔍 Checking prerequisites...")
        
        # Check if tools directory exists
        if not self.tools_dir.exists():
            logger.error(f"Tools directory not found: {self.tools_dir}")
            return False
        
        # Check if all step scripts exist
        for step in self.pipeline_steps:
            script_path = self.tools_dir / step["script"]
            if not script_path.exists():
                logger.error(f"Step script not found: {script_path}")
                return False
        
        # Check if data directory exists
        if not self.data_dir.exists():
            logger.info("Data directory not found, creating...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Prerequisites check passed")
        return True
    
    def run_pipeline_step(self, step: Dict, step_number: int) -> bool:
        """Run a single pipeline step."""
        logger.info(f"🚀 Running {step['name']}...")
        logger.info(f"   Description: {step['description']}")
        
        try:
            # Change to pipeline root directory for relative path resolution
            original_cwd = Path.cwd()
            os.chdir(self.pipeline_root)
            
            # Run the command
            result = subprocess.run(
                step["command"],
                capture_output=True,
                text=True,
                cwd=self.pipeline_root
            )
            
            # Restore original working directory
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                logger.info(f"✅ {step['name']} completed successfully")
                if result.stdout.strip():
                    logger.info(f"   Output: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ {step['name']} failed with return code {result.returncode}")
                if result.stderr.strip():
                    logger.error(f"   Error: {result.stderr.strip()}")
                if result.stdout.strip():
                    logger.info(f"   Output: {result.stdout.strip()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running {step['name']}: {e}")
            return False
    
    def run_pipeline(self, start_from_step: int = 1) -> bool:
        """Run the pipeline from the specified step."""
        logger.info("🚀 Starting Scientific Publications Pipeline...")
        logger.info(f"Pipeline will run from step {start_from_step} to step 8")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("❌ Prerequisites check failed")
            return False
        
        # Run pipeline steps
        for i, step in enumerate(self.pipeline_steps, 1):
            if i < start_from_step:
                logger.info(f"⏭️  Skipping {step['name']} (start from step {start_from_step})")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {i}/8: {step['name']}")
            logger.info(f"{'='*60}")
            
            if not self.run_pipeline_step(step, i):
                logger.error(f"❌ Pipeline failed at step {i}")
                return False
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📝 Next steps:")
        logger.info("   1. Review processed data in data/transformed_data/")
        logger.info("   2. Run embed_all_processed_chunks.py to create search index")
        logger.info("   3. Test the Streamlit application")
        logger.info(f"{'='*60}")
        
        return True


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Scientific Publications Pipeline Orchestrator (Steps 1-8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run entire pipeline (steps 1-8)
    uv run python3 run_scientific_publications_pipeline.py
    
    # Run from step 5 onwards
    uv run python3 run_scientific_publications_pipeline.py --start-from-step 5
    
    # Run with custom base directory
    uv run python3 run_scientific_publications_pipeline.py --base-dir /path/to/data

Note: This pipeline processes PDFs through steps 1-8. After completion, 
run embed_all_processed_chunks.py separately to create the FAISS search index.
        """
    )
    
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        choices=range(1, 9),
        help="Start pipeline from this step number (default: 1)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for the pipeline (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Validate start step
    if args.start_from_step < 1 or args.start_from_step > 8:
        logger.error("Start step must be between 1 and 8")
        sys.exit(1)
    
    # Get pipeline root directory
    pipeline_root = Path(args.base_dir).resolve()
    if not pipeline_root.exists():
        logger.error(f"Base directory does not exist: {pipeline_root}")
        sys.exit(1)
    
    logger.info(f"🔧 Pipeline root: {pipeline_root}")
    logger.info(f"🚀 Starting from step: {args.start_from_step}")
    
    try:
        orchestrator = ScientificPublicationsPipelineOrchestrator(pipeline_root)
        success = orchestrator.run_pipeline(args.start_from_step)
        
        if success:
            logger.info("✅ Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Pipeline failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import os
    main() 