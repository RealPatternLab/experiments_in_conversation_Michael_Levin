#!/usr/bin/env python3
"""
Scientific Publications Pipeline Orchestrator

This script orchestrates the complete scientific publications processing pipeline,
running each step sequentially with proper error handling and logging.

Usage:
    python3 run_scientific_publications_pipeline.py                    # Run all steps
    python3 run_scientific_publications_pipeline.py --start-from-step 5  # Start from step 5
    python3 run_scientific_publications_pipeline.py --dry-run          # Show what would be run
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/pipeline_orchestrator.log')
    ]
)
logger = logging.getLogger(__name__)

class ScientificPublicationsPipelineOrchestrator:
    """Orchestrates the scientific publications processing pipeline."""
    
    def __init__(self, pipeline_root: Path):
        self.pipeline_root = pipeline_root
        self.tools_dir = pipeline_root / "tools"
        self.data_dir = pipeline_root / "data"
        
        # Create logs directory
        logs_dir = pipeline_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Define pipeline steps with default arguments
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
                "name": "Step 2: Detect Corruption and Sanitize PDFs",
                "script": "step02_detect_corruption_and_sanitize_pdfs.py",
                "command": ["uv", "run", "python3", "tools/step02_detect_corruption_and_sanitize_pdfs.py", "--base-dir", "data/source_data"],
                "required_inputs": ["data/source_data/raw_pdf"],
                "outputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "description": "Sanitize PDFs and move to preprocessed directory"
            },
            {
                "name": "Step 3: Extract Quick Metadata with Gemini",
                "script": "step03_extract_quick_metadata_with_gemini.py",
                "command": ["uv", "run", "python3", "tools/step03_extract_quick_metadata_with_gemini.py", "--input-dir", "data/source_data/preprocessed/sanitized/pdfs", "--output-dir", "data/transformed_data/quick_metadata"],
                "required_inputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/transformed_data/quick_metadata"],
                "description": "Extract quick metadata using Gemini Pro"
            },
            {
                "name": "Step 4: Deduplicate PDFs and Move to DLQ",
                "script": "step04_deduplicate_pdfs_and_move_to_dlq.py",
                "command": ["uv", "run", "python3", "tools/step04_deduplicate_pdfs_and_move_to_dlq.py", "--metadata-dir", "data/transformed_data/quick_metadata", "--pdf-dir", "data/source_data/preprocessed/sanitized/pdfs", "--dlq-dir", "data/source_data/DLQ"],
                "required_inputs": ["data/transformed_data/quick_metadata", "data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/source_data/DLQ"],
                "description": "Deduplicate PDFs based on metadata and move duplicates to DLQ"
            },
            {
                "name": "Step 5: Extract Full Text Content from PDFs",
                "script": "step05_extract_full_text_content_from_pdfs.py",
                "command": ["uv", "run", "python3", "tools/step05_extract_full_text_content_from_pdfs.py", "--ingested-dir", "data/source_data/preprocessed/sanitized/pdfs", "--extracted-text-dir", "data/transformed_data/text_extraction"],
                "required_inputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/transformed_data/text_extraction"],
                "description": "Extract full text content from sanitized PDFs"
            },
            {
                "name": "Step 6: Extract Metadata from Extracted Text",
                "script": "step06_extract_metadata_from_extracted_text.py",
                "command": ["uv", "run", "python3", "tools/step06_extract_metadata_from_extracted_text.py", "--input-dir", "data/transformed_data/text_extraction", "--output-dir", "data/transformed_data/metadata_extraction"],
                "required_inputs": ["data/transformed_data/text_extraction"],
                "outputs": ["data/transformed_data/metadata_extraction"],
                "description": "Extract metadata using rule-based methods from extracted text"
            },
            {
                "name": "Step 7: Create Semantic Chunks from Text",
                "script": "step07_create_semantic_chunks_from_text.py",
                "command": ["uv", "run", "python3", "tools/step07_create_semantic_chunks_from_text.py", "--input-dir", "data/transformed_data/text_extraction", "--output-dir", "data/transformed_data/semantic_chunks"],
                "required_inputs": ["data/transformed_data/text_extraction"],
                "outputs": ["data/transformed_data/semantic_chunks"],
                "description": "Create semantic chunks using Gemini 2.5 Flash with metadata enrichment"
            },
            {
                "name": "Step 8: Enrich Metadata with Crossref API",
                "script": "step08_enrich_metadata_with_crossref_api.py",
                "command": ["uv", "run", "python3", "tools/step08_enrich_metadata_with_crossref_api.py", "--input-dir", "data/transformed_data/quick_metadata", "--output-dir", "data/transformed_data/metadata_enrichment"],
                "required_inputs": ["data/transformed_data/quick_metadata"],
                "outputs": ["data/transformed_data/metadata_enrichment"],
                "description": "Enrich metadata using Crossref and Unpaywall APIs"
            },
            {
                "name": "Step 9: Generate Vector Embeddings for Chunks",
                "script": "step09_generate_vector_embeddings_for_chunks.py",
                "command": ["uv", "run", "python3", "tools/step09_generate_vector_embeddings_for_chunks.py", "--input-dir", "data/transformed_data/semantic_chunks", "--output-dir", "data/transformed_data/vector_embeddings"],
                "required_inputs": ["data/transformed_data/semantic_chunks"],
                "outputs": ["data/transformed_data/vector_embeddings"],
                "description": "Generate vector embeddings and create FAISS index"
            }
        ]
    
    def check_prerequisites(self) -> bool:
        """Check if all required tools and directories exist."""
        logger.info("🔍 Checking pipeline prerequisites...")
        
        # Check if tools directory exists
        if not self.tools_dir.exists():
            logger.error(f"❌ Tools directory not found: {self.tools_dir}")
            return False
        
        # Check if all pipeline scripts exist
        missing_scripts = []
        for step in self.pipeline_steps:
            script_path = self.tools_dir / step["script"]
            if not script_path.exists():
                missing_scripts.append(step["script"])
        
        if missing_scripts:
            logger.error(f"❌ Missing pipeline scripts: {missing_scripts}")
            return False
        
        # Check if uv is available
        try:
            subprocess.run(["uv", "--version"], capture_output=True, check=True)
            logger.info("✅ uv is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ uv is not available. Please install uv first.")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def create_directory_structure(self):
        """Create the required directory structure."""
        logger.info("📁 Creating directory structure...")
        
        directories = [
            "data/source_data/raw",
            "data/source_data/raw_pdf", 
            "data/source_data/archive",
            "data/source_data/DLQ",
            "data/source_data/preprocessed/sanitized/pdfs",
            "data/transformed_data/quick_metadata",
            "data/transformed_data/text_extraction",
            "data/transformed_data/metadata_extraction",
            "data/transformed_data/semantic_chunks",
            "data/transformed_data/metadata_enrichment",
            "data/transformed_data/vector_embeddings",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.pipeline_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"   📁 Created: {directory}")
        
        logger.info("✅ Directory structure created")
    
    def run_pipeline_step(self, step: Dict, dry_run: bool = False) -> bool:
        """Run a single pipeline step."""
        logger.info(f"🚀 {step['name']}")
        logger.info(f"   📝 {step['description']}")
        
        if dry_run:
            logger.info(f"   🔍 Would run: {' '.join(step['command'])}")
            return True
        
        try:
            # Run the command
            result = subprocess.run(
                step['command'],
                cwd=self.pipeline_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"   ✅ {step['name']} completed successfully")
            if result.stdout:
                logger.debug(f"   📄 Output: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ {step['name']} failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"   📄 Stdout: {e.stdout}")
            if e.stderr:
                logger.error(f"   ❌ Stderr: {e.stderr}")
            return False
        
        except Exception as e:
            logger.error(f"   ❌ Unexpected error in {step['name']}: {e}")
            return False
    
    def run_pipeline(self, start_from_step: int = 1, dry_run: bool = False) -> bool:
        """Run the complete pipeline or from a specific step."""
        logger.info("🚀 Starting Scientific Publications Pipeline")
        logger.info(f"📍 Pipeline root: {self.pipeline_root}")
        
        if dry_run:
            logger.info("🔍 DRY RUN MODE - No actual processing will occur")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Create directory structure if not dry run
        if not dry_run:
            self.create_directory_structure()
        
        # Run pipeline steps
        successful_steps = 0
        failed_steps = 0
        
        for i, step in enumerate(self.pipeline_steps, 1):
            if i < start_from_step:
                logger.info(f"⏭️  Skipping {step['name']} (before start-from-step {start_from_step})")
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {i}/{len(self.pipeline_steps)}: {step['name']}")
            logger.info(f"{'='*60}")
            
            if self.run_pipeline_step(step, dry_run):
                successful_steps += 1
                logger.info(f"✅ Step {i} completed successfully")
            else:
                failed_steps += 1
                logger.error(f"❌ Step {i} failed")
                if not dry_run:
                    logger.error("Pipeline stopped due to step failure")
                    break
            
            # Add a small delay between steps
            if not dry_run and i < len(self.pipeline_steps):
                logger.info("⏳ Waiting 2 seconds before next step...")
                time.sleep(2)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📋 PIPELINE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successful steps: {successful_steps}")
        logger.info(f"❌ Failed steps: {failed_steps}")
        logger.info(f"📊 Total steps: {len(self.pipeline_steps)}")
        
        if failed_steps == 0:
            logger.info("🎉 Pipeline completed successfully!")
            return True
        else:
            logger.error(f"💥 Pipeline failed with {failed_steps} failed steps")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Scientific Publications Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python3 run_scientific_publications_pipeline.py
  
  # Start from step 5
  python3 run_scientific_publications_pipeline.py --start-from-step 5
  
  # Dry run to see what would be executed
  python3 run_scientific_publications_pipeline.py --dry-run
  
  # Run from specific step with dry run
  python3 run_scientific_publications_pipeline.py --start-from-step 3 --dry-run
        """
    )
    
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        help="Start pipeline from this step number (default: 1)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually executing"
    )
    
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory of the pipeline (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Validate start-from-step
    if args.start_from_step < 1 or args.start_from_step > 9:
        logger.error("❌ start-from-step must be between 1 and 9")
        sys.exit(1)
    
    try:
        # Create and run the pipeline
        orchestrator = ScientificPublicationsPipelineOrchestrator(args.pipeline_root)
        success = orchestrator.run_pipeline(
            start_from_step=args.start_from_step,
            dry_run=args.dry_run
        )
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 