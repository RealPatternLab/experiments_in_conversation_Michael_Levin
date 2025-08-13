#!/usr/bin/env python3
"""
Scientific Publications Pipeline Orchestrator

This script orchestrates the complete scientific publications processing pipeline,
integrating with the pipeline tracker for comprehensive monitoring and control.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from pipeline_tracker import PipelineTracker

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
    
    def __init__(self, base_dir: Path = Path("data/source_data"), force_reprocess: bool = False):
        self.base_dir = base_dir
        self.force_reprocess = force_reprocess
        
        # Initialize pipeline tracker
        self.tracker = PipelineTracker(base_dir)
        
        # Pipeline steps configuration
        self.pipeline_steps = [
            {
                "name": "Step 1: Sort and Archive Incoming Files",
                "script": "step01_sort_and_archive_incoming_files.py",
                "command": ["uv", "run", "python3", "tools/step01_sort_and_archive_incoming_files.py", "--base-dir", "data/source_data"],
                "required_inputs": ["data/source_data/raw"],
                "outputs": ["data/source_data/raw_pdf"],
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
                "description": "Extract quick metadata using Gemini Pro"
            },
            {
                "name": "Step 4: Deduplicate PDFs and Move to DLQ",
                "script": "step04_deduplicate_pdfs_and_move_to_dlq.py",
                "command": ["uv", "run", "python3", "tools/step04_deduplicate_pdfs_and_move_to_dlq.py", "--metadata-dir", "data/transformed_data/quick_metadata", "--pdf-dir", "data/source_data/preprocessed/sanitized/pdfs", "--dlq-dir", "data/source_data/DLQ"],
                "required_inputs": ["data/transformed_data/quick_metadata", "data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/source_data/preprocessed/sanitized/pdfs"],
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
        """Check if all prerequisites are met."""
        logger.info("🔍 Checking pipeline prerequisites...")
        
        # Check if uv is available
        try:
            import subprocess
            result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ uv is available")
            else:
                logger.error("❌ uv is not available")
                return False
        except Exception as e:
            logger.error(f"❌ Error checking uv: {e}")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    def create_directory_structure(self):
        """Create all necessary directories."""
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
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"   📁 Created: {directory}")
        
        logger.info("✅ Directory structure created")
    
    def run_pipeline_step(self, step_info: dict, step_number: int) -> bool:
        """Run a single pipeline step."""
        step_name = step_info["name"]
        script_name = step_info["script"]
        command = step_info["command"]
        
        logger.info(f"🚀 {step_name}")
        logger.info(f"   📝 {step_info['description']}")
        
        # Start tracking this step
        input_files = self._get_input_files(step_info["required_inputs"])
        self.tracker.start_step(step_name, step_number, input_files)
        
        try:
            # Execute the command
            import subprocess
            result = subprocess.run(command, capture_output=True, text=True, cwd=self.base_dir.parent)
            
            if result.returncode == 0:
                # Check outputs
                output_files = self._get_output_files(step_info["outputs"])
                self.tracker.end_step(step_number, "completed", output_files)
                logger.info(f"   ✅ {step_name} completed successfully")
                return True
            else:
                error_msg = f"Command failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr}"
                
                self.tracker.end_step(step_number, "failed", error_message=error_msg)
                logger.error(f"   ❌ {step_name} failed: {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Exception during execution: {str(e)}"
            self.tracker.end_step(step_number, "failed", error_message=error_msg)
            logger.error(f"   ❌ {step_name} failed with exception: {e}")
            return False
    
    def _get_input_files(self, input_dirs: list) -> list:
        """Get list of input files for a step."""
        input_files = []
        for input_dir in input_dirs:
            dir_path = Path(input_dir)
            if dir_path.exists():
                for file_path in dir_path.glob("*"):
                    if file_path.is_file():
                        input_files.append(str(file_path))
        return input_files
    
    def _get_output_files(self, output_dirs: list) -> list:
        """Get list of output files for a step."""
        output_files = []
        for output_dir in output_dirs:
            dir_path = Path(output_dir)
            if dir_path.exists():
                for file_path in dir_path.glob("*"):
                    if file_path.is_file():
                        output_files.append(str(file_path))
        return output_files
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        logger.info("🚀 Starting Scientific Publications Pipeline")
        logger.info(f"📍 Pipeline root: {self.base_dir.parent}")
        
        # Start pipeline run tracking
        run_id = self.tracker.start_pipeline_run()
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.tracker.end_pipeline_run(run_id, "failed")
            return False
        
        # Create directory structure
        self.create_directory_structure()
        
        # Run each step
        successful_steps = 0
        failed_steps = 0
        
        for i, step_info in enumerate(self.pipeline_steps, 1):
            logger.info("=" * 60)
            logger.info(f"Step {i}/9: {step_info['name']}")
            logger.info("=" * 60)
            
            if self.run_pipeline_step(step_info, i):
                successful_steps += 1
                logger.info(f"✅ Step {i} completed successfully")
            else:
                failed_steps += 1
                logger.error(f"❌ Step {i} failed")
                break  # Stop pipeline on first failure
            
            # Wait between steps (except after the last one)
            if i < len(self.pipeline_steps):
                logger.info("⏳ Waiting 2 seconds before next step...")
                time.sleep(2)
        
        # Pipeline summary
        logger.info("=" * 60)
        logger.info("📋 PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Successful steps: {successful_steps}")
        logger.info(f"❌ Failed steps: {failed_steps}")
        logger.info(f"📊 Total steps: {len(self.pipeline_steps)}")
        
        if failed_steps == 0:
            logger.info("🎉 Pipeline completed successfully!")
            self.tracker.end_pipeline_run(run_id, "completed")
            return True
        else:
            logger.error("💥 Pipeline failed!")
            self.tracker.end_pipeline_run(run_id, "failed")
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
  
  # Force reprocessing of all files
  python3 run_scientific_publications_pipeline.py --force-reprocess
  
  # Run with custom base directory
  python3 run_scientific_publications_pipeline.py --base-dir /path/to/data
        """
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/source_data"),
        help="Base directory for pipeline data (default: data/source_data)"
    )
    
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of all files (ignore completion status)"
    )
    
    args = parser.parse_args()
    
    try:
        orchestrator = ScientificPublicationsPipelineOrchestrator(
            base_dir=args.base_dir,
            force_reprocess=args.force_reprocess
        )
        
        success = orchestrator.run_pipeline()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 