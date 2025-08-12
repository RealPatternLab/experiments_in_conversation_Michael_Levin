#!/usr/bin/env python3
"""
Scientific Publications Pipeline - Complete Orchestration Script

This script runs the entire scientific publications pipeline from start to finish,
processing PDFs through all stages: ingestion → sanitization → metadata → deduplication → 
text extraction → chunking → embeddings.

Usage:
    python run_pipeline.py [--max-files N] [--dry-run] [--verbose] [--start-from-step N]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import json


class PipelineOrchestrator:
    """Orchestrates the complete scientific publications pipeline."""
    
    def __init__(self, max_files: Optional[int] = None, dry_run: bool = False, verbose: bool = False):
        self.max_files = max_files
        self.dry_run = dry_run
        self.verbose = verbose
        self.start_time = time.time()
        self.step_results = {}
        
        # Pipeline configuration - must be set before setup_logging
        self.pipeline_root = Path(__file__).parent
        self.tools_dir = self.pipeline_root / "tools"
        self.data_dir = self.pipeline_root / "data"
        
        # Setup logging
        self.setup_logging()
        
        # Pipeline steps configuration
        self.pipeline_steps = [
            {
                "name": "File Sorting & Archiving",
                "script": "step01_sort_and_archive_incoming_files.py",
                "description": "Sort incoming files by type and archive them",
                "required_inputs": ["data/source_data/raw"],
                "outputs": ["data/source_data/archive", "data/source_data/raw_pdf"],
                "optional": True  # Skip if no files in raw/
            },
            {
                "name": "PDF Sanitization",
                "script": "step02_detect_corruption_and_sanitize_pdfs.py",
                "description": "Detect corruption and sanitize PDF filenames",
                "required_inputs": ["data/source_data/raw_pdf"],
                "outputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "optional": False
            },
            {
                "name": "Quick Metadata Extraction",
                "script": "step03_extract_quick_metadata_with_gemini.py",
                "description": "Extract metadata from first 3 pages using Gemini for deduplication",
                "required_inputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/transformed_data/quick_metadata"],
                "optional": False
            },
            {
                "name": "PDF Deduplication",
                "script": "step04_deduplicate_pdfs_and_move_to_dlq.py",
                "description": "Identify and remove duplicate PDFs",
                "required_inputs": ["data/source_data/preprocessed/sanitized/pdfs", "data/transformed_data/quick_metadata"],
                "outputs": ["data/source_data/DLQ"],
                "optional": False
            },
            {
                "name": "Full Text Extraction",
                "script": "step05_extract_full_text_content_from_pdfs.py",
                "description": "Extract complete text content from PDFs",
                "required_inputs": ["data/source_data/preprocessed/sanitized/pdfs"],
                "outputs": ["data/transformed_data/extracted_text"],
                "optional": False
            },
            {
                "name": "Metadata Extraction",
                "script": "step06_extract_metadata_from_extracted_text.py",
                "description": "Extract metadata from extracted text",
                "required_inputs": ["data/transformed_data/extracted_text"],
                "outputs": ["data/transformed_data/metadata_extraction"],
                "optional": False
            },
            {
                "name": "Semantic Chunking",
                "script": "step07_create_semantic_chunks_from_text.py",
                "description": "Create semantic chunks from extracted text",
                "required_inputs": ["data/transformed_data/extracted_text"],
                "outputs": ["data/transformed_data/semantic_chunks"],
                "optional": False
            },
            {
                "name": "Metadata Enrichment",
                "script": "step08_enrich_metadata_with_crossref_api.py",
                "description": "Enrich metadata using Crossref API",
                "required_inputs": ["data/transformed_data/metadata_extraction"],
                "outputs": ["data/transformed_data/metadata_enrichment"],
                "optional": False
            },
            {
                "name": "Vector Embeddings",
                "script": "step09_generate_vector_embeddings_for_chunks.py",
                "description": "Generate vector embeddings and FAISS index",
                "required_inputs": ["data/transformed_data/semantic_chunks"],
                "outputs": ["data/transformed_data/vector_embeddings"],
                "optional": False
            }
        ]
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create logs directory if it doesn't exist
        logs_dir = self.pipeline_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"pipeline_run_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🚀 Starting Scientific Publications Pipeline")
        self.logger.info(f"📝 Log file: {log_file}")
        self.logger.info(f"🔧 Dry run mode: {self.dry_run}")
        if self.max_files:
            self.logger.info(f"📊 Max files to process: {self.max_files}")
    
    def check_prerequisites(self) -> bool:
        """Check if all required directories and tools exist."""
        self.logger.info("🔍 Checking pipeline prerequisites...")
        
        # Check if tools directory exists
        if not self.tools_dir.exists():
            self.logger.error(f"❌ Tools directory not found: {self.tools_dir}")
            return False
        
        # Check if all pipeline scripts exist
        missing_scripts = []
        for step in self.pipeline_steps:
            script_path = self.tools_dir / step["script"]
            if not script_path.exists():
                missing_scripts.append(step["script"])
        
        if missing_scripts:
            self.logger.error(f"❌ Missing pipeline scripts: {missing_scripts}")
            return False
        
        # Check if data directory exists
        if not self.data_dir.exists():
            self.logger.info(f"📁 Creating data directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if source archive has PDFs
        archive_dir = self.data_dir / "source_data" / "archive"
        if archive_dir.exists():
            pdf_count = len(list(archive_dir.glob("*.pdf")))
            self.logger.info(f"📚 Found {pdf_count} PDFs in archive directory")
            if pdf_count == 0:
                self.logger.warning("⚠️  No PDFs found in archive directory")
        else:
            self.logger.warning("⚠️  Archive directory not found - will be created during pipeline")
        
        self.logger.info("✅ Prerequisites check complete")
        return True
    
    def create_directory_structure(self):
        """Create the required directory structure."""
        self.logger.info("📁 Creating pipeline directory structure...")
        
        directories = [
            "data/source_data/raw",
            "data/source_data/raw_pdf",  # Add raw_pdf directory
            "data/source_data/archive", 
            "data/source_data/preprocessed/sanitized/pdfs",
            "data/source_data/DLQ",
            "data/transformed_data/quick_metadata",
            "data/transformed_data/extracted_text",
            "data/transformed_data/metadata_extraction",
            "data/transformed_data/semantic_chunks",
            "data/transformed_data/metadata_enrichment",
            "data/transformed_data/vector_embeddings"
        ]
        
        for dir_path in directories:
            full_path = self.tools_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"   Created: {full_path}")
        
        self.logger.info("✅ Directory structure created")
    
    def prepare_pipeline_inputs(self):
        """Prepare pipeline inputs by copying a PDF from archive if needed."""
        self.logger.info("🔧 Preparing pipeline inputs...")
        
        # Check if we have PDFs in raw_pdf directory
        raw_pdf_dir = self.tools_dir / "data/source_data/raw_pdf"
        if raw_pdf_dir.exists() and any(raw_pdf_dir.glob("*.pdf")):
            self.logger.info("✅ PDFs already exist in raw_pdf directory")
            return True
        
        # Check if we have PDFs in the actual archive (in data directory)
        actual_archive_dir = self.data_dir / "source_data/archive"
        if not actual_archive_dir.exists():
            self.logger.warning("⚠️  Actual archive directory not found")
            return False
        
        pdf_files = list(actual_archive_dir.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning("⚠️  No PDFs found in actual archive directory")
            return False
        
        # Copy first PDF to raw_pdf directory for processing
        source_pdf = pdf_files[0]
        target_pdf = raw_pdf_dir / source_pdf.name
        
        if not self.dry_run:
            import shutil
            shutil.copy2(source_pdf, target_pdf)
            self.logger.info(f"📄 Copied {source_pdf.name} from actual archive to tools/raw_pdf for processing")
        else:
            self.logger.info(f"🔍 DRY RUN - Would copy {source_pdf.name} from actual archive to tools/raw_pdf")
        
        return True
    
    def check_step_inputs(self, step: Dict) -> bool:
        """Check if required inputs exist for a step."""
        for input_path in step["required_inputs"]:
            full_path = self.tools_dir / input_path
            if not full_path.exists():
                self.logger.warning(f"⚠️  Input directory not found: {input_path}")
                if not step.get("optional", False):
                    return False
        return True
    
    def run_pipeline_step(self, step: Dict, step_number: int) -> bool:
        """Run a single pipeline step."""
        step_name = step["name"]
        script_name = step["script"]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🔄 Step {step_number}: {step_name}")
        self.logger.info(f"📝 {step['description']}")
        self.logger.info(f"📁 Script: {script_name}")
        self.logger.info(f"{'='*60}")
        
        # Check inputs
        if not self.check_step_inputs(step):
            if step.get("optional", False):
                self.logger.info(f"⏭️  Skipping optional step: {step_name}")
                return True
            else:
                self.logger.error(f"❌ Required inputs missing for step: {step_name}")
                return False
        
        # Check if step should be skipped (no files to process)
        if self.should_skip_step(step):
            self.logger.info(f"⏭️  No files to process - skipping step: {step_name}")
            return True
        
        # Build command
        script_path = self.tools_dir / script_name
        cmd = [sys.executable, str(script_path)]
        
        # Add max-files argument if specified and tool supports it
        if self.max_files and "max-files" in step.get("script", ""):
            cmd.extend(["--max-files", str(self.max_files)])
        
        # Add dry-run flag if requested and tool supports it
        if self.dry_run and "dry-run" in step.get("script", ""):
            cmd.append("--dry-run")
        
        self.logger.info(f"🚀 Running command: {' '.join(cmd)}")
        
        if self.dry_run:
            self.logger.info("🔍 DRY RUN - Command would be executed")
            return True
        
        # Execute the step
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.tools_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            execution_time = time.time() - start_time
            
            # Log output
            if result.stdout:
                self.logger.info(f"📤 Step output:\n{result.stdout}")
            if result.stderr:
                self.logger.warning(f"⚠️  Step warnings:\n{result.stderr}")
            
            # Check result
            if result.returncode == 0:
                self.logger.info(f"✅ Step {step_number} completed successfully in {execution_time:.2f}s")
                self.step_results[step_number] = {
                    "name": step_name,
                    "success": True,
                    "execution_time": execution_time,
                    "return_code": result.returncode
                }
                return True
            else:
                self.logger.error(f"❌ Step {step_number} failed with return code {result.returncode}")
                self.logger.error(f"Error output:\n{result.stderr}")
                self.step_results[step_number] = {
                    "name": step_name,
                    "success": False,
                    "execution_time": execution_time,
                    "return_code": result.returncode,
                    "error": result.stderr
                }
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ Step {step_number} timed out after 1 hour")
            self.step_results[step_number] = {
                "name": step_name,
                "success": False,
                "execution_time": 3600,
                "return_code": -1,
                "error": "Timeout expired"
            }
            return False
        except Exception as e:
            self.logger.error(f"❌ Step {step_number} failed with exception: {e}")
            self.step_results[step_number] = {
                "name": step_name,
                "success": False,
                "execution_time": 0,
                "return_code": -1,
                "error": str(e)
            }
            return False
    
    def should_skip_step(self, step: Dict) -> bool:
        """Check if a step should be skipped due to no files to process."""
        # For file sorting step, check if raw directory has files
        if step["script"] == "step01_sort_and_archive_incoming_files.py":
            raw_dir = self.tools_dir / "data/source_data/raw"
            if raw_dir.exists():
                files = list(raw_dir.glob("*"))
                return len([f for f in files if f.is_file()]) == 0
            return True
        
        # For other steps, check if input directories have files
        for input_path in step["required_inputs"]:
            full_path = self.tools_dir / input_path
            if full_path.exists():
                if "pdfs" in input_path:
                    files = list(full_path.glob("*.pdf"))
                else:
                    files = list(full_path.glob("*"))
                if len([f for f in files if f.is_file()]) > 0:
                    return False
        
        return True
    
    def run_pipeline(self, start_from_step: int = 1) -> bool:
        """Run the complete pipeline."""
        self.logger.info(f"🚀 Starting pipeline from step {start_from_step}")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Prepare pipeline inputs (copy PDF from archive if needed)
        if not self.prepare_pipeline_inputs():
            self.logger.warning("⚠️  No PDFs found in archive or raw_pdf. Cannot proceed with pipeline.")
            return False
        
        # Run each step
        for i, step in enumerate(self.pipeline_steps, 1):
            if i < start_from_step:
                self.logger.info(f"⏭️  Skipping step {i} (starting from step {start_from_step})")
                continue
            
            success = self.run_pipeline_step(step, i)
            if not success:
                self.logger.error(f"❌ Pipeline failed at step {i}: {step['name']}")
                return False
        
        self.logger.info("🎉 Pipeline completed successfully!")
        return True
    
    def generate_summary(self):
        """Generate a summary of the pipeline run."""
        total_time = time.time() - self.start_time
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📊 PIPELINE RUN SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"⏱️  Total execution time: {total_time:.2f}s")
        self.logger.info(f"📋 Steps executed: {len(self.step_results)}")
        
        successful_steps = sum(1 for result in self.step_results.values() if result["success"])
        failed_steps = len(self.step_results) - successful_steps
        
        self.logger.info(f"✅ Successful steps: {successful_steps}")
        self.logger.info(f"❌ Failed steps: {failed_steps}")
        
        if failed_steps > 0:
            self.logger.info("\n❌ Failed steps details:")
            for step_num, result in self.step_results.items():
                if not result["success"]:
                    self.logger.info(f"   Step {step_num}: {result['name']} - {result.get('error', 'Unknown error')}")
        
        # Save results to JSON file
        summary_file = self.pipeline_root / "logs" / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "max_files": self.max_files,
            "dry_run": self.dry_run,
            "verbose": self.verbose,
            "step_results": self.step_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        self.logger.info(f"📄 Summary saved to: {summary_file}")
        
        return failed_steps == 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run the complete Scientific Publications Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py
  
  # Test with only 5 files
  python run_pipeline.py --max-files 5
  
  # Dry run to see what would happen
  python run_pipeline.py --dry-run
  
  # Start from step 3 (skip file sorting and sanitization)
  python run_pipeline.py --start-from-step 3
  
  # Verbose logging
  python run_pipeline.py --verbose
        """
    )
    
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process (useful for testing)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--start-from-step",
        type=int,
        default=1,
        help="Start pipeline from this step number (default: 1)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            max_files=args.max_files,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        # Check prerequisites
        if not orchestrator.check_prerequisites():
            sys.exit(1)
        
        # Run pipeline
        success = orchestrator.run_pipeline(start_from_step=args.start_from_step)
        
        # Generate summary
        orchestrator.generate_summary()
        
        # Exit with appropriate code
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 