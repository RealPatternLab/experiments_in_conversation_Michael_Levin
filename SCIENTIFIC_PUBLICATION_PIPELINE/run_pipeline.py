#!/usr/bin/env python3
"""
Pipeline Runner - Automated Execution of All Pipeline Steps

This script runs the complete Michael Levin Scientific Publications Pipeline
from start to finish, executing all 7 steps in sequence with proper error
handling and progress tracking.

Usage:
    uv run python run_pipeline.py                    # Run all steps
    uv run python run_pipeline.py --step 3           # Run from step 3 onwards
    uv run python run_pipeline.py --force            # Force run even if no files
    uv run python run_pipeline.py --status          # Show pipeline status only
    uv run python run_pipeline.py --clean           # Clean failed entries and restart
    uv run python run_pipeline.py --no-embedding    # Skip the embedding step (step 6)
"""

import argparse
import logging
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_runner.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Automated runner for the complete scientific publications pipeline."""
    
    def __init__(self):
        """Initialize the pipeline runner."""
        self.steps = [
            {
                'name': 'Hash Validation',
                'script': 'step_01_unique_hashcode_validator.py',
                'description': 'Validate file uniqueness and detect duplicates'
            },
            {
                'name': 'Metadata Extraction',
                'script': 'step_02_metadata_extractor.py',
                'description': 'Extract publication metadata from PDFs'
            },
            {
                'name': 'Text Extraction',
                'script': 'step_03_text_extractor.py',
                'description': 'Extract text content from PDFs'
            },
            {
                'name': 'Metadata Enrichment',
                'script': 'step_04_optional_metadata_enrichment.py',
                'description': 'Enhance metadata with Crossref API'
            },
            {
                'name': 'Semantic Chunking',
                'script': 'step_05_semantic_chunker_split.py',
                'description': 'Create semantic text chunks'
            },
            {
                'name': 'Consolidated Embedding',
                'script': 'step_06_consolidated_embedding.py',
                'description': 'Generate FAISS vector embeddings',
                'skip_flag': '--no-embedding'
            },
            {
                'name': 'Archive Management',
                'script': 'step_07_move_to_archive.py',
                'description': 'Organize processed files'
            }
        ]
        
        self.current_step = 0
        self.results = {}
        self.start_time = None
        
    def show_pipeline_status(self):
        """Display current pipeline status."""
        logger.info("🔍 Pipeline Status Check")
        logger.info("=" * 50)
        
        try:
            with open('pipeline_progress_queue.json', 'r') as f:
                data = json.load(f)
            
            pipeline_progress = data.get('pipeline_progress', {})
            
            if not pipeline_progress:
                logger.info("📭 No documents in pipeline queue")
                return
            
            # Count documents by status
            status_counts = {}
            for doc_id, doc_data in pipeline_progress.items():
                current_status = doc_data.get('current_status', 'unknown')
                status_counts[current_status] = status_counts.get(current_status, 0) + 1
            
            logger.info(f"📊 Total Documents: {len(pipeline_progress)}")
            for status, count in status_counts.items():
                logger.info(f"   {status.capitalize()}: {count}")
            
            # Show recent documents
            logger.info("\n📋 Recent Documents:")
            for i, (doc_id, doc_data) in enumerate(list(pipeline_progress.items())[-5:]):
                filename = doc_data.get('filename', 'Unknown')
                status = doc_data.get('current_status', 'unknown')
                logger.info(f"   {filename}: {status}")
                
        except FileNotFoundError:
            logger.warning("❌ Pipeline progress queue not found")
        except Exception as e:
            logger.error(f"❌ Error reading pipeline status: {e}")
    
    def check_prerequisites(self) -> bool:
        """Check if pipeline can run."""
        logger.info("🔍 Checking pipeline prerequisites...")
        
        # Check if step_01_raw has files
        raw_dir = Path("step_01_raw")
        if not raw_dir.exists():
            logger.error("❌ step_01_raw directory not found")
            return False
        
        pdf_files = list(raw_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("⚠️  No PDF files found in step_01_raw/")
            logger.info("💡 Add PDF files to step_01_raw/ to start processing")
            return False
        
        logger.info(f"✅ Found {len(pdf_files)} PDF files to process")
        return True
    
    def run_step(self, step_info: Dict[str, str], force_run: bool = False, skip_embedding: bool = False) -> bool:
        """Run a single pipeline step."""
        step_name = step_info['name']
        script = step_info['script']
        description = step_info['description']
        
        # Check if this step should be skipped
        if skip_embedding and step_name == 'Consolidated Embedding':
            logger.info(f"⏭️  Skipping Step {self.current_step + 1}: {step_name}")
            logger.info(f"📝 {description}")
            logger.info("💡 Skipped due to --no-embedding flag")
            logger.info("-" * 60)
            
            self.results[step_name] = {
                'status': 'skipped',
                'reason': '--no-embedding flag specified'
            }
            return True
        
        logger.info(f"🚀 Starting Step {self.current_step + 1}: {step_name}")
        logger.info(f"📝 {description}")
        logger.info(f"📜 Script: {script}")
        logger.info("-" * 60)
        
        try:
            # Run the step script
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Step {self.current_step + 1} ({step_name}) completed successfully")
                self.results[step_name] = {
                    'status': 'success',
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                return True
            else:
                logger.error(f"❌ Step {self.current_step + 1} ({step_name}) failed")
                logger.error(f"Return code: {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr}")
                
                self.results[step_name] = {
                    'status': 'failed',
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running step {step_name}: {e}")
            self.results[step_name] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def clean_failed_entries(self):
        """Clean failed entries from the pipeline queue."""
        logger.info("🧹 Cleaning failed pipeline entries...")
        
        try:
            with open('pipeline_progress_queue.json', 'r') as f:
                data = json.load(f)
            
            pipeline_progress = data.get('pipeline_progress', {})
            
            # Remove failed entries
            failed_removed = 0
            cleaned_progress = {}
            
            for doc_id, doc_data in pipeline_progress.items():
                if doc_data.get('current_status') != 'failed':
                    cleaned_progress[doc_id] = doc_data
                else:
                    failed_removed += 1
            
            # Update the queue
            data['pipeline_progress'] = cleaned_progress
            data['last_updated'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            
            with open('pipeline_progress_queue.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Cleaned {failed_removed} failed entries from pipeline queue")
            logger.info(f"📊 Remaining documents: {len(cleaned_progress)}")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning failed entries: {e}")
    
    def run_pipeline(self, start_step: int = 0, force_run: bool = False, skip_embedding: bool = False) -> bool:
        """Run the complete pipeline from the specified step."""
        logger.info("🚀 Starting Michael Levin Scientific Publications Pipeline")
        if skip_embedding:
            logger.info("⏭️  Embedding step will be skipped (--no-embedding flag)")
        logger.info("=" * 70)
        
        self.start_time = time.time()
        self.current_step = start_step
        
        # Check prerequisites
        if not self.check_prerequisites():
            if not force_run:
                logger.error("❌ Prerequisites not met. Use --force to run anyway.")
                return False
            else:
                logger.warning("⚠️  Prerequisites not met, but continuing due to --force flag")
        
        # Run each step
        for i, step_info in enumerate(self.steps[start_step:], start_step + 1):
            self.current_step = i - 1
            
            logger.info(f"\n📋 Pipeline Progress: {i}/{len(self.steps)}")
            logger.info(f"⏱️  Elapsed time: {time.time() - self.start_time:.1f}s")
            
            success = self.run_step(step_info, force_run, skip_embedding)
            
            if not success:
                logger.error(f"❌ Pipeline failed at step {i}: {step_info['name']}")
                logger.error("🛑 Stopping pipeline execution")
                return False
            
            # Brief pause between steps
            if i < len(self.steps):
                logger.info("⏳ Waiting 2 seconds before next step...")
                time.sleep(2)
        
        # Pipeline completed successfully
        total_time = time.time() - self.start_time
        logger.info("\n🎉 Pipeline completed successfully!")
        logger.info(f"⏱️  Total execution time: {total_time:.1f}s")
        if skip_embedding:
            logger.info("⏭️  Note: Embedding step was skipped")
        logger.info("=" * 70)
        
        return True
    
    def show_summary(self):
        """Show pipeline execution summary."""
        if not self.results:
            logger.info("📊 No pipeline execution results to show")
            return
        
        logger.info("\n📊 Pipeline Execution Summary")
        logger.info("=" * 50)
        
        for step_name, result in self.results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                logger.info(f"✅ {step_name}: Success")
            elif status == 'skipped':
                logger.info(f"⏭️  {step_name}: Skipped ({result.get('reason', 'Unknown reason')})")
            elif status == 'failed':
                logger.info(f"❌ {step_name}: Failed (return code: {result.get('return_code', 'N/A')})")
            else:
                logger.info(f"⚠️  {step_name}: Error ({result.get('error', 'Unknown error')})")
        
        # Count successes, failures, and skips
        successes = sum(1 for r in self.results.values() if r.get('status') == 'success')
        skips = sum(1 for r in self.results.values() if r.get('status') == 'skipped')
        failures = len(self.results) - successes - skips
        
        logger.info(f"\n📈 Results: {successes} successful, {skips} skipped, {failures} failed")
        
        if failures == 0:
            if skips > 0:
                logger.info("🎉 Pipeline completed successfully (with skipped steps)!")
            else:
                logger.info("🎉 All steps completed successfully!")
        else:
            logger.warning(f"⚠️  {failures} step(s) had issues")

def main():
    """Main entry point for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the complete Michael Levin Scientific Publications Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python run_pipeline.py                    # Run all steps
  uv run python run_pipeline.py --step 3           # Start from step 3
  uv run python run_pipeline.py --force            # Force run
  uv run python run_pipeline.py --status          # Show status only
  uv run python run_pipeline.py --clean           # Clean failed entries
  uv run python run_pipeline.py --no-embedding    # Skip embedding step
        """
    )
    
    parser.add_argument(
        '--step', 
        type=int, 
        default=1,
        help='Start pipeline from this step (1-7)'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force run even if prerequisites not met'
    )
    
    parser.add_argument(
        '--status', 
        action='store_true',
        help='Show pipeline status only (do not run)'
    )
    
    parser.add_argument(
        '--clean', 
        action='store_true',
        help='Clean failed entries from pipeline queue'
    )
    
    parser.add_argument(
        '--no-embedding', 
        action='store_true',
        help='Skip the embedding step (step 6) to save time'
    )
    
    args = parser.parse_args()
    
    # Validate step number
    if args.step < 1 or args.step > 7:
        logger.error("❌ Step number must be between 1 and 7")
        sys.exit(1)
    
    # Initialize pipeline runner
    runner = PipelineRunner()
    
    try:
        if args.status:
            # Show status only
            runner.show_pipeline_status()
        elif args.clean:
            # Clean failed entries
            runner.clean_failed_entries()
        else:
            # Run pipeline
            success = runner.run_pipeline(args.step - 1, args.force, args.no_embedding)
            runner.show_summary()
            
            if not success:
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("\n⚠️  Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
