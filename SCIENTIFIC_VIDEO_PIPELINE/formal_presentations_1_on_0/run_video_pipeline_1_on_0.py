#!/usr/bin/env python3
"""
Consolidated Video Pipeline 1-on-0
Runs all 7 steps of the scientific video processing pipeline in sequence.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoPipeline1On0:
    """Consolidated pipeline runner for all 7 steps"""
    
    def __init__(self):
        self.steps = [
            {
                'name': 'Step 1: Playlist Processing',
                'script': 'step_01_playlist_processor.py',
                'description': 'Process YouTube playlist URLs and extract metadata'
            },
            {
                'name': 'Step 2: Video Download',
                'script': 'step_02_video_downloader.py',
                'description': 'Download videos using yt-dlp with metadata extraction'
            },
            {
                'name': 'Step 3: Transcription',
                'script': 'step_03_transcription_webhook.py',
                'description': 'Create high-quality transcripts using AssemblyAI webhooks (better for long videos)'
            },
            {
                'name': 'Step 4: Semantic Chunking',
                'script': 'step_04_extract_chunks.py',
                'description': 'Create semantic chunks with LLM enhancement'
            },
            {
                'name': 'Step 5: Frame Extraction',
                'script': 'step_05_frame_extractor.py',
                'description': 'Extract video frames at regular intervals'
            },
            {
                'name': 'Step 6: Frame-Chunk Alignment',
                'script': 'step_06_frame_chunk_alignment.py',
                'description': 'Align extracted frames with transcript chunks'
            },
            {
                'name': 'Step 7: Consolidated Embedding',
                'script': 'step_07_consolidated_embedding.py',
                'description': 'Create embeddings and build FAISS indices'
            }
        ]
        
        self.execution_results = []
        self.start_time = None
        self.end_time = None
    
    def run_step(self, step_info: Dict[str, str]) -> Dict[str, Any]:
        """Run a single pipeline step"""
        step_name = step_info['name']
        script_name = step_info['script']
        description = step_info['description']
        
        logger.info(f"🚀 Starting {step_name}")
        logger.info(f"   Description: {description}")
        logger.info(f"   Script: {script_name}")
        
        step_start = time.time()
        result = {
            'step_name': step_name,
            'script_name': script_name,
            'description': description,
            'start_time': datetime.now().isoformat(),
            'status': 'unknown',
            'error': None,
            'duration_seconds': 0
        }
        
        try:
            # Check if script exists
            if not Path(script_name).exists():
                raise FileNotFoundError(f"Script {script_name} not found")
            
            # Run the script using uv
            import subprocess
            cmd = ['uv', 'run', script_name]
            
            logger.info(f"   Executing: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Success
            result['status'] = 'completed'
            result['stdout'] = process.stdout
            result['stderr'] = process.stderr
            logger.info(f"✅ {step_name} completed successfully")
            
        except subprocess.CalledProcessError as e:
            # Script execution failed
            result['status'] = 'failed'
            result['error'] = f"Script execution failed with exit code {e.returncode}"
            result['stdout'] = e.stdout
            result['stderr'] = e.stderr
            logger.error(f"❌ {step_name} failed: {result['error']}")
            logger.error(f"   STDOUT: {e.stdout}")
            logger.error(f"   STDERR: {e.stderr}")
            
        except FileNotFoundError as e:
            # Script file not found
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"❌ {step_name} failed: {result['error']}")
            
        except Exception as e:
            # Unexpected error
            result['status'] = 'failed'
            result['error'] = f"Unexpected error: {str(e)}"
            logger.error(f"❌ {step_name} failed with unexpected error: {str(e)}")
        
        finally:
            # Calculate duration
            step_end = time.time()
            result['duration_seconds'] = step_end - step_start
            result['end_time'] = datetime.now().isoformat()
            
            # Log duration
            duration_str = f"{result['duration_seconds']:.1f}s"
            logger.info(f"   Duration: {duration_str}")
            logger.info("")
        
        return result
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline"""
        logger.info("🎬 Starting Video Pipeline 1-on-0")
        logger.info("=" * 60)
        logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total steps: {len(self.steps)}")
        logger.info("=" * 60)
        logger.info("")
        
        self.start_time = time.time()
        
        # Run each step
        for i, step_info in enumerate(self.steps, 1):
            logger.info(f"📋 Step {i}/{len(self.steps)}")
            result = self.run_step(step_info)
            self.execution_results.append(result)
            
            # Check if step failed
            if result['status'] == 'failed':
                logger.error(f"🚨 Pipeline failed at {step_info['name']}")
                logger.error("Stopping pipeline execution")
                return False
        
        # All steps completed successfully
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        return True
    
    def generate_report(self) -> str:
        """Generate a detailed execution report"""
        if not self.execution_results:
            return "No execution results available"
        
        report_lines = []
        report_lines.append("📊 PIPELINE EXECUTION REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        total_steps = len(self.execution_results)
        completed_steps = sum(1 for r in self.execution_results if r['status'] == 'completed')
        failed_steps = sum(1 for r in self.execution_results if r['status'] == 'failed')
        
        report_lines.append("📈 EXECUTION SUMMARY")
        report_lines.append(f"Total Steps: {total_steps}")
        report_lines.append(f"Completed: {completed_steps}")
        report_lines.append(f"Failed: {failed_steps}")
        report_lines.append(f"Success Rate: {(completed_steps/total_steps)*100:.1f}%")
        report_lines.append("")
        
        # Step details
        report_lines.append("🔍 STEP DETAILS")
        for i, result in enumerate(self.execution_results, 1):
            status_icon = "✅" if result['status'] == 'completed' else "❌"
            duration = f"{result['duration_seconds']:.1f}s"
            
            report_lines.append(f"{i}. {status_icon} {result['step_name']}")
            report_lines.append(f"   Duration: {duration}")
            report_lines.append(f"   Status: {result['status']}")
            
            if result['error']:
                report_lines.append(f"   Error: {result['error']}")
            report_lines.append("")
        
        # Timing information
        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            report_lines.append("⏱️ TIMING INFORMATION")
            report_lines.append(f"Total Pipeline Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_report(self, filename: str = None):
        """Save the execution report to a file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_report_{timestamp}.txt"
        
        report_content = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Execution report saved to: {filename}")
        return filename

def main():
    """Main entry point"""
    try:
        # Create and run pipeline
        pipeline = VideoPipeline1On0()
        success = pipeline.run_pipeline()
        
        # Generate and save report
        report_file = pipeline.save_report()
        
        # Print report to console
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*60)
        print(pipeline.generate_report())
        print("="*60)
        
        if success:
            print("\n🎉 All steps completed successfully!")
            print("The pipeline is ready for use.")
            sys.exit(0)
        else:
            print("\n🚨 Pipeline execution failed!")
            print("Check the logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
