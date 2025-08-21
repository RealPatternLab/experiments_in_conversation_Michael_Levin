"""
Centralized logging configuration for the conversations video pipeline.
All logs are funneled to the logs/ directory for better organization.
"""

import logging
import os
from pathlib import Path
from datetime import datetime

# Get the pipeline root directory
PIPELINE_ROOT = Path(__file__).parent
LOGS_DIR = PIPELINE_ROOT / "logs"
DOCS_DIR = PIPELINE_ROOT / "docs"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

def setup_logging(step_name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for a specific pipeline step.
    
    Args:
        step_name: Name of the pipeline step (e.g., 'playlist_processing', 'video_download')
        log_level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(step_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{step_name}_{timestamp}.log"
    log_path = LOGS_DIR / log_filename
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return LOGS_DIR

def get_docs_dir() -> Path:
    """Get the docs directory path."""
    return DOCS_DIR

def cleanup_old_logs(max_logs_per_step: int = 5):
    """
    Clean up old log files, keeping only the most recent ones per step.
    
    Args:
        max_logs_per_step: Maximum number of log files to keep per step
    """
    try:
        # Group log files by step name
        step_logs = {}
        for log_file in LOGS_DIR.glob("*.log"):
            # Extract step name from filename (e.g., "playlist_processing_20250817_123456.log")
            if "_" in log_file.stem:
                step_name = log_file.stem.rsplit("_", 2)[0]  # Remove timestamp
                if step_name not in step_logs:
                    step_logs[step_name] = []
                step_logs[step_name].append(log_file)
        
        # Keep only the most recent logs per step
        for step_name, log_files in step_logs.items():
            if len(log_files) > max_logs_per_step:
                # Sort by modification time (newest first)
                log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Remove old log files
                for old_log in log_files[max_logs_per_step:]:
                    try:
                        old_log.unlink()
                        print(f"Removed old log: {old_log.name}")
                    except Exception as e:
                        print(f"Failed to remove old log {old_log.name}: {e}")
                        
    except Exception as e:
        print(f"Error during log cleanup: {e}")

# Clean up old logs when module is imported
cleanup_old_logs()
