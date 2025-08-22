#!/usr/bin/env python3
"""
Centralized logging configuration for the 1-on-2 Conversations Video Pipeline.
Provides consistent logging across all pipeline steps with proper formatting and file output.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Default logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FORMAT_VERBOSE = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'

# Log file configuration
LOG_DIR_NAME = "logs"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

def get_logs_dir() -> Path:
    """Get the logs directory path, creating it if it doesn't exist"""
    logs_dir = Path.cwd() / LOG_DIR_NAME
    logs_dir.mkdir(exist_ok=True)
    return logs_dir

def setup_logging(
    logger_name: str,
    log_level: Optional[int] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    verbose: bool = False
) -> logging.Logger:
    """
    Set up logging for a specific component of the pipeline.
    
    Args:
        logger_name: Name of the logger (e.g., 'pipeline_execution', 'transcription')
        log_level: Logging level (defaults to DEFAULT_LOG_LEVEL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        verbose: Whether to use verbose formatting
    
    Returns:
        Configured logger instance
    """
    if log_level is None:
        log_level = DEFAULT_LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set format
    if verbose:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT_VERBOSE)
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        logs_dir = get_logs_dir()
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = logs_dir / f"{logger_name}_{timestamp}.log"
        
        # Use rotating file handler to manage log file size
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def setup_pipeline_logging(
    pipeline_name: str = "conversations_1_on_2",
    log_level: Optional[int] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Set up logging specifically for pipeline execution.
    
    Args:
        pipeline_name: Name of the pipeline
        log_level: Logging level
        verbose: Whether to use verbose formatting
    
    Returns:
        Configured pipeline logger
    """
    return setup_logging(
        logger_name=f"pipeline_{pipeline_name}",
        log_level=log_level,
        log_to_file=True,
        log_to_console=True,
        verbose=verbose
    )

def setup_step_logging(
    step_name: str,
    log_level: Optional[int] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Set up logging for a specific pipeline step.
    
    Args:
        step_name: Name of the pipeline step (e.g., 'transcription', 'chunking')
        log_level: Logging level
        verbose: Whether to use verbose formatting
    
    Returns:
        Configured step logger
    """
    return setup_logging(
        logger_name=f"step_{step_name}",
        log_level=log_level,
        log_to_file=True,
        log_to_console=True,
        verbose=verbose
    )

def get_log_file_path(logger_name: str) -> Optional[Path]:
    """
    Get the current log file path for a logger.
    
    Args:
        logger_name: Name of the logger
    
    Returns:
        Path to the current log file, or None if not found
    """
    logger = logging.getLogger(logger_name)
    
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            return Path(handler.baseFilename)
    
    return None

def rotate_logs():
    """Manually rotate all log files"""
    logs_dir = get_logs_dir()
    
    for log_file in logs_dir.glob("*.log"):
        try:
            # Check if file needs rotation
            if log_file.stat().st_size > LOG_FILE_MAX_BYTES:
                # Create backup filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{log_file.stem}_{timestamp}{log_file.suffix}"
                backup_path = log_file.parent / backup_name
                
                # Rename current file
                log_file.rename(backup_path)
                print(f"Rotated log file: {log_file.name} -> {backup_name}")
        except Exception as e:
            print(f"Error rotating log file {log_file}: {e}")

def cleanup_old_logs(max_days: int = 30):
    """
    Clean up log files older than specified days.
    
    Args:
        max_days: Maximum age of log files in days
    """
    logs_dir = get_logs_dir()
    cutoff_time = datetime.now().timestamp() - (max_days * 24 * 60 * 60)
    
    for log_file in logs_dir.glob("*.log"):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                print(f"Removed old log file: {log_file.name}")
        except Exception as e:
            print(f"Error removing old log file {log_file}: {e}")

def log_pipeline_start(pipeline_name: str, step_count: int):
    """
    Log the start of a pipeline execution.
    
    Args:
        pipeline_name: Name of the pipeline
        step_count: Number of steps in the pipeline
    """
    logger = setup_pipeline_logging(pipeline_name)
    logger.info("🚀 Starting Pipeline Execution")
    logger.info(f"Pipeline: {pipeline_name}")
    logger.info(f"Total Steps: {step_count}")
    logger.info(f"Start Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

def log_pipeline_step(step_name: str, step_number: int, total_steps: int, status: str = "started"):
    """
    Log pipeline step execution.
    
    Args:
        step_name: Name of the step
        step_number: Current step number
        total_steps: Total number of steps
        status: Step status (started, completed, failed)
    """
    logger = setup_pipeline_logging()
    
    if status == "started":
        logger.info(f"📋 Step {step_number}/{total_steps}: {step_name}")
        logger.info("=" * 40)
    elif status == "completed":
        logger.info(f"✅ Step {step_number}/{total_steps}: {step_name} completed")
    elif status == "failed":
        logger.error(f"❌ Step {step_number}/{total_steps}: {step_name} failed")

def log_pipeline_completion(pipeline_name: str, total_duration: float, success: bool):
    """
    Log pipeline completion.
    
    Args:
        pipeline_name: Name of the pipeline
        total_duration: Total execution time in seconds
        success: Whether the pipeline completed successfully
    """
    logger = setup_pipeline_logging(pipeline_name)
    
    if success:
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"Total execution time: {total_duration:.2f} seconds")
    else:
        logger.error("❌ Pipeline execution failed")
    
    logger.info(f"Completion Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

# Example usage and testing
if __name__ == "__main__":
    # Test logging setup
    print("Testing logging configuration...")
    
    # Test pipeline logging
    pipeline_logger = setup_pipeline_logging("test_pipeline")
    pipeline_logger.info("Pipeline logging test successful")
    
    # Test step logging
    step_logger = setup_step_logging("test_step")
    step_logger.info("Step logging test successful")
    
    # Test file logging
    file_logger = setup_logging("test_file", log_to_console=False, log_to_file=True)
    file_logger.info("File logging test successful")
    
    print("Logging configuration test completed!")
    print(f"Logs directory: {get_logs_dir()}")
    
    # Show log file paths
    for logger_name in ["test_pipeline", "test_step", "test_file"]:
        log_path = get_log_file_path(logger_name)
        if log_path:
            print(f"Log file for {logger_name}: {log_path}")
