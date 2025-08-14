#!/usr/bin/env python3
"""
Pipeline Progress Queue Manager

Tracks the progress of documents through each step of the scientific publications pipeline.
Provides thread-safe status updates and querying capabilities.
"""

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PipelineProgressQueue:
    """Manages pipeline progress tracking for documents."""
    
    def __init__(self, queue_file: Path = Path("pipeline_progress_queue.json")):
        self.queue_file = queue_file
        self.lock = threading.Lock()
        
        # Initialize the queue file if it doesn't exist
        self._initialize_queue_file()
    
    def _initialize_queue_file(self):
        """Initialize the progress queue file with default structure."""
        if not self.queue_file.exists():
            initial_data = {
                "pipeline_progress": {},
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            self._save_queue_data(initial_data)
            logger.info(f"Created new pipeline progress queue: {self.queue_file}")
    
    def _load_queue_data(self) -> Dict[str, Any]:
        """Load the current queue data from file."""
        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file is corrupted, reinitialize
            logger.warning("Queue file corrupted, reinitializing...")
            self._initialize_queue_file()
            return self._load_queue_data()
    
    def _save_queue_data(self, data: Dict[str, Any]):
        """Save queue data to file."""
        try:
            # Ensure directory exists
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save queue data: {e}")
            raise
    
    def add_document(self, doc_id: str, filename: str) -> bool:
        """Add a new document to the progress queue."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                # Check if document already exists
                if doc_id in data["pipeline_progress"]:
                    logger.warning(f"Document {doc_id} already exists in progress queue")
                    return False
                
                # Initialize document with default status
                data["pipeline_progress"][doc_id] = {
                    "filename": filename,
                    "step_01_hash_validation": "pending",
                    "step_02_metadata_extraction": "pending",
                    "step_03_text_extraction": "pending",
                    "step_04_metadata_enrichment": "pending",
                    "step_05_semantic_chunking": "pending",
                    "step_06_embedding": "pending",
                    "current_status": "processing",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "chunk_count": 0,
                    "error_count": 0,
                    "errors": []
                }
                
                data["last_updated"] = datetime.now().isoformat()
                self._save_queue_data(data)
                
                logger.info(f"Added document {doc_id} to progress queue")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add document {doc_id}: {e}")
                return False
    
    def update_step_status(self, doc_id: str, step: str, status: str, 
                          additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """Update the status of a specific step for a document."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if doc_id not in data["pipeline_progress"]:
                    logger.error(f"Document {doc_id} not found in progress queue")
                    return False
                
                doc_data = data["pipeline_progress"][doc_id]
                
                # Update step status
                if step in doc_data:
                    doc_data[step] = status
                else:
                    logger.warning(f"Unknown step {step} for document {doc_id}")
                    return False
                
                # Update additional info if provided
                if additional_info:
                    for key, value in additional_info.items():
                        doc_data[key] = value
                
                # Update timestamps and status
                doc_data["last_updated"] = datetime.now().isoformat()
                data["last_updated"] = datetime.now().isoformat()
                
                # Update current status based on step completion
                self._update_current_status(doc_data)
                
                self._save_queue_data(data)
                
                logger.info(f"Updated {doc_id} {step}: {status}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update step status for {doc_id} {step}: {e}")
                return False
    
    def _update_current_status(self, doc_data: Dict[str, Any]):
        """Update the current status based on step completion."""
        steps = [
            "step_01_hash_validation",
            "step_02_metadata_extraction", 
            "step_03_text_extraction",
            "step_04_metadata_enrichment",
            "step_05_semantic_chunking"
        ]
        
        # Check if all steps are complete
        if all(doc_data.get(step) == "complete" for step in steps):
            # Documents that have completed steps 1-5 are always ready for embedding
            # (even if they've been embedded before, they can be re-embedded)
            doc_data["current_status"] = "ready_for_embedding"
        elif any(doc_data.get(step) == "failed" for step in steps):
            doc_data["current_status"] = "failed"
        elif any(doc_data.get(step) == "complete" for step in steps):
            doc_data["current_status"] = "processing"
        else:
            doc_data["current_status"] = "pending"
    
    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a specific document."""
        with self.lock:
            try:
                data = self._load_queue_data()
                return data["pipeline_progress"].get(doc_id)
            except Exception as e:
                logger.error(f"Failed to get document status for {doc_id}: {e}")
                return None
    
    def get_ready_for_embedding(self) -> List[str]:
        """Get list of document IDs that are ready for embedding."""
        with self.lock:
            try:
                data = self._load_queue_data()
                ready_docs = []
                
                for doc_id, doc_data in data["pipeline_progress"].items():
                    # Check if all required steps are complete
                    required_steps = [
                        "step_01_hash_validation",
                        "step_02_metadata_extraction",
                        "step_03_text_extraction",
                        "step_04_metadata_enrichment",
                        "step_05_semantic_chunking"
                    ]
                    
                    # Document is ready for embedding if all required steps are complete
                    # Note: We don't check step_06_embedding status - documents can be re-embedded
                    if all(doc_data.get(step) == "complete" for step in required_steps):
                        ready_docs.append(doc_id)
                
                return ready_docs
                
            except Exception as e:
                logger.error(f"Failed to get ready for embedding documents: {e}")
                return []
    
    def get_embedding_history(self, doc_id: str) -> List[str]:
        """Get list of embedding timestamps for a specific document."""
        with self.lock:
            try:
                data = self._load_queue_data()
                if doc_id in data["pipeline_progress"]:
                    doc_data = data["pipeline_progress"][doc_id]
                    return doc_data.get("embedding_timestamps", [])
                return []
                
            except Exception as e:
                logger.error(f"Failed to get embedding history for {doc_id}: {e}")
                return []
    
    def get_documents_with_embeddings(self) -> List[str]:
        """Get list of document IDs that have been embedded at least once."""
        with self.lock:
            try:
                data = self._load_queue_data()
                embedded_docs = []
                
                for doc_id, doc_data in data["pipeline_progress"].items():
                    if doc_data.get("embedding_timestamps") and len(doc_data["embedding_timestamps"]) > 0:
                        embedded_docs.append(doc_id)
                
                return embedded_docs
                
            except Exception as e:
                logger.error(f"Failed to get documents with embeddings: {e}")
                return []
    
    def move_to_archive(self, doc_id: str) -> bool:
        """Move a fully processed document from processing to archive directory."""
        try:
            data = self._load_queue_data()
            if doc_id not in data["pipeline_progress"]:
                logger.error(f"Document {doc_id} not found in progress queue")
                return False
            
            doc_data = data["pipeline_progress"][doc_id]
            new_filename = doc_data.get("new_filename")
            
            if not new_filename:
                logger.error(f"No filename found for document {doc_id}")
                return False
            
            # Check if document has completed steps 1-5
            required_steps = [
                "step_01_hash_validation",
                "step_02_metadata_extraction",
                "step_03_text_extraction",
                "step_04_metadata_enrichment",
                "step_05_semantic_chunking"
            ]
            
            if not all(doc_data.get(step) == "complete" for step in required_steps):
                logger.warning(f"Document {doc_id} has not completed all required steps yet")
                return False
            
            # Define paths
            from pathlib import Path
            processing_path = Path("processing") / new_filename
            archive_path = Path("step_07_archive") / new_filename
            
            # Check if file exists in processing
            if not processing_path.exists():
                logger.warning(f"File {new_filename} not found in processing directory")
                return False
            
            # Move file to archive
            try:
                import shutil
                shutil.move(str(processing_path), str(archive_path))
                logger.info(f"✅ Moved {new_filename} from processing to archive")
                
                # Update progress queue with archive location
                doc_data["archive_location"] = str(archive_path)
                doc_data["archived_at"] = datetime.now().isoformat()
                self._save_queue_data(data)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to move file to archive: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to move document {doc_id} to archive: {e}")
            return False
    
    def get_ready_for_archive(self) -> List[str]:
        """Get list of document IDs that are ready to be moved to archive."""
        with self.lock:
            try:
                data = self._load_queue_data()
                ready_for_archive = []
                
                for doc_id, doc_data in data["pipeline_progress"].items():
                    # Check if all required steps are complete
                    required_steps = [
                        "step_01_hash_validation",
                        "step_02_metadata_extraction",
                        "step_03_text_extraction",
                        "step_04_metadata_enrichment",
                        "step_05_semantic_chunking"
                    ]
                    
                    # Document is ready for archive if all required steps are complete
                    # and it hasn't been archived yet
                    if (all(doc_data.get(step) == "complete" for step in required_steps) and
                        not doc_data.get("archive_location")):
                        ready_for_archive.append(doc_id)
                
                return ready_for_archive
                
            except Exception as e:
                logger.error(f"Failed to get documents ready for archive: {e}")
                return []
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire pipeline status."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                summary = {
                    "total_documents": len(data["pipeline_progress"]),
                    "ready_for_embedding": 0,
                    "processing": 0,
                    "pending": 0,
                    "failed": 0,
                    "total_embeddings": 0,
                    "documents_with_embeddings": 0,
                    "archived": 0,
                    "step_completion": {
                        "step_01_hash_validation": 0,
                        "step_02_metadata_extraction": 0,
                        "step_03_text_extraction": 0,
                        "step_04_metadata_enrichment": 0,
                        "step_05_semantic_chunking": 0,
                        "step_06_embedding": 0
                    }
                }
                
                for doc_data in data["pipeline_progress"].values():
                    status = doc_data.get("current_status", "unknown")
                    if status in summary:
                        summary[status] += 1
                    
                    # Count step completions
                    for step in summary["step_completion"]:
                        if doc_data.get(step) == "complete":
                            summary["step_completion"][step] += 1
                    
                    # Count embedding information
                    embedding_timestamps = doc_data.get("embedding_timestamps", [])
                    if embedding_timestamps:
                        summary["total_embeddings"] += len(embedding_timestamps)
                        summary["documents_with_embeddings"] += 1
                    
                    # Count archived documents
                    if doc_data.get("archive_location"):
                        summary["archived"] += 1
                
                return summary
                
            except Exception as e:
                logger.error(f"Failed to get pipeline summary: {e}")
                return {}
    
    def mark_embedding_complete(self, doc_id: str, chunk_count: int = 0, additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """Mark embedding as complete for a document."""
        info = {"chunk_count": chunk_count, "current_status": "complete"}
        if additional_info:
            info.update(additional_info)
        
        # Get current document data to update embedding timestamps list
        with self.lock:
            try:
                data = self._load_queue_data()
                if doc_id in data["pipeline_progress"]:
                    doc_data = data["pipeline_progress"][doc_id]
                    
                    # Initialize embedding_timestamps as list if it doesn't exist
                    if "embedding_timestamps" not in doc_data:
                        doc_data["embedding_timestamps"] = []
                    
                    # Add new timestamp to the list
                    if additional_info and "embedding_timestamp" in additional_info:
                        doc_data["embedding_timestamps"].append(additional_info["embedding_timestamp"])
                    
                    # Keep the most recent timestamp for backward compatibility
                    if doc_data["embedding_timestamps"]:
                        doc_data["latest_embedding_timestamp"] = doc_data["embedding_timestamps"][-1]
                    
                    self._save_queue_data(data)
                    
            except Exception as e:
                logger.error(f"Failed to update embedding timestamps for {doc_id}: {e}")
        
        return self.update_step_status(
            doc_id, 
            "step_06_embedding", 
            "complete",
            info
        )
    
    def add_error(self, doc_id: str, step: str, error_message: str) -> bool:
        """Add an error for a document and step."""
        with self.lock:
            try:
                data = self._load_queue_data()
                
                if doc_id not in data["pipeline_progress"]:
                    logger.error(f"Document {doc_id} not found in progress queue")
                    return False
                
                doc_data = data["pipeline_progress"][doc_id]
                
                # Add error to document
                if "errors" not in doc_data:
                    doc_data["errors"] = []
                
                error_info = {
                    "step": step,
                    "error": error_message,
                    "timestamp": datetime.now().isoformat()
                }
                doc_data["errors"].append(error_info)
                
                # Update error count
                doc_data["error_count"] = len(doc_data["errors"])
                
                # Mark step as failed
                doc_data[step] = "failed"
                
                # Update current status
                self._update_current_status(doc_data)
                
                data["last_updated"] = datetime.now().isoformat()
                self._save_queue_data(data)
                
                logger.error(f"Added error for {doc_id} {step}: {error_message}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add error for {doc_id} {step}: {e}")
                return False
    
    def clear_queue(self) -> bool:
        """Clear the entire progress queue (useful for testing)."""
        with self.lock:
            try:
                initial_data = {
                    "pipeline_progress": {},
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
                self._save_queue_data(initial_data)
                logger.info("Cleared pipeline progress queue")
                return True
                
            except Exception as e:
                logger.error(f"Failed to clear queue: {e}")
                return False
    
    def get_document_progress(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed progress information for a document."""
        doc_status = self.get_document_status(doc_id)
        if not doc_status:
            return None
        
        # Calculate completion percentage
        steps = [
            "step_01_hash_validation",
            "step_02_metadata_extraction", 
            "step_03_text_extraction",
            "step_04_metadata_enrichment",
            "step_05_semantic_chunking"
        ]
        
        completed_steps = sum(1 for step in steps if doc_status.get(step) == "complete")
        total_steps = len(steps)
        completion_percentage = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
        
        progress_info = doc_status.copy()
        progress_info["completion_percentage"] = completion_percentage
        progress_info["completed_steps"] = completed_steps
        progress_info["total_steps"] = total_steps
        
        return progress_info

def main():
    """Test the progress queue functionality."""
    queue = PipelineProgressQueue()
    
    # Test adding a document
    doc_id = "test_doc_001"
    filename = "test_paper.pdf"
    
    print("Testing Pipeline Progress Queue...")
    
    # Add document
    if queue.add_document(doc_id, filename):
        print(f"✅ Added document: {doc_id}")
    
    # Update step status
    if queue.update_step_status(doc_id, "step_01_hash_validation", "complete"):
        print(f"✅ Updated step 1 status")
    
    # Get document status
    status = queue.get_document_status(doc_id)
    print(f"📊 Document status: {status['current_status']}")
    
    # Get pipeline summary
    summary = queue.get_pipeline_summary()
    print(f"📈 Pipeline summary: {summary}")
    
    # Clear queue
    if queue.clear_queue():
        print("🧹 Cleared queue")

if __name__ == "__main__":
    main()
