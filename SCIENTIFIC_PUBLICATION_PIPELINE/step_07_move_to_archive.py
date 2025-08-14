#!/usr/bin/env python3
"""
Step 7: Move fully processed documents to archive directory.

This step moves documents that have completed steps 1-5 from the 'processing' 
directory to the 'step_07_archive' directory, indicating they are fully 
processed and ready for embedding.
"""

import logging
from pathlib import Path
from pipeline_progress_queue import PipelineProgressQueue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('archive_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArchiveMigration:
    """Move fully processed documents to archive directory."""
    
    def __init__(self):
        """Initialize the archive migration process."""
        self.progress_queue = PipelineProgressQueue()
        self.archive_dir = Path("step_07_archive")
        
        # Ensure archive directory exists
        self.archive_dir.mkdir(exist_ok=True)
        
        logger.info("Archive migration initialized successfully")
    
    def get_documents_ready_for_archive(self) -> list:
        """Get list of documents ready to be moved to archive."""
        return self.progress_queue.get_ready_for_archive()
    
    def move_document_to_archive(self, doc_id: str) -> bool:
        """Move a specific document to archive."""
        return self.progress_queue.move_to_archive(doc_id)
    
    def process_archive_migration(self) -> dict:
        """Process all documents ready for archive migration."""
        logger.info("🚀 Starting archive migration process...")
        
        # Get documents ready for archive
        ready_docs = self.get_documents_ready_for_archive()
        
        if not ready_docs:
            logger.info("✅ No documents ready for archive migration")
            return {
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'message': 'No documents ready for archive'
            }
        
        logger.info(f"📦 Found {len(ready_docs)} documents ready for archive")
        
        successful = 0
        failed = 0
        
        for doc_id in ready_docs:
            logger.info(f"🔄 Moving document to archive: {doc_id}")
            
            if self.move_document_to_archive(doc_id):
                successful += 1
                logger.info(f"✅ Successfully moved {doc_id} to archive")
            else:
                failed += 1
                logger.error(f"❌ Failed to move {doc_id} to archive")
        
        logger.info(f"📋 Archive Migration Summary:")
        logger.info(f"   Documents processed: {len(ready_docs)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        
        return {
            'processed': len(ready_docs),
            'successful': successful,
            'failed': failed,
            'message': f'Processed {len(ready_docs)} documents'
        }

def main():
    """Main execution function."""
    try:
        # Initialize archive migration
        archiver = ArchiveMigration()
        
        # Process archive migration
        result = archiver.process_archive_migration()
        
        print("\n" + "=" * 60)
        print("STEP 7: ARCHIVE MIGRATION COMPLETE")
        print("=" * 60)
        print(f"Message: {result['message']}")
        print(f"Documents processed: {result['processed']}")
        print(f"Successful: {result['successful']}")
        print(f"Failed: {result['failed']}")
        print("=" * 60)
        
        if result['failed'] == 0:
            print("✅ Archive migration completed successfully!")
        else:
            print("⚠️  Archive migration completed with some failures")
            
    except Exception as e:
        logger.error(f"Archive migration failed: {e}")
        print(f"❌ Archive migration failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
