#!/usr/bin/env python3
"""
Deduplicate PDFs and Move Duplicates to DLQ

This tool compares quick metadata extracted from PDFs to identify duplicates
and moves them to the Dead Letter Queue (DLQ) before expensive full processing.
It's designed to work with the Gemini-based quick metadata extraction.

Part of the Scientific Publications Pipeline - Step 4: Deduplication
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/deduplicate_pdfs_and_move_to_dlq.log')
    ]
)
logger = logging.getLogger(__name__)

class PDFDeduplicator:
    """
    Deduplicate PDFs based on quick metadata and move duplicates to DLQ.
    
    When duplicates are found:
    - PDF files are moved to DLQ for potential review/recovery
    - Metadata files are deleted (not needed since we keep the original)
    - Only the highest confidence version of each document is retained
    """
    
    def __init__(self, metadata_dir: Path, pdf_dir: Path, dlq_dir: Path):
        self.metadata_dir = metadata_dir
        self.pdf_dir = pdf_dir
        self.dlq_dir = dlq_dir
        
        # Create DLQ directory if it doesn't exist
        self.dlq_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
    
    def normalize_text(self, text: Optional[str]) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        # Convert to lowercase, remove extra whitespace, normalize punctuation
        return text.lower().strip().replace('  ', ' ')
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1, text2).ratio()
    
    def are_pdfs_duplicates(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Determine if two PDFs are duplicates based on metadata comparison.
        
        Returns:
            (is_duplicate, reason, confidence_score)
        """
        # Check document type first
        if metadata1.get('document_type') != metadata2.get('document_type'):
            return False, "Different document types", 0.0
        
        # Compare titles
        title1 = self.normalize_text(metadata1.get('title'))
        title2 = self.normalize_text(metadata2.get('title'))
        
        if title1 and title2:
            title_similarity = self.calculate_similarity(title1, title2)
            if title_similarity > 0.9:  # Very similar titles
                return True, f"Similar titles (similarity: {title_similarity:.2f})", title_similarity
        
        # Compare authors
        authors1 = metadata1.get('authors', [])
        authors2 = metadata2.get('authors', [])
        
        if authors1 and authors2:
            # Normalize author lists
            norm_authors1 = [self.normalize_text(auth) for auth in authors1 if auth]
            norm_authors2 = [self.normalize_text(auth) for auth in authors2 if auth]
            
            if norm_authors1 and norm_authors2:
                # Check if there's significant author overlap
                common_authors = set(norm_authors1) & set(norm_authors2)
                if len(common_authors) >= min(len(norm_authors1), len(norm_authors2)) * 0.7:
                    return True, f"Significant author overlap: {list(common_authors)}", 0.8
        
        # Compare DOIs
        doi1 = metadata1.get('doi')
        doi2 = metadata2.get('doi')
        
        if doi1 and doi2 and doi1 == doi2:
            return True, f"Same DOI: {doi1}", 1.0
        
        # Compare journals
        journal1 = self.normalize_text(metadata1.get('journal'))
        journal2 = self.normalize_text(metadata2.get('journal'))
        
        if journal1 and journal2:
            journal_similarity = self.calculate_similarity(journal1, journal2)
            if journal_similarity > 0.8:
                # If journals are very similar AND titles are somewhat similar
                if title1 and title2:
                    title_similarity = self.calculate_similarity(title1, title2)
                    if title_similarity > 0.7:
                        return True, f"Similar journal and title (journal similarity: {journal_similarity:.2f}, title similarity: {title_similarity:.2f})", (journal_similarity + title_similarity) / 2
        
        # Compare publication years
        year1 = metadata1.get('publication_year')
        year2 = metadata2.get('publication_year')
        
        if year1 and year2 and year1 == year2:
            # Same year + similar title might indicate duplicate
            if title1 and title2:
                title_similarity = self.calculate_similarity(title1, title2)
                if title_similarity > 0.8:
                    return True, f"Same year ({year1}) and similar title (similarity: {title_similarity:.2f})", title_similarity
        
        return False, "No significant similarity detected", 0.0
    
    def load_metadata_files(self) -> List[Dict[str, Any]]:
        """Load all quick metadata files."""
        metadata_files = list(self.metadata_dir.glob("*_quick_metadata.json"))
        
        if not metadata_files:
            logger.warning("No quick metadata files found")
            return []
        
        metadata_list = []
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata['metadata_file'] = metadata_file
                    metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Error loading {metadata_file.name}: {e}")
        
        logger.info(f"Loaded {len(metadata_list)} metadata files")
        return metadata_list
    
    def find_duplicates(self, metadata_list: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any], str, float]]:
        """Find duplicate PDFs based on metadata comparison."""
        duplicates = []
        processed = set()
        
        for i, metadata1 in enumerate(metadata_list):
            if i in processed:
                continue
            
            for j, metadata2 in enumerate(metadata_list[i+1:], i+1):
                if j in processed:
                    continue
                
                is_duplicate, reason, confidence = self.are_pdfs_duplicates(metadata1, metadata2)
                
                if is_duplicate:
                    # Determine which one to keep (usually the one with higher confidence)
                    keep_metadata = metadata1 if metadata1.get('extraction_confidence', 'low') >= metadata2.get('extraction_confidence', 'low') else metadata2
                    remove_metadata = metadata2 if keep_metadata == metadata1 else metadata1
                    
                    duplicates.append((keep_metadata, remove_metadata, reason, confidence))
                    processed.add(i)
                    processed.add(j)
        
        return duplicates
    
    def move_to_dlq(self, pdf_filename: str, reason: str) -> bool:
        """Move a duplicate PDF to DLQ and delete its metadata."""
        try:
            # Find the PDF file
            pdf_path = self.pdf_dir / pdf_filename
            if not pdf_path.exists():
                logger.warning(f"PDF file not found: {pdf_filename}")
                return False
            
            # Create a unique filename in DLQ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dlq_filename = f"{pdf_path.stem}_duplicate_{timestamp}.pdf"
            dlq_path = self.dlq_dir / dlq_filename
            
            # Move the PDF to DLQ
            shutil.move(str(pdf_path), str(dlq_path))
            logger.info(f"✅ Moved PDF to DLQ: {pdf_filename} → {dlq_filename}")
            
            # Delete the metadata file (not needed since we're keeping the original)
            metadata_filename = f"{pdf_filename.replace('.pdf', '')}_quick_metadata.json"
            metadata_path = self.metadata_dir / metadata_filename
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info(f"🗑️  Deleted metadata: {metadata_filename}")
            else:
                logger.info(f"ℹ️  No metadata file found to delete: {metadata_filename}")
            
            logger.info(f"✅ Successfully processed duplicate: {pdf_filename} (Reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing duplicate {pdf_filename}: {e}")
            return False
    
    def run_deduplication(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run the deduplication process."""
        logger.info("🔍 Starting PDF deduplication...")
        
        # Load metadata
        metadata_list = self.load_metadata_files()
        if not metadata_list:
            return {'error': 'No metadata files found'}
        
        # Find duplicates
        duplicates = self.find_duplicates(metadata_list)
        logger.info(f"🔍 Found {len(duplicates)} duplicate pairs")
        
        # Process duplicates
        moved_count = 0
        failed_moves = 0
        
        for keep_metadata, remove_metadata, reason, confidence in duplicates:
            logger.info(f"🔄 Processing duplicate:")
            logger.info(f"   Keep: {keep_metadata['filename']}")
            logger.info(f"   Remove: {remove_metadata['filename']}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Confidence: {confidence:.2f}")
            
            if not dry_run:
                if self.move_to_dlq(remove_metadata['filename'], reason):
                    moved_count += 1
                else:
                    failed_moves += 1
            else:
                logger.info(f"   [DRY RUN] Would move PDF to DLQ and delete metadata: {remove_metadata['filename']}")
        
        # Summary
        remaining_pdfs = len([m for m in metadata_list if not any(m == remove_m for _, remove_m, _, _ in duplicates)])
        
        result = {
            'total_pdfs': len(metadata_list),
            'duplicates_found': len(duplicates),
            'moved_to_dlq': moved_count,
            'failed_moves': failed_moves,
            'remaining_pdfs': remaining_pdfs,
            'dry_run': dry_run
        }
        
        logger.info("📋 Deduplication Summary:")
        logger.info(f"   Total PDFs: {result['total_pdfs']}")
        logger.info(f"   Duplicates found: {result['duplicates_found']}")
        logger.info(f"   PDFs moved to DLQ: {result['moved_to_dlq']}")
        logger.info(f"   Failed operations: {result['failed_moves']}")
        logger.info(f"   Remaining PDFs: {result['remaining_pdfs']}")
        
        return result

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Deduplicate PDFs based on quick metadata. Duplicates are moved to DLQ, metadata files are deleted."
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/transformed_data/quick_metadata"),
        help="Directory containing quick metadata JSON files (default: data/transformed_data/quick_metadata)"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/source_data/preprocessed/sanitized/pdfs"),
        help="Directory containing PDF files (default: data/source_data/preprocessed/sanitized/pdfs)"
    )
    parser.add_argument(
        "--dlq-dir",
        type=Path,
        default=Path("data/source_data/DLQ"),
        help="Directory to move duplicates to (default: data/source_data/DLQ)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually moving files"
    )
    parser.add_argument(
        "--verbose", '-v',
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate directories
    if not args.metadata_dir.exists():
        logger.error(f"Metadata directory does not exist: {args.metadata_dir}")
        sys.exit(1)
    
    if not args.pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {args.pdf_dir}")
        sys.exit(1)
    
    try:
        deduplicator = PDFDeduplicator(args.metadata_dir, args.pdf_dir, args.dlq_dir)
        result = deduplicator.run_deduplication(dry_run=args.dry_run)
        
        if 'error' in result:
            logger.error(f"❌ Deduplication failed: {result['error']}")
            sys.exit(1)
        
        if args.dry_run:
            print("✅ Dry run completed - no files were moved")
        else:
            print("✅ Deduplication completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error during deduplication: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 