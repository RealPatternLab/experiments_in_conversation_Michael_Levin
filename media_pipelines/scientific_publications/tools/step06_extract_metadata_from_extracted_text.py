#!/usr/bin/env python3
"""
Extract Metadata from Text Tool

This tool analyzes extracted text files from PDFs to extract basic metadata
including potential titles, authors, abstracts, DOIs, and publication information.
It creates initial metadata JSON files that can later be enriched with external APIs.

Part of the Scientific Publications Pipeline - Step 4: Metadata Extraction
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/extract_metadata_from_text.log')
    ]
)
logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Extracts metadata from extracted text files."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
    
    def extract_title(self, text: str) -> Optional[str]:
        """Extract potential title from the beginning of the text."""
        lines = text.strip().split('\n')
        
        # Look for title in first few lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Skip lines that look like headers or metadata
                if not re.match(r'^(Abstract|Introduction|Abstract:|Introduction:|[A-Z\s]+:?\s*\d+$)', line):
                    return line
        
        return None
    
    def extract_authors(self, text: str) -> List[str]:
        """Extract potential author names from the text."""
        # Common author patterns
        author_patterns = [
            r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'Authors?:\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*)\s+et al\.',
        ]
        
        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, text[:2000], re.IGNORECASE)
            for match in matches:
                if match and len(match.strip()) > 5:
                    authors.append(match.strip())
        
        return list(set(authors))  # Remove duplicates
    
    def extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from the text."""
        doi_pattern = r'10\.\d{4,}/[-._;()/:\w]+'
        match = re.search(doi_pattern, text)
        return match.group() if match else None
    
    def extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from the text."""
        # Look for abstract section
        abstract_patterns = [
            r'Abstract[:\s]*([^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z]|$)',
            r'ABSTRACT[:\s]*([^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z]|$)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50 and len(abstract) < 2000:
                    return abstract
        
        return None
    
    def extract_publication_info(self, text: str) -> Dict[str, Any]:
        """Extract publication-related information."""
        info = {}
        
        # Look for journal names
        journal_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Journal|Review|Letters|Proceedings))',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Transactions|Communications|Reports))',
        ]
        
        for pattern in journal_patterns:
            match = re.search(pattern, text[:3000])
            if match:
                info['journal'] = match.group(1).strip()
                break
        
        # Look for publication year
        year_pattern = r'(?:19|20)\d{2}'
        year_match = re.search(year_pattern, text[:2000])
        if year_match:
            info['publication_year'] = int(year_match.group())
        
        # Look for volume/issue numbers
        volume_pattern = r'Vol(?:ume)?\.?\s*(\d+)'
        volume_match = re.search(volume_pattern, text[:3000], re.IGNORECASE)
        if volume_match:
            info['volume'] = int(volume_match.group(1))
        
        return info
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from the text."""
        # Look for keywords section
        keywords_pattern = r'Keywords?[:\s]*([^\n]+(?:\n[^\n]+)*?)(?=\n[A-Z]|$)'
        match = re.search(keywords_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            keywords_text = match.group(1).strip()
            # Split by common separators
            keywords = re.split(r'[,;]\s*', keywords_text)
            # Clean up keywords
            keywords = [kw.strip() for kw in keywords if kw.strip() and len(kw.strip()) > 2]
            return keywords[:20]  # Limit to 20 keywords
        
        return []
    
    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from text."""
        metadata = {
            'filename': filename,
            'extraction_timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'title': self.extract_title(text),
            'authors': self.extract_authors(text),
            'doi': self.extract_doi(text),
            'abstract': self.extract_abstract(text),
            'keywords': self.extract_keywords(text),
            'extraction_method': 'rule_based',
            'confidence_score': 0.7,  # Base confidence for rule-based extraction
        }
        
        # Add publication info
        publication_info = self.extract_publication_info(text)
        metadata.update(publication_info)
        
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return metadata
    
    def get_text_files_to_process(self, max_files: int = None) -> List[Path]:
        """Get list of text files to process."""
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return []
        
        text_files = list(self.input_dir.glob("*_extracted_text.txt"))
        
        if not text_files:
            logger.warning("No extracted text files found in input directory")
            return []
        
        # Check which files haven't been processed yet
        unprocessed_files = []
        for text_file in text_files:
            metadata_file = self.output_dir / f"{text_file.stem.replace('_extracted_text', '')}_metadata.json"
            if not metadata_file.exists():
                unprocessed_files.append(text_file)
            else:
                logger.debug(f"Already processed: {text_file.name}")
        
        if not unprocessed_files:
            logger.info("✅ All text files have already been processed!")
            return []
        
        # Apply max_files limit if specified
        if max_files:
            unprocessed_files = unprocessed_files[:max_files]
        
        logger.info(f"Found {len(unprocessed_files)} unprocessed text files")
        return unprocessed_files
    
    def process_file(self, text_file: Path) -> bool:
        """Process a single text file and extract metadata."""
        try:
            logger.info(f"Processing: {text_file.name}")
            
            # Read the text file
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract metadata
            metadata = self.extract_metadata(text, text_file.name)
            
            # Generate output filename
            base_name = text_file.stem.replace('_extracted_text', '')
            output_file = self.output_dir / f"{base_name}_metadata.json"
            
            # Save metadata
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Successfully extracted metadata: {output_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {text_file.name}: {e}")
            return False
    
    def process_files(self, max_files: int = None) -> Dict[str, int]:
        """Process multiple text files."""
        text_files = self.get_text_files_to_process(max_files)
        
        if not text_files:
            return {'processed': 0, 'failed': 0, 'total': 0}
        
        processed = 0
        failed = 0
        
        for text_file in text_files:
            if self.process_file(text_file):
                processed += 1
            else:
                failed += 1
        
        results = {
            'processed': processed,
            'failed': failed,
            'total': len(text_files)
        }
        
        logger.info(f"📊 Processing complete: {processed} successful, {failed} failed")
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from extracted text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all unprocessed text files
  python 04_extract_metadata_from_text.py
  
  # Process only 5 files (useful for testing)
  python 04_extract_metadata_from_text.py --max-files 5
  
  # Use custom input/output directories
  python 04_extract_metadata_from_text.py --input-dir custom_input --output-dir custom_output
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path("data/transformed_data/text_extraction"),
        help="Directory containing extracted text files (default: data/transformed_data/text_extraction)"
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("data/transformed_data/metadata_extraction"),
        help="Directory to save metadata JSON files (default: data/transformed_data/metadata_extraction)"
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help="Maximum number of files to process (useful for testing)"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate directories
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create extractor and process files
    extractor = MetadataExtractor(args.input_dir, args.output_dir)
    results = extractor.process_files(args.max_files)
    
    # Exit with appropriate code
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 