#!/usr/bin/env python3
"""
Extract Quick Metadata with Gemini for Deduplication

This tool uses Google Gemini Pro to extract metadata from the first 3 pages of PDFs
for the purpose of identifying duplicates before expensive full processing.
It's designed to be fast, reliable, and work across all publication formats.

Part of the Scientific Publications Pipeline - Step 3: Quick Metadata for Deduplication
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/extract_quick_metadata_with_gemini.log')
    ]
)
logger = logging.getLogger(__name__)

class GeminiQuickMetadataExtractor:
    """Extracts metadata from first 3 pages using Gemini Pro for deduplication."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Try to import Gemini
        self.gemini_available = self._import_gemini()
    
    def _import_gemini(self) -> bool:
        """Import Google Gemini library."""
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
            import os
            
            # Load environment variables
            load_dotenv()
            api_key = os.getenv('GOOGLE_API_KEY')
            
            if not api_key:
                logger.error("❌ GOOGLE_API_KEY not found in environment variables")
                logger.error("Please set GOOGLE_API_KEY in your .env file")
                return False
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Test the API with a simple call
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content("Hello")
            
            logger.info("✅ Gemini Pro API configured successfully")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Missing required package: {e}")
            logger.error("Please install: pip install google-generativeai python-dotenv")
            return False
        except Exception as e:
            logger.error(f"❌ Gemini API configuration failed: {e}")
            return False
    
    def create_metadata_prompt(self) -> str:
        """Create a prompt for Gemini to extract metadata for deduplication."""
        prompt = """
        Please analyze this document and extract metadata for deduplication purposes.
        Focus on the first 3 pages only. Extract exactly what you see.
        
        DOCUMENT CLASSIFICATION:
        - "research_paper": Academic journal article, preprint, or conference paper
        - "qa_interview": Q&A format, interview, or discussion
        - "patent": Patent application or patent document
        - "book_excerpt": Chapter or section from a book
        - "editorial": Editorial, commentary, or opinion piece
        - "review": Literature review or review article
        - "case_study": Case study or technical report
        - "other": Other document types
        
        METADATA EXTRACTION (for deduplication):
        
        For RESEARCH PAPERS:
        - title: Exact title as it appears
        - authors: Full names, separated by commas
        - journal: Journal or conference name
        - publication_year: Year of publication
        - doi: DOI if available
        - abstract: First paragraph of abstract if available
        
        For Q&A/INTERVIEWS:
        - title: Title or topic
        - participants: Interviewer and interviewee names
        - publication: Publication or platform name
        - publication_date: Date if available
        
        For PATENTS:
        - title: Patent title
        - inventors: Inventor names
        - patent_number: Patent number if visible
        
        For OTHER TYPES:
        - title: Document title
        - authors: Authors or contributors
        - publication_source: Where it was published
        
        IMPORTANT: Only extract what you can clearly see. Use null for missing fields.
        Be precise and extract exactly what appears in the document.
        
        Respond in this JSON format:
        {
            "document_type": "research_paper",
            "title": "exact title",
            "authors": ["Author 1", "Author 2"],
            "journal": "journal name",
            "publication_year": 2023,
            "doi": "10.1234/example",
            "abstract": "abstract text if available",
            "extraction_confidence": "high/medium/low"
        }
        """
        return prompt
    
    def extract_metadata_with_gemini(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata using Gemini Pro from first 3 pages."""
        if not self.gemini_available:
            logger.warning("⚠️  Gemini not available, returning basic metadata")
            return self._extract_basic_metadata(pdf_path)
        
        try:
            import google.generativeai as genai
            import fitz  # PyMuPDF
            
            # Create the metadata extraction prompt
            prompt = self.create_metadata_prompt()
            
            # Generate content with Gemini
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Create image parts for first 3 pages
            doc = fitz.open(pdf_path)
            image_parts = []
            
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                image_parts.append({
                    "mime_type": "image/png",
                    "data": base64.b64encode(img_data).decode('utf-8')
                })
            
            doc.close()
            
            # Generate response
            response = model.generate_content([prompt] + image_parts)
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                response_text = response.text
                
                # Look for JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response_text[json_start:json_end]
                    metadata = json.loads(json_str)
                    
                    # Add file information
                    metadata['filename'] = pdf_path.name
                    metadata['extraction_timestamp'] = datetime.now().isoformat()
                    metadata['extraction_method'] = 'gemini_pro'
                    metadata['pages_analyzed'] = min(3, len(fitz.open(pdf_path)))
                    
                    logger.info("✅ Successfully extracted metadata with Gemini")
                    return metadata
                else:
                    logger.warning("⚠️  No JSON found in Gemini response")
                    return self._extract_basic_metadata(pdf_path)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Failed to parse Gemini response as JSON: {e}")
                return self._extract_basic_metadata(pdf_path)
                
        except Exception as e:
            logger.error(f"❌ Gemini extraction failed: {e}")
            return self._extract_basic_metadata(pdf_path)
    
    def _extract_basic_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Fallback to basic metadata extraction if Gemini fails."""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            basic_metadata = {
                'filename': pdf_path.name,
                'document_type': 'research_paper',
                'title': None,
                'authors': [],
                'journal': None,
                'publication_year': None,
                'doi': None,
                'abstract': None,
                'extraction_confidence': 'low',
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_method': 'basic_fallback',
                'pages_analyzed': min(3, len(doc))
            }
            
            # Try to get basic info from first page
            if len(doc) > 0:
                first_page = doc[0]
                text = first_page.get_text()
                if text:
                    # Simple title extraction (first non-empty line)
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    if lines:
                        basic_metadata['title'] = lines[0][:200]  # Limit length
            
            doc.close()
            return basic_metadata
            
        except Exception as e:
            logger.error(f"Basic metadata extraction failed: {e}")
            return {
                'filename': pdf_path.name,
                'document_type': 'unknown',
                'extraction_confidence': 'very_low',
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_method': 'failed',
                'error': str(e)
            }
    
    def get_pdfs_to_process(self, max_files: int = None) -> List[Path]:
        """Get list of PDFs to process."""
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return []
        
        pdf_files = list(self.input_dir.glob("*.pdf"))
        pdf_files.extend(list(self.input_dir.glob("*.PDF")))
        
        if not pdf_files:
            logger.warning("No PDF files found in input directory")
            return []
        
        # TEMPORARY: Only process PDFs with today's date (for testing)
        today = datetime.now().strftime("%Y%m%d")
        today_pdfs = []
        for pdf_file in pdf_files:
            if today in pdf_file.name:
                today_pdfs.append(pdf_file)
                logger.info(f"Found today's PDF: {pdf_file.name}")
        
        if not today_pdfs:
            logger.info(f"No PDFs found with today's date ({today})")
            return []
        
        logger.info(f"Found {len(today_pdfs)} PDFs with today's date ({today})")
        
        # Check which files haven't been processed yet
        unprocessed_pdfs = []
        for pdf_file in today_pdfs:  # Changed from pdf_files to today_pdfs
            metadata_file = self.output_dir / f"{pdf_file.stem}_quick_metadata.json"
            if not metadata_file.exists():
                unprocessed_pdfs.append(pdf_file)
            else:
                logger.debug(f"Already processed: {pdf_file.name}")
        
        if not unprocessed_pdfs:
            logger.info("✅ All PDFs have already been processed!")
            return []
        
        # Apply max_files limit if specified
        if max_files:
            unprocessed_pdfs = unprocessed_pdfs[:max_files]
        
        logger.info(f"Found {len(unprocessed_pdfs)} unprocessed PDFs")
        return unprocessed_pdfs
    
    def process_file(self, pdf_path: Path) -> bool:
        """Process a single PDF file and extract quick metadata."""
        try:
            logger.info(f"Processing: {pdf_path.name}")
            
            # Extract metadata using Gemini
            metadata = self.extract_metadata_with_gemini(pdf_path)
            
            # Generate output filename
            output_file = self.output_dir / f"{pdf_path.stem}_quick_metadata.json"
            
            # Save metadata
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Successfully extracted quick metadata: {output_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path.name}: {e}")
            return False
    
    def process_files(self, max_files: int = None) -> Dict[str, int]:
        """Process multiple PDF files."""
        pdf_files = self.get_pdfs_to_process(max_files)
        
        if not pdf_files:
            return {'processed': 0, 'failed': 0, 'total': 0}
        
        processed = 0
        failed = 0
        
        for pdf_file in pdf_files:
            if self.process_file(pdf_file):
                processed += 1
            else:
                failed += 1
        
        results = {
            'processed': processed,
            'failed': failed,
            'total': len(pdf_files)
        }
        
        logger.info(f"📊 Processing complete: {processed} successful, {failed} failed")
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract quick metadata using Gemini Pro for deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all unprocessed PDFs
  python 03_extract_quick_metadata_with_gemini.py
  
  # Process only 5 files (useful for testing)
  python 03_extract_quick_metadata_with_gemini.py --max-files 5
  
  # Use custom input/output directories
  python 03_extract_quick_metadata_with_gemini.py --input-dir custom_input --output-dir custom_output
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path("data/source_data/preprocessed/sanitized/pdfs"),
        help="Directory containing sanitized PDF files (default: data/source_data/preprocessed/sanitized/pdfs)"
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("data/transformed_data/quick_metadata"),
        help="Directory to save quick metadata JSON files (default: data/transformed_data/quick_metadata)"
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
    extractor = GeminiQuickMetadataExtractor(args.input_dir, args.output_dir)
    results = extractor.process_files(args.max_files)
    
    # Exit with appropriate code
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 