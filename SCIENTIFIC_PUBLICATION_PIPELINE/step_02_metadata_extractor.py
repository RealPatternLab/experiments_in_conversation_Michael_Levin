#!/usr/bin/env python3
"""
Step 2: Metadata Extraction with Gemini 1.5 Flash

This script extracts metadata from PDFs using Google Gemini 1.5 Flash to identify
publication details and check for metadata duplicates before proceeding to full processing.

Part of the Scientific Publications Pipeline - Step 2: Metadata Extraction
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import base64

# Progress queue integration
try:
    from pipeline_progress_queue import PipelineProgressQueue
    PROGRESS_QUEUE_AVAILABLE = True
except ImportError:
    PROGRESS_QUEUE_AVAILABLE = False
    logging.warning("Progress queue not available. Pipeline tracking will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('step_02_metadata_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

class MetadataExtractor:
    """Extracts metadata from PDFs using Gemini 1.5 Flash for deduplication."""
    
    def __init__(self, input_dir: Path, output_dir: Path, metadata_dir: str = "step_02_metadata"):
        self.input_dir = input_dir
        self.output_dir = output_dir  # Keep for compatibility but not used
        self.metadata_dir = Path(metadata_dir)
        
        # Create metadata directory if it doesn't exist
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to import Gemini
        self.gemini_available = self._import_gemini()
        
        # Initialize progress queue if available
        self.progress_queue = None
        if PROGRESS_QUEUE_AVAILABLE:
            try:
                self.progress_queue = PipelineProgressQueue()
                logger.info("Progress queue initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize progress queue: {e}")
                self.progress_queue = None
    
    def _import_gemini(self) -> bool:
        """Import Google Gemini library."""
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
            
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
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Hello")
            
            logger.info("✅ Gemini 1.5 Flash API configured successfully")
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
        Focus on the first 2 pages only. Extract exactly what you see.
        
        CRITICAL INSTRUCTION: Systematically scan ALL areas of each page including:
        - Main content areas (center of page)
        - Headers and footers (even small print)
        - Margins and edges
        - Bottom of pages for DOIs and identifiers
        - Any small text that might contain important metadata
        - Page numbers, publication details, and copyright information
        
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
        - title: Exact title as it appears (usually large text at top)
        - authors: Full names, separated by commas (usually below title)
        - journal: Journal or conference name (often in header/footer or below title)
        - publication_year: Year of publication (check headers, footers, and main text)
        - doi: DOI if available (CRITICAL: Check footers, headers, and small print at bottom of pages)
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
        
        SEARCH STRATEGY:
        1. Start with the main content area (center of page)
        2. Then scan the header area (top of page)
        3. Then scan the footer area (bottom of page) - especially for DOIs
        4. Check left and right margins for additional information
        5. Look for any small print, even if it seems unimportant
        
        IMPORTANT: Only extract what you can clearly see. Use null for missing fields.
        Be precise and extract exactly what appears in the document.
        Pay special attention to small text in footers and headers.
        
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
        """Extract metadata using Gemini 1.5 Flash from first 2 pages."""
        if not self.gemini_available:
            logger.warning("⚠️  Gemini not available, returning basic metadata")
            return self._extract_basic_metadata(pdf_path)
        
        try:
            import google.generativeai as genai
            import fitz  # PyMuPDF
            
            # Create the metadata extraction prompt
            prompt = self.create_metadata_prompt()
            
            # Generate content with Gemini
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Create image parts for first 2 pages
            doc = fitz.open(pdf_path)
            image_parts = []
            
            for page_num in range(min(2, len(doc))):
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
                    metadata['extraction_method'] = 'gemini_1.5_flash'
                    metadata['pages_analyzed'] = min(2, len(fitz.open(pdf_path)))
                    
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
                'pages_analyzed': min(2, len(doc))
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
    
    def check_metadata_duplicate(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if metadata matches an existing publication.
        
        Args:
            metadata: Extracted metadata to check
            
        Returns:
            Tuple of (is_duplicate, matching_filename) where matching_filename is None if no match
        """
        try:
            # Check all existing metadata JSON files in the metadata directory
            for json_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        stored_metadata = json.load(f)
                        
                        # Check for exact matches on key fields
                        if self._is_metadata_duplicate(metadata, stored_metadata):
                            logger.info(f"Metadata duplicate found: {metadata.get('title', 'Unknown')} matches {stored_metadata.get('filename', 'Unknown')}")
                            return True, stored_metadata.get('filename')
                except Exception as e:
                    logger.warning(f"Error reading metadata file {json_file}: {e}")
                    continue
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking metadata files: {e}")
            return False, None
    
    def _is_metadata_duplicate(self, new_metadata: Dict[str, Any], stored_metadata: Dict[str, Any]) -> bool:
        """Check if two metadata records represent the same publication."""
        # Check title similarity (case-insensitive)
        new_title = new_metadata.get('title', '').lower().strip()
        stored_title = stored_metadata.get('title', '').lower().strip()
        
        if new_title and stored_title and new_title == stored_title:
            return True
        
        # Check DOI match
        new_doi = new_metadata.get('doi', '').lower().strip()
        stored_doi = stored_metadata.get('doi', '').lower().strip()
        
        if new_doi and stored_doi and new_doi == stored_doi:
            return True
        
        # Check author list similarity (if both have authors)
        new_authors = set(a.lower().strip() for a in new_metadata.get('authors', []))
        stored_authors = set(a.lower().strip() for a in stored_metadata.get('authors', []))
        
        if new_authors and stored_authors and len(new_authors.intersection(stored_authors)) >= 2:
            return True
        
        return False
    
    def record_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata as individual JSON file named after the PDF filename."""
        try:
            # Get the base filename without extension
            base_filename = Path(metadata.get('filename', 'unknown')).stem
            
            # Create JSON filename
            json_filename = f"{base_filename}_metadata.json"
            json_path = self.metadata_dir / json_filename
            
            # Save metadata as JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved metadata to: {json_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def process_file(self, pdf_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Process a single PDF file through metadata extraction and validation.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            Tuple of (is_unique, new_filename) where new_filename is None if file was deleted
        """
        filename = pdf_path.name
        logger.info(f"Processing file: {filename}")
        
        # Find document ID from progress queue if available
        doc_id = None
        if self.progress_queue:
            try:
                # Search for document with this filename in progress queue
                for doc_id_candidate, doc_data in self.progress_queue._load_queue_data()["pipeline_progress"].items():
                    if doc_data.get("new_filename") == filename:
                        doc_id = doc_id_candidate
                        break
                
                if doc_id:
                    logger.info(f"Found document {doc_id} in progress queue")
                else:
                    logger.warning(f"Document {filename} not found in progress queue")
            except Exception as e:
                logger.warning(f"Progress queue lookup error: {e}")
                doc_id = None
        
        try:
            # Step 1: Extract metadata using Gemini
            metadata = self.extract_metadata_with_gemini(pdf_path)
            
            # Step 2: Check for metadata duplicates
            is_duplicate, matching_filename = self.check_metadata_duplicate(metadata)
            
            if is_duplicate:
                # File has duplicate metadata - log and delete
                logger.warning(
                    f"File {filename} has duplicate metadata matching: {matching_filename}"
                )
                
                # Update progress queue if available
                if self.progress_queue and doc_id:
                    try:
                        self.progress_queue.add_error(
                            doc_id, 
                            "step_02_metadata_extraction", 
                            f"Duplicate metadata - matches {matching_filename}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update progress queue: {e}")
                
                # Delete the duplicate file
                os.remove(pdf_path)
                logger.info(f"Deleted duplicate file: {filename}")
                
                return False, None  # File was not unique, was deleted
            else:
                # File is unique - record metadata and keep in processing directory
                self.record_metadata(metadata)
                
                # Update progress queue if available
                if self.progress_queue and doc_id:
                    try:
                        self.progress_queue.update_step_status(
                            doc_id, 
                            "step_02_metadata_extraction", 
                            "complete"
                        )
                        logger.info(f"Updated progress queue for {doc_id}")
                    except Exception as e:
                        logger.warning(f"Failed to update progress queue: {e}")
                
                logger.info(
                    f"File {filename} is unique. Metadata extracted and file ready for next step."
                )
                
                return True, filename  # File is unique, proceed to next step
                
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise
    
    def get_pdfs_to_process(self) -> List[Path]:
        """Get list of PDFs to process from input directory."""
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return []
        
        pdf_files = list(self.input_dir.glob("*.pdf"))
        pdf_files.extend(list(self.input_dir.glob("*.PDF")))
        
        if not pdf_files:
            logger.warning("No PDF files found in input directory")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        return pdf_files
    
    def process_files(self) -> Dict[str, int]:
        """Process all PDF files in the input directory."""
        pdf_files = self.get_pdfs_to_process()
        
        if not pdf_files:
            return {'processed': 0, 'duplicates': 0, 'total': 0}
        
        processed = 0
        duplicates = 0
        
        for pdf_file in pdf_files:
            try:
                is_unique, new_filename = self.process_file(pdf_file)
                
                if is_unique:
                    processed += 1
                else:
                    duplicates += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
        
        results = {
            'processed': processed,
            'duplicates': duplicates,
            'total': len(pdf_files)
        }
        
        logger.info(f"📊 Processing complete: {processed} unique, {duplicates} duplicates")
        return results


def main():
    """Main function to run the metadata extraction pipeline."""
    parser = argparse.ArgumentParser(description="Step 2: Metadata Extraction with Gemini")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("processing"),
        help="Directory containing unique PDFs from Step 1 (default: processing)"
    )
    

    
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="step_02_metadata",
        help="Directory to store metadata JSON files (default: step_02_metadata)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create extractor and process files
    extractor = MetadataExtractor(args.input_dir, Path("processing"), args.metadata_dir)
    results = extractor.process_files()
    
    # Print summary
    print("\n" + "=" * 60)
    print("STEP 2: METADATA EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Files processed: {results['total']}")
    print(f"Unique files: {results['processed']}")
    print(f"Duplicate files: {results['duplicates']}")
    print("=" * 60)
    
    # Exit with appropriate code
    if results['duplicates'] > 0:
        logger.info("Some files were duplicates and were deleted")
    
    return 0


if __name__ == "__main__":
    exit(main())
