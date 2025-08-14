#!/usr/bin/env python3
"""
Step 3: Text Extraction with Multi-Pass Consensus

This step extracts full text content from PDFs using Gemini 1.5 Pro with a sophisticated
multi-pass consensus approach. The strategy involves:

1. Chunking PDFs into overlapping 5-page segments
2. Running 3 parallel extraction passes
3. Building consensus from all successful extractions
4. Ensuring complete text capture with robust error handling

This addresses the common challenges of PDF text extraction by combining multiple
extraction attempts into a unified, complete result.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# Progress queue integration
try:
    from pipeline_progress_queue import PipelineProgressQueue
    PROGRESS_QUEUE_AVAILABLE = True
except ImportError:
    PROGRESS_QUEUE_AVAILABLE = False
    logging.warning("Progress queue not available. Pipeline tracking will be limited.")

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: uv add google-generativeai python-dotenv")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('text_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    """Extracts all text from PDFs using Gemini 1.5 Pro with multiple parallel passes."""
    
    def __init__(self, processing_dir: Path, extracted_text_dir: Path):
        self.processing_dir = processing_dir
        self.extracted_text_dir = extracted_text_dir
        
        # Create directories
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment and configure Gemini
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Configure the model
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=50,
            max_output_tokens=32768,  # Handle larger PDFs
            candidate_count=1
        )
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        self.model = genai.GenerativeModel(
            'gemini-1.5-pro',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Extraction parameters
        self.max_pages_per_chunk = 5
        self.overlap = 1
        self.max_retries = 3
        self.num_passes = 3
        
        # Initialize progress queue if available
        self.progress_queue = None
        if PROGRESS_QUEUE_AVAILABLE:
            try:
                self.progress_queue = PipelineProgressQueue()
                logger.info("Progress queue initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize progress queue: {e}")
                self.progress_queue = None
        
    def _load_extraction_prompt(self) -> str:
        """Load the text extraction prompt."""
        prompt = """
Extract all text content from this scientific PDF, from the very beginning to the very end, including any introductory material preceding the main title. Prioritize capturing all content that appears to be part of the scientific paper itself.

Specifically:

* **Include:** Absolutely all text from the beginning of the PDF to the end, including but not limited to the Introduction, Methods, Results, Discussion, Conclusion, and any other sections containing core scientific findings or arguments. This includes any content appearing *before* the main title or abstract. Preserve section headings and paragraph structure. Pay special attention to introductory sections and ensure no paragraphs are missed, regardless of their title (e.g., "WHAT DOES IT FEEL LIKE TO BE A PANCREAS?").

* **Section Headers with Page Numbers:** For each major section header (Introduction, Methods, Results, Discussion, Conclusion, etc.), include the page number where that section begins. Format section headers as "[PAGE X] SECTION TITLE" where X is the page number. For example: "[PAGE 3] INTRODUCTION" or "[PAGE 7] RESULTS AND DISCUSSION".

* **Exclude:** Headers, footers, page numbers, and publication metadata (e.g., journal name, publication date). References/bibliography sections, figure captions, table captions, and the content of figures and tables themselves should also be excluded.

* **Prioritize:** Complete capture of all textual content within the scientific paper, from start to finish. Accuracy and completeness of the entire text extraction are paramount. Do not cut off content mid-section.

* **Do Not:** Summarize, paraphrase, add commentary, or include any explanatory text of your own. Only extract the verbatim text from the PDF.

Return the extracted text as a single, continuous block of text with section headers marked with their page numbers.
"""
        return prompt.strip()
    
    def get_pdfs_to_process(self) -> List[Dict]:
        """Get list of PDFs to process from the processing directory that don't already have extracted text."""
        pdfs = []
        
        if not self.processing_dir.exists():
            logger.error(f"Processing directory does not exist: {self.processing_dir}")
            return pdfs
        
        # Find all PDF files in the processing directory
        pdf_files = list(self.processing_dir.glob("*.pdf"))
        pdf_files.extend(list(self.processing_dir.glob("*.PDF")))
        
        if not pdf_files:
            logger.warning("No PDF files found in processing directory")
            return pdfs
        
        # Get list of already processed PDFs from the output directory
        already_processed = set()
        if self.extracted_text_dir.exists():
            for text_file in self.extracted_text_dir.glob("*_extracted_text.txt"):
                # Parse filename: 7d121587_20250813_165059_671_extracted_text.txt -> 7d121587_20250813_165059_671.pdf
                base_name = text_file.stem.replace("_extracted_text", "")
                pdf_name = f"{base_name}.pdf"
                already_processed.add(pdf_name)
                logger.debug(f"Already processed: {pdf_name}")
        
        # Filter out already processed PDFs
        unprocessed_pdfs = []
        for pdf_path in pdf_files:
            if pdf_path.name not in already_processed:
                unprocessed_pdfs.append(pdf_path)
            else:
                logger.info(f"Skipping already processed PDF: {pdf_path.name}")
        
        if not unprocessed_pdfs:
            logger.info("✅ All PDFs have already been processed!")
            return pdfs
        
        for pdf_path in unprocessed_pdfs:
            pdf_metadata = {
                'original_filename': pdf_path.name,
                'file_path': pdf_path,
                'file_size': pdf_path.stat().st_size,
                'modified_time': datetime.fromtimestamp(pdf_path.stat().st_mtime)
            }
            pdfs.append(pdf_metadata)
        
        logger.info(f"Found {len(pdfs)} unprocessed PDFs to process")
        return pdfs
    
    def process_single_pass(self, pdf_path: Path, pdf_metadata: Dict, pass_num: int) -> Optional[str]:
        """Process a single extraction pass."""
        logger.info(f"🔄 Starting extraction pass {pass_num + 1}/{self.num_passes} for {pdf_metadata['original_filename']}")
        
        try:
            # Create a fresh reader for this pass to avoid thread safety issues
            from pypdf import PdfReader, PdfWriter
            import io
            
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            logger.info(f"📄 PDF has {total_pages} pages, processing in chunks of {self.max_pages_per_chunk} pages")
            
            pass_text = []
            
            for chunk_start in range(0, total_pages, self.max_pages_per_chunk - self.overlap):
                chunk_end = min(chunk_start + self.max_pages_per_chunk, total_pages)
                chunk_num = len(pass_text) + 1
                total_chunks = (total_pages + self.max_pages_per_chunk - self.overlap - 1) // (self.max_pages_per_chunk - self.overlap)
                
                logger.info(f"📄 Processing chunk {chunk_num}/{total_chunks}: pages {chunk_start + 1}-{chunk_end} (pass {pass_num + 1})")
                
                # Create a new PDF with just this chunk of pages
                writer = PdfWriter()
                
                for page_num in range(chunk_start, chunk_end):
                    writer.add_page(reader.pages[page_num])
                
                # Write chunk to memory
                chunk_pdf = io.BytesIO()
                writer.write(chunk_pdf)
                chunk_pdf.seek(0)
                pdf_content = chunk_pdf.getvalue()
                
                # Validate PDF chunk was created successfully
                if len(pdf_content) == 0:
                    logger.error(f"❌ Failed to create PDF chunk for pages {chunk_start + 1}-{chunk_end}")
                    return None
                
                # Prepare prompt for Gemini
                prompt = f"""
{self._load_extraction_prompt()}

This is chunk {chunk_num} of {total_chunks} (pass {pass_num + 1} of {self.num_passes}) covering pages {chunk_start + 1}-{chunk_end} of {total_pages}.

PDF content:
"""
                
                # Retry logic for each chunk
                chunk_text = None
                last_error = None
                
                for attempt in range(self.max_retries):
                    try:
                        logger.info(f"🔄 Attempt {attempt + 1}/{self.max_retries} for chunk {chunk_num} (pass {pass_num + 1})")
                        
                        # Send PDF chunk to Gemini
                        content_parts = [prompt]
                        content_parts.append({
                            "mime_type": "application/pdf",
                            "data": pdf_content
                        })
                        
                        response = self.model.generate_content(content_parts)
                        
                        if response.text and len(response.text.strip()) > 100:  # Minimum content validation
                            chunk_text = response.text.strip()
                            logger.info(f"✅ Successfully processed chunk {chunk_num}: pages {chunk_start + 1}-{chunk_end} ({len(chunk_text)} characters)")
                            break
                        else:
                            logger.warning(f"⚠️ Attempt {attempt + 1}: Empty or too short response for chunk {chunk_num} ({len(response.text) if response.text else 0} characters)")
                            if attempt < self.max_retries - 1:
                                time.sleep(2)  # Wait before retry
                            else:
                                last_error = "Empty or insufficient response from Gemini"
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Attempt {attempt + 1}: Error processing chunk {chunk_num}: {e}")
                        last_error = str(e)
                        if attempt < self.max_retries - 1:
                            time.sleep(2)  # Wait before retry
                
                # If all retries failed for this chunk, fail the entire process
                if chunk_text is None:
                    logger.error(f"❌ Failed to process chunk {chunk_num} after {self.max_retries} attempts. Last error: {last_error}")
                    logger.error(f"❌ Cannot proceed with incomplete text extraction. Failed at pages {chunk_start + 1}-{chunk_end}")
                    return None
                
                pass_text.append(chunk_text)
            
            # Combine chunks for this pass
            pass_combined = "\n\n".join(pass_text)
            
            logger.info(f"✅ Completed pass {pass_num + 1}: {len(pass_combined)} characters")
            return pass_combined
            
        except Exception as e:
            logger.error(f"❌ Error in pass {pass_num + 1}: {e}")
            return None
    
    def create_consensus_extraction(self, extractions: List[str], pdf_metadata: Dict) -> str:
        """Create a consensus extraction from multiple passes."""
        try:
            if not extractions or all(len(ext) == 0 for ext in extractions):
                logger.error("❌ No valid extractions found for consensus")
                return ""
            
            if len(extractions) == 1:
                logger.info("📝 Only one successful pass, using it directly")
                return extractions[0]
            
            logger.info(f"🤝 Creating consensus extraction from {len(extractions)} successful passes...")
            
            # Prepare all extractions for comparison
            extractions_text = "\n\n".join([f"EXTRACTION {i+1}:\n{ext}" for i, ext in enumerate(extractions)])
            
            prompt = f"""
Consolidate the following {len(extractions)} text extractions from a scientific PDF into a single, high-quality extraction. Each extraction is delimited by ```. The final output must contain the full, combined text of the extractions, prioritizing completeness, accuracy, and a coherent structure. Preserve all page numbers associated with section headers. If any extraction contains section headers with page numbers (e.g., "[PAGE 3] INTRODUCTION"), make sure to keep these page numbers in the final consensus. Include all content present in any of the extractions. Remove redundant information while preserving unique details. If discrepancies exist, choose the most complete and accurate version. Maintain the original document's logical flow, including section headers, figures, tables, and references. Do not include any commentary or descriptions about your selection process or the source of the content. Output only the combined, consolidated text of the scientific PDF.

```
{extractions_text}
```
"""
            
            response = self.model.generate_content(prompt)
            
            if response.text and len(response.text.strip()) > max(len(ext) for ext in extractions) * 0.3:
                logger.info(f"✅ Successfully created consensus extraction: {len(response.text)} characters")
                return response.text.strip()
            else:
                logger.warning("⚠️ Consensus creation failed, returning longest extraction")
                return max(extractions, key=len)
            
        except Exception as e:
            logger.error(f"❌ Failed to create consensus extraction: {e}")
            return max(extractions, key=len) if extractions else ""
    
    def extract_text_from_pdf(self, pdf_path: Path, pdf_metadata: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
        """Extract all text from a PDF using multiple parallel passes with consensus."""
        try:
            logger.info(f"📄 Extracting text from PDF: {pdf_metadata['original_filename']}")
            
            # Run all passes in parallel
            logger.info(f"🚀 Starting {self.num_passes} parallel extraction passes...")
            with ThreadPoolExecutor(max_workers=self.num_passes) as executor:
                # Submit all passes
                future_to_pass = {executor.submit(self.process_single_pass, pdf_path, pdf_metadata, pass_num): pass_num for pass_num in range(self.num_passes)}
                
                # Collect results
                all_extractions = []
                failed_passes = []
                for future in as_completed(future_to_pass):
                    pass_num = future_to_pass[future]
                    try:
                        result = future.result()
                        if result is not None:
                            all_extractions.append(result)
                            logger.info(f"✅ Pass {pass_num + 1} completed successfully")
                        else:
                            logger.error(f"❌ Pass {pass_num + 1} failed")
                            failed_passes.append(pass_num + 1)
                    except Exception as e:
                        logger.error(f"❌ Pass {pass_num + 1} failed with exception: {e}")
                        failed_passes.append(pass_num + 1)
            
            # Check if we have enough successful passes to proceed
            if len(all_extractions) < 1:
                logger.error(f"❌ No passes completed successfully. Cannot proceed with extraction.")
                return False, None, "No passes completed successfully"
            
            if len(all_extractions) < 2:
                logger.warning(f"⚠️ Only {len(all_extractions)} pass completed successfully. Using single pass extraction.")
                consensus_text = all_extractions[0]
            else:
                if failed_passes:
                    logger.warning(f"⚠️ {len(failed_passes)} passes failed: {failed_passes}. Proceeding with {len(all_extractions)} successful passes.")
                
                # Create consensus extraction from successful passes
                consensus_text = self.create_consensus_extraction(all_extractions, pdf_metadata)
            
            # Final validation - ensure we have substantial content
            if len(consensus_text.strip()) < 1000:
                logger.error(f"❌ Final text is too short ({len(consensus_text)} characters). Extraction may have failed.")
                return False, None, "Final text too short"
            
            logger.info(f"✅ Successfully extracted text: {len(consensus_text)} characters")
            return True, consensus_text, None
            
        except Exception as e:
            error_msg = f"Error extracting text from PDF: {e}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
    
    def save_extracted_text(self, pdf_metadata: Dict, extracted_text: str) -> bool:
        """Save extracted text to a file."""
        try:
            # Create filename for extracted text
            text_filename = f"{pdf_metadata['original_filename'].replace('.pdf', '')}_extracted_text.txt"
            text_file = self.extracted_text_dir / text_filename
            
            # Save the extracted text
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            logger.info(f"💾 Saved extracted text to: {text_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving extracted text: {e}")
            return False
    
    def process_pdf(self, pdf_metadata: Dict) -> Dict:
        """Process a single PDF through text extraction."""
        result = {
            'success': False,
            'pdf_filename': pdf_metadata['original_filename'],
            'text_length': 0,
            'text_file_path': None,
            'error_message': None
        }
        
        try:
            # Check if PDF file exists
            pdf_path = self.processing_dir / pdf_metadata['original_filename']
            if not pdf_path.exists():
                result['error_message'] = f"PDF file not found: {pdf_path}"
                return result
            
            # Find document ID from progress queue if available
            doc_id = None
            if self.progress_queue:
                try:
                    # Search for document with this filename in progress queue
                    for doc_id_candidate, doc_data in self.progress_queue._load_queue_data()["pipeline_progress"].items():
                        if doc_data.get("new_filename") == pdf_metadata['original_filename']:
                            doc_id = doc_id_candidate
                            break
                    
                    if doc_id:
                        logger.info(f"Found document {doc_id} in progress queue")
                    else:
                        logger.warning(f"Document {pdf_metadata['original_filename']} not found in progress queue")
                except Exception as e:
                    logger.warning(f"Progress queue lookup error: {e}")
                    doc_id = None
            
            # Extract text with Gemini (multiple passes with consensus)
            extraction_success, extracted_text, extraction_error = self.extract_text_from_pdf(pdf_path, pdf_metadata)
            
            if not extraction_success:
                result['error_message'] = extraction_error
                return result
            
            # Save extracted text
            save_success = self.save_extracted_text(pdf_metadata, extracted_text)
            
            if not save_success:
                result['error_message'] = "Failed to save extracted text"
                return result
            
            result['success'] = True
            result['text_length'] = len(extracted_text)
            result['text_file_path'] = str(self.extracted_text_dir / f"{pdf_metadata['original_filename'].replace('.pdf', '')}_extracted_text.txt")
            
            # Update progress queue if available
            if self.progress_queue and doc_id:
                try:
                    self.progress_queue.update_step_status(
                        doc_id, 
                        "step_03_text_extraction", 
                        "complete",
                        {"text_length": len(extracted_text)}
                    )
                    logger.info(f"Updated progress queue for {doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to update progress queue: {e}")
            
            logger.info(f"✅ Successfully extracted text from PDF: {pdf_metadata['original_filename']} ({len(extracted_text)} characters)")
            
        except Exception as e:
            result['error_message'] = f"Unexpected error: {e}"
            logger.error(f"❌ Error processing PDF {pdf_metadata['original_filename']}: {e}")
            
            # Update progress queue if available
            if self.progress_queue and doc_id:
                try:
                    self.progress_queue.add_error(doc_id, "step_03_text_extraction", f"Unexpected error: {e}")
                except Exception as queue_error:
                    logger.warning(f"Failed to update progress queue: {queue_error}")
        
        return result
    
    def run_extraction(self) -> Dict:
        """Run the text extraction process."""
        logger.info("🚀 Starting PDF text extraction with multiple parallel passes...")
        
        # Get PDFs to process
        pdfs = self.get_pdfs_to_process()
        logger.info(f"📊 Found {len(pdfs)} PDFs to extract text from")
        
        if not pdfs:
            logger.info("✅ No PDFs to process - all done!")
            return {'processed': 0, 'successful': 0, 'failed': 0, 'total_text_length': 0}
        
        # Process all PDFs in parallel
        logger.info(f"🚀 Processing {len(pdfs)} PDFs in parallel...")
        results = []
        successful = 0
        failed = 0
        total_text_length = 0
        
        with ThreadPoolExecutor(max_workers=len(pdfs)) as executor:
            # Submit all PDFs for processing
            future_to_pdf = {executor.submit(self.process_pdf, pdf_metadata): pdf_metadata for pdf_metadata in pdfs}
            
            # Collect results as they complete
            for future in as_completed(future_to_pdf):
                pdf_metadata = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        successful += 1
                        total_text_length += result['text_length']
                        logger.info(f"✅ Success: {pdf_metadata['original_filename']} ({result['text_length']} characters)")
                    else:
                        failed += 1
                        logger.error(f"❌ Failed: {pdf_metadata['original_filename']} - {result['error_message']}")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Exception processing {pdf_metadata['original_filename']}: {e}")
                    results.append({
                        'success': False,
                        'pdf_filename': pdf_metadata['original_filename'],
                        'text_length': 0,
                        'text_file_path': None,
                        'error_message': f"Exception: {e}"
                    })
        
        # Summary
        summary = {
            'processed': len(pdfs),
            'successful': successful,
            'failed': failed,
            'total_text_length': total_text_length,
            'results': results
        }
        
        logger.info("📋 Text Extraction Summary:")
        logger.info(f"   PDFs processed: {summary['processed']}")
        logger.info(f"   Successful: {summary['successful']}")
        logger.info(f"   Failed: {summary['failed']}")
        logger.info(f"   Total text extracted: {summary['total_text_length']} characters")
        
        return summary

def main():
    """Main function to run the text extraction pipeline."""
    parser = argparse.ArgumentParser(description="Step 3: Text Extraction with Multi-Pass Consensus")
    parser.add_argument(
        "--processing-dir",
        type=Path,
        default=Path("processing"),
        help="Directory containing PDFs from Step 2 (default: processing)"
    )
    parser.add_argument(
        "--extracted-text-dir",
        type=str,
        default="step_03_extracted_text",
        help="Directory to store extracted text files (default: step_03_extracted_text)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create extractor and process files
        extractor = PDFTextExtractor(args.processing_dir, Path(args.extracted_text_dir))
        summary = extractor.run_extraction()
        
        print("\n" + "="*60)
        print("STEP 3: TEXT EXTRACTION SUMMARY")
        print("="*60)
        print(f"Files processed: {summary['processed']}")
        print(f"Unique files: {summary['successful']}")
        print(f"Failed files: {summary['failed']}")
        print(f"Total text extracted: {summary['total_text_length']} characters")
        
        if summary['failed'] > 0:
            print("\n⚠️ Some files failed to process. Check logs for details.")
        else:
            print("\n✅ All files processed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error during text extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
