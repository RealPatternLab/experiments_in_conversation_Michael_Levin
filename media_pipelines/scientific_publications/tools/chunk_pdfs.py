#!/usr/bin/env python3
"""
Semantic PDF Chunking Tool

This tool processes PDFs using Gemini Pro to extract semantic chunks.
It loads the generated prompt, processes PDFs, validates responses with Pydantic,
and stores chunks in the database with comprehensive metadata.
"""

import argparse
import json
import os
import sqlite3
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
import multiprocessing
from functools import partial

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    from chunk_models import validate_gemini_response, create_enriched_chunk, SemanticChunksResponse, SemanticChunk
    from db_utils import get_db_connection
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: pip install google-generativeai python-dotenv pydantic")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/chunk_pdfs.log')
    ]
)
logger = logging.getLogger(__name__)

class SemanticPDFChunker:
    """Handles semantic chunking of PDFs using Gemini Pro."""
    
    def __init__(self, db_path: Path, ingested_dir: Path, chunked_dir: Path, prompt_file: Path):
        self.db_path = db_path
        self.ingested_dir = ingested_dir
        self.chunked_dir = chunked_dir
        self.prompt_file = prompt_file
        
        # Create directories
        self.chunked_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment and configure Gemini
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Configure the model with generation parameters for complete responses
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Low temperature for consistent output
            top_p=0.8,
            top_k=40,
            max_output_tokens=32768,  # Increased from 8192 to handle larger PDFs
            candidate_count=1
        )
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            'gemini-1.5-pro',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Load the chunking prompt
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """Load the chunking prompt from file."""
        if not self.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        logger.info(f"📝 Loaded chunking prompt ({len(prompt)} characters)")
        return prompt
    
    def get_pdfs_to_process(self, max_files: int = 5) -> List[Dict]:
        """Get PDFs that need to be processed by looking at the ingestion directory directly."""
        logger.info(f"🔍 Scanning ingestion directory: {self.ingested_dir}")
        
        if not self.ingested_dir.exists():
            logger.error(f"❌ Ingestion directory does not exist: {self.ingested_dir}")
            return []
        
        # Get all PDF files in the ingestion directory
        pdf_files = list(self.ingested_dir.glob("*.pdf"))
        logger.info(f"📄 Found {len(pdf_files)} PDF files in ingestion directory")
        
        # Check which PDFs are already processed (have corresponding files in chunked directory)
        chunked_pdfs = {f.stem for f in Path("data/semantically-chunked/pdfs").glob("*.pdf")}
        logger.info(f"📁 Found {len(chunked_pdfs)} already processed PDFs in chunked directory")
        
        # Filter out already processed PDFs
        unprocessed_pdfs = []
        for pdf_file in pdf_files:
            pdf_stem = pdf_file.stem
            if pdf_stem not in chunked_pdfs:
                unprocessed_pdfs.append(pdf_file)
                logger.info(f"✅ Found unprocessed PDF: {pdf_file.name}")
            else:
                logger.info(f"⏭️  Skipping already processed PDF: {pdf_file.name}")
        
        logger.info(f"📊 Total unprocessed PDFs: {len(unprocessed_pdfs)}")
        
        # Limit to max_files
        pdfs_to_process = unprocessed_pdfs[:max_files]
        logger.info(f"🎯 Will process {len(pdfs_to_process)} PDFs (limited by max_files={max_files})")
        
        # Convert to metadata format (simplified since we're not using database)
        pdfs = []
        for pdf_file in pdfs_to_process:
            try:
                # Get basic file info
                file_size = pdf_file.stat().st_size
                original_filename = pdf_file.name
                sanitized_filename = pdf_file.name  # Same as original for now
                
                logger.info(f"📋 Preparing metadata for: {original_filename} ({file_size} bytes)")
                
                pdfs.append({
                    'id': None,  # No database ID
                    'original_filename': original_filename,
                    'sanitized_filename': sanitized_filename,
                    'file_size': file_size,
                    'title': 'Unknown',  # Will be extracted during processing
                    'authors': 'Unknown',
                    'doi': None,
                    'publication_date': None,
                    'journal': 'Unknown',
                    'document_type': 'research_paper',  # Default assumption
                    'confidence_score': 0.0,
                    'extraction_method': 'file_scan',
                    'metadata_extraction_id': None
                })
                
            except Exception as e:
                logger.error(f"❌ Error preparing metadata for {pdf_file.name}: {e}")
                continue
        
        logger.info(f"✅ Prepared {len(pdfs)} PDFs for processing")
        return pdfs
    
    def process_pdf_with_gemini(self, pdf_path: Path, pdf_metadata: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
        """Process a PDF with Gemini Pro and return the response."""
        try:
            logger.info(f"🤖 Processing PDF with Gemini: {pdf_metadata['original_filename']}")
            
            # Step 1: Read the PDF using pypdf
            logger.info("📄 Reading PDF with pypdf...")
            from pypdf import PdfReader, PdfWriter
            import io
            
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            logger.info(f"📄 Found {total_pages} pages in PDF")
            
            # Step 2: Create PDF writer and add all pages
            writer = PdfWriter()
            for page_num in range(total_pages):
                writer.add_page(reader.pages[page_num])
            
            # Step 3: Convert PDF to binary
            logger.info("🔄 Converting PDF to binary...")
            pdf_buffer = io.BytesIO()
            writer.write(pdf_buffer)
            pdf_buffer.seek(0)
            pdf_binary = pdf_buffer.getvalue()
            
            logger.info(f"📄 Binary size: {len(pdf_binary)} bytes")
            
            # Step 4: Create the full prompt with explicit JSON requirements
            full_prompt = f"""
{self.prompt}

CRITICAL JSON FORMAT REQUIREMENTS:
- You MUST return ONLY valid JSON in this exact format: {{"chunks": [{{"text": "...", "section": "...", "topic": "...", "chunk_summary": "...", "position_in_section": "...", "certainty_level": "...", "citation_context": "..."}}]}}
- Do NOT include any markdown formatting, code blocks, or explanatory text
- Do NOT include line breaks within string values - use spaces instead
- Do NOT include extra quotes within text fields
- Ensure all string values are properly quoted
- Ensure all arrays and objects are properly closed
- The response must be parseable by json.loads() without any preprocessing

IMPORTANT: This is a {len(pdf_binary)} byte PDF document with {total_pages} pages. You MUST process the ENTIRE document from the very first page to the very last page. Do not stop early. Process every single section including Abstract, Introduction, Methods, Results, Discussion, Conclusion, References, and any appendices.

Please process the attached PDF and return the semantic chunks as specified above.
"""
            
            # Step 5: Send binary PDF to Gemini
            logger.info("🚀 Sending binary PDF to Gemini...")
            
            # Prepare content parts
            content_parts = [full_prompt]
            content_parts.append({
                "mime_type": "application/pdf",
                "data": pdf_binary
            })
            
            # Send to Gemini
            response = self.model.generate_content(content_parts)
            
            if response.text:
                response_length = len(response.text)
                logger.info(f"✅ Received structured response from Gemini ({response_length} characters)")
                
                # Check for potential truncation
                if response_length < 5000:
                    logger.warning(f"⚠️  Response seems unusually short ({response_length} chars) for a full paper. This may indicate incomplete processing.")
                
                # Check if response ends abruptly (potential truncation)
                if response.text.strip().endswith('...') or not response.text.strip().endswith(']'):
                    logger.warning(f"⚠️  Response appears to be truncated. Last 100 chars: {response.text[-100:]}")
                
                return True, response.text, None
            else:
                logger.warning("⚠️  No response text from Gemini")
                return False, None, "No response text from Gemini"
                
        except Exception as e:
            error_msg = f"Error processing PDF with Gemini: {e}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
    
    def _fix_json_response(self, response_text: str) -> str:
        """Fix common JSON formatting issues in Gemini responses."""
        import re
        
        # Remove markdown code blocks
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Fix the specific issue we're seeing: extra quotes in text fields
        # Pattern: "text": " "text content" -> "text": "text content"
        cleaned = re.sub(r'"text":\s*"\s*"([^"]*)"', r'"text": "\1"', cleaned)
        
        # Fix another pattern: "text": " "text content" -> "text": "text content"
        cleaned = re.sub(r'"text":\s*"\s*([^"]*)"', r'"text": "\1"', cleaned)
        
        # Fix line breaks within text values
        # This regex finds text values and replaces newlines with spaces
        text_pattern = r'"text":\s*"([^"]*(?:\\"[^"]*)*)"'
        def fix_text_value(match):
            text_content = match.group(1)
            # Replace newlines, tabs, and extra whitespace
            fixed_text = re.sub(r'\s+', ' ', text_content).strip()
            # Remove any extra quotes that might have been added
            fixed_text = fixed_text.strip('"')
            # Escape any remaining quotes that aren't already escaped
            fixed_text = fixed_text.replace('"', '\\"')
            return f'"text": "{fixed_text}"'
        
        cleaned = re.sub(text_pattern, fix_text_value, cleaned)
        
        # Fix line breaks within other string values
        string_pattern = r'"([^"]*)"'
        def fix_string_value(match):
            string_content = match.group(1)
            # Only fix if it's not a text field (which we already fixed)
            if '"text":' in match.group(0):
                return match.group(0)  # Don't modify text fields again
            # Replace newlines and extra whitespace
            fixed_string = re.sub(r'\s+', ' ', string_content).strip()
            # Remove any extra quotes
            fixed_string = fixed_string.strip('"')
            # Escape any remaining quotes that aren't already escaped
            fixed_string = fixed_string.replace('"', '\\"')
            return f'"{fixed_string}"'
        
        cleaned = re.sub(string_pattern, fix_string_value, cleaned)
        
        # Remove any trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix missing quotes around field names
        cleaned = re.sub(r'(\s+)(\w+):\s*"', r'\1"\2": "', cleaned)
        
        # Fix missing quotes around string values
        cleaned = re.sub(r':\s*([^"][^,}]*[^"\s,}])\s*([,}])', r': "\1"\2', cleaned)
        
        # Fix unterminated strings by finding the last complete object
        # Look for complete JSON objects and truncate if necessary
        try:
            # Try to find the last complete object
            last_complete = cleaned.rfind('},')
            if last_complete != -1:
                # Find the closing bracket for the array
                closing_bracket = cleaned.find(']', last_complete)
                if closing_bracket != -1:
                    cleaned = cleaned[:closing_bracket + 1]
                else:
                    # If no closing bracket, just take up to the last complete object
                    cleaned = cleaned[:last_complete + 1] + ']'
        except:
            pass
        
        # Final cleanup - remove any double quotes that might have been created
        cleaned = re.sub(r'""', r'"', cleaned)
        
        return cleaned
    
    def validate_and_store_chunks(self, gemini_response: str, pdf_metadata: Dict) -> Tuple[bool, int, Optional[str], List[SemanticChunk]]:
        """Validate Gemini response and store chunks (database storage optional)."""
        try:
            # Log the raw response for debugging
            logger.info(f"🔍 Raw Gemini response length: {len(gemini_response)} characters")
            logger.info(f"🔍 First 500 chars: {gemini_response[:500]}")
            logger.info(f"🔍 Last 500 chars: {gemini_response[-500:]}")
            
            # Try multiple approaches to parse the JSON
            chunks_data = None
            
            # Approach 1: Try direct parsing
            try:
                parsed_response = json.loads(gemini_response)
                logger.info("✅ Successfully parsed JSON response directly")
                
                # Handle structured output format (chunks nested under "chunks" key)
                if isinstance(parsed_response, dict) and "chunks" in parsed_response:
                    chunks_data = parsed_response["chunks"]
                    logger.info(f"✅ Found {len(chunks_data)} chunks in structured response")
                elif isinstance(parsed_response, list):
                    chunks_data = parsed_response
                    logger.info(f"✅ Found {len(chunks_data)} chunks in array response")
                else:
                    raise ValueError("Invalid response format - expected 'chunks' key or array")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Direct JSON parsing failed: {e}")
                
                # Approach 2: Try with basic cleaning
                try:
                    cleaned_response = self._fix_json_response(gemini_response)
                    parsed_response = json.loads(cleaned_response)
                    logger.info("✅ Successfully parsed JSON response after cleaning")
                    
                    # Handle structured output format
                    if isinstance(parsed_response, dict) and "chunks" in parsed_response:
                        chunks_data = parsed_response["chunks"]
                        logger.info(f"✅ Found {len(chunks_data)} chunks in structured response")
                    elif isinstance(parsed_response, list):
                        chunks_data = parsed_response
                        logger.info(f"✅ Found {len(chunks_data)} chunks in array response")
                    else:
                        raise ValueError("Invalid response format - expected 'chunks' key or array")
                        
                except json.JSONDecodeError as e2:
                    logger.warning(f"⚠️  Cleaned JSON parsing failed: {e2}")
                    
                    # Approach 3: Try to extract JSON from markdown blocks
                    try:
                        # Look for JSON array between ```json and ``` markers
                        import re
                        json_match = re.search(r'```json\s*(\[.*?\])\s*```', gemini_response, re.DOTALL)
                        if json_match:
                            json_content = json_match.group(1)
                            chunks_data = json.loads(json_content)
                            logger.info("✅ Successfully parsed JSON from markdown blocks")
                        else:
                            # Look for any JSON array in the response
                            json_match = re.search(r'(\[.*\])', gemini_response, re.DOTALL)
                            if json_match:
                                json_content = json_match.group(1)
                                chunks_data = json.loads(json_content)
                                logger.info("✅ Successfully parsed JSON array from response")
                            else:
                                raise ValueError("No valid JSON array found in response")
                    except Exception as e3:
                        logger.error(f"❌ All JSON parsing approaches failed: {e3}")
                        raise ValueError(f"Failed to parse JSON response: {e}")
            
            # Validate the response structure
            if not isinstance(chunks_data, list):
                raise ValueError("Response must be a JSON array")
            
            # Validate each chunk using Pydantic
            chunks = []
            for i, chunk_data in enumerate(chunks_data):
                # Debug: Log the chunk data
                logger.info(f"🔍 Processing chunk {i}: {chunk_data}")
                try:
                    chunk = SemanticChunk(**chunk_data)
                    chunks.append(chunk)
                except Exception as e:
                    logger.error(f"❌ Error validating chunk {i}: {e}")
                    logger.error(f"🔍 Chunk data: {chunk_data}")
                    raise
            
            # Check if we have database IDs for storage
            if pdf_metadata.get('id') is not None:
                logger.info("💾 Storing chunks in database...")
                # Store chunks in database
                stored_count = 0
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            # Calculate word and character counts
                            word_count = len(chunk.text.split())
                            character_count = len(chunk.text)
                            
                            # Debug: Log the metadata values
                            logger.info(f"🔍 PDF metadata: id={pdf_metadata['id']}, metadata_extraction_id={pdf_metadata['metadata_extraction_id']}")
                            
                            # Insert into database directly (skip create_enriched_chunk)
                            cursor.execute("""
                                INSERT INTO semantic_chunks (
                                    sanitized_file_id, chunk_index, text, section, topic, 
                                    chunk_summary, position_in_section, certainty_level, citation_context,
                                    processing_status, gemini_response_json, processed_at,
                                    document_type, author, authors, year, source_title, journal, doi,
                                    publication_date, confidence_score, extraction_method,
                                    word_count, character_count, metadata_extraction_id
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                pdf_metadata['id'],
                                i,
                                chunk.text,
                                chunk.section,
                                chunk.topic,
                                chunk.chunk_summary,
                                chunk.position_in_section,
                                chunk.certainty_level,
                                chunk.citation_context,
                                'completed',
                                gemini_response,
                                datetime.now().isoformat(),
                                pdf_metadata['document_type'],
                                pdf_metadata['authors'], # Using authors for both author and authors for now
                                pdf_metadata['authors'],
                                str(pdf_metadata['publication_date']) if pdf_metadata['publication_date'] else 'unknown', # Convert to string, use 'unknown' if None
                                pdf_metadata['journal'],
                                pdf_metadata['journal'],
                                pdf_metadata['doi'],
                                str(pdf_metadata['publication_date']) if pdf_metadata['publication_date'] else 'unknown', # Convert to string for publication_date
                                pdf_metadata['confidence_score'],
                                pdf_metadata['extraction_method'],
                                word_count,
                                character_count,
                                pdf_metadata['metadata_extraction_id']
                            ))
                            
                            stored_count += 1
                            logger.info(f"✅ Successfully stored chunk {i} in database")
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing chunk {i}: {e}")
                            logger.error(f"🔍 Chunk data: {chunk}")
                            logger.error(f"🔍 PDF metadata: {pdf_metadata}")
                            raise
                    
                    conn.commit()
                logger.info(f"💾 Successfully stored {stored_count} chunks in database")
            else:
                logger.info("📄 Skipping database storage (no database ID available)")
                stored_count = len(chunks)
                logger.info(f"✅ Validated {stored_count} chunks (file-based processing)")
            
            # Save raw Gemini response to JSON file for backup
            self.save_gemini_response_json(pdf_metadata, gemini_response)
            
            logger.info(f"✅ Successfully processed {stored_count} chunks")
            return True, stored_count, None, chunks
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response: {e}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"🔍 JSON Error details: {e}")
            logger.error(f"🔍 Response preview: {gemini_response[:1000]}...")
            return False, 0, error_msg, []
        except Exception as e:
            error_msg = f"Error validating/storing chunks: {e}"
            logger.error(f"❌ {error_msg}")
            return False, 0, error_msg, []
    
    def move_processed_pdf(self, pdf_metadata: Dict) -> bool:
        """Move processed PDF to the chunked directory with comprehensive error handling."""
        try:
            source_file = self.ingested_dir / pdf_metadata['sanitized_filename']
            dest_file = self.chunked_dir / pdf_metadata['sanitized_filename']
            
            # Check if source file exists
            if not source_file.exists():
                logger.error(f"❌ Source file not found: {source_file}")
                return False
            
            # Check if destination already exists (shouldn't happen, but safety check)
            if dest_file.exists():
                logger.warning(f"⚠️  Destination file already exists: {dest_file}")
                # Create backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = dest_file.parent / f"{dest_file.stem}_{timestamp}{dest_file.suffix}"
                shutil.move(str(dest_file), str(backup_file))
                logger.info(f"📁 Created backup: {backup_file}")
            
            # Ensure destination directory exists
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy first, then delete (safer than move)
            shutil.copy2(str(source_file), str(dest_file))
            
            # Verify copy was successful
            if not dest_file.exists():
                logger.error(f"❌ Copy failed - destination file not found: {dest_file}")
                return False
            
            # Verify file sizes match
            if source_file.stat().st_size != dest_file.stat().st_size:
                logger.error(f"❌ Copy failed - file sizes don't match: {source_file.stat().st_size} vs {dest_file.stat().st_size}")
                dest_file.unlink()  # Remove the incomplete copy
                return False
            
            # Now delete the source file
            source_file.unlink()
            
            logger.info(f"📁 Successfully moved processed PDF: {pdf_metadata['original_filename']} -> {dest_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error moving processed PDF: {e}")
            # Try to clean up any partial files
            try:
                if dest_file.exists():
                    dest_file.unlink()
                    logger.info(f"🧹 Cleaned up partial file: {dest_file}")
            except Exception as cleanup_error:
                logger.error(f"❌ Failed to clean up partial file: {cleanup_error}")
            return False
    
    def verify_pdf_integrity(self, pdf_metadata: Dict) -> bool:
        """Verify that PDF file is accessible and not corrupted."""
        try:
            pdf_path = self.ingested_dir / pdf_metadata['sanitized_filename']
            
            if not pdf_path.exists():
                logger.error(f"❌ PDF file not found: {pdf_path}")
                return False
            
            # Check file size
            file_size = pdf_path.stat().st_size
            if file_size == 0:
                logger.error(f"❌ PDF file is empty: {pdf_path}")
                return False
            
            # Try to open and read first few bytes to verify it's a PDF
            try:
                with open(pdf_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        logger.error(f"❌ File is not a valid PDF: {pdf_path}")
                        return False
            except Exception as e:
                logger.error(f"❌ Cannot read PDF file: {pdf_path} - {e}")
                return False
            
            logger.info(f"✅ PDF integrity verified: {pdf_metadata['original_filename']} ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying PDF integrity: {e}")
            return False
    
    def save_chunks_json(self, pdf_metadata: Dict, chunks_response: SemanticChunksResponse, gemini_response: str):
        """Save chunks as JSON backup file."""
        try:
            # Create enriched chunks with metadata
            enriched_chunks = []
            for i, chunk in enumerate(chunks_response.chunks):
                enriched_chunk = create_enriched_chunk(
                    chunk=chunk,
                    chunk_index=i,
                    filename=pdf_metadata['sanitized_filename'],
                    original_filename=pdf_metadata['original_filename'],
                    author=pdf_metadata['authors'],
                    year=pdf_metadata['publication_date'][:4] if pdf_metadata['publication_date'] else None,
                    source_title=pdf_metadata['journal'],
                    doi=pdf_metadata['doi']
                )
                
                # Create a simplified chunk structure with PDF filename included
                chunk_data = {
                    'text': chunk.text,
                    'section': chunk.section,
                    'topic': chunk.topic,
                    'chunk_summary': chunk.chunk_summary,
                    'position_in_section': chunk.position_in_section,
                    'certainty_level': chunk.certainty_level,
                    'citation_context': chunk.citation_context,
                    'pdf_filename': pdf_metadata['sanitized_filename'],  # Add PDF filename to each chunk
                    'original_filename': pdf_metadata['original_filename'],
                    'authors': pdf_metadata['authors'],
                    'year': pdf_metadata['publication_date'][:4] if pdf_metadata['publication_date'] else None,
                    'journal': pdf_metadata['journal'],
                    'doi': pdf_metadata['doi']
                }
                enriched_chunks.append(chunk_data)
            
            # Create output data
            output_data = {
                'pdf_metadata': pdf_metadata,
                'chunks': enriched_chunks,
                'gemini_response': gemini_response,
                'processed_at': datetime.now().isoformat()
            }
            
            # Save to JSON file
            json_dir = Path("data/semantically-chunked/chunks")
            json_dir.mkdir(parents=True, exist_ok=True)
            
            json_filename = f"{pdf_metadata['sanitized_filename'].replace('.pdf', '')}_chunks.json"
            json_file = json_dir / json_filename
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved chunks JSON: {json_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving chunks JSON: {e}")
    
    def save_gemini_response_json(self, pdf_metadata: Dict, gemini_response: str):
        """Save the raw Gemini JSON response to a JSON file in the logs directory."""
        try:
            # Check if we have a valid response
            if not gemini_response or not gemini_response.strip():
                logger.warning("⚠️  No Gemini response to save")
                return
            
            # Create a logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Create a filename based on the sanitized filename and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{pdf_metadata['sanitized_filename'].replace('.pdf', '')}_{timestamp}_gemini_response.json"
            response_file = logs_dir / filename

            # Try to parse and save the JSON response
            try:
                parsed_response = json.loads(gemini_response)
                with open(response_file, 'w', encoding='utf-8') as f:
                    json.dump(parsed_response, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Saved raw Gemini response to: {response_file}")
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  Invalid JSON in Gemini response: {e}")
                # Save the raw text as a fallback
                with open(response_file, 'w', encoding='utf-8') as f:
                    f.write(gemini_response)
                logger.info(f"💾 Saved raw text response to: {response_file}")
                
        except Exception as e:
            logger.error(f"❌ Error saving raw Gemini response JSON: {e}")
    
    def process_pdf(self, pdf_metadata: Dict) -> Dict:
        """Process a single PDF through the complete chunking pipeline with comprehensive safeguards."""
        result = {
            'success': False,
            'pdf_filename': pdf_metadata['original_filename'],
            'chunks_count': 0,
            'error_message': None,
            'moved_to_chunked': False,
            'json_created': False,
            'rollback_performed': False
        }
        
        # Track the original PDF location for potential rollback
        original_pdf_path = self.ingested_dir / pdf_metadata['sanitized_filename']
        chunked_pdf_path = self.chunked_dir / pdf_metadata['sanitized_filename']
        
        try:
            # Step 1: Verify PDF integrity before processing
            if not self.verify_pdf_integrity(pdf_metadata):
                result['error_message'] = "PDF integrity check failed"
                return result
            
            # Step 2: Process with Gemini
            gemini_success, gemini_response, gemini_error = self.process_pdf_with_gemini(
                self.ingested_dir / pdf_metadata['sanitized_filename'], 
                pdf_metadata
            )
            
            if not gemini_success:
                result['error_message'] = gemini_error
                return result
            
            # Step 3: Validate and store chunks
            validation_success, chunks_count, validation_error, parsed_chunks = self.validate_and_store_chunks(gemini_response, pdf_metadata)
            
            if not validation_success:
                result['error_message'] = validation_error
                return result
            
            # Step 4: Move processed PDF FIRST (before creating JSON)
            logger.info(f"📁 Moving PDF to chunked directory: {pdf_metadata['original_filename']}")
            move_success = self.move_processed_pdf(pdf_metadata)
            
            if not move_success:
                result['error_message'] = "Failed to move PDF to chunked directory"
                return result
            
            result['moved_to_chunked'] = True
            logger.info(f"✅ PDF successfully moved to chunked directory")
            
            # Step 5: Verify PDF was moved successfully
            if not chunked_pdf_path.exists():
                result['error_message'] = f"PDF not found in chunked directory after move: {chunked_pdf_path}"
                return result
            
            # Step 6: Create JSON file AFTER PDF is successfully moved
            logger.info(f"💾 Creating JSON backup for: {pdf_metadata['original_filename']}")
            try:
                # Use the already parsed chunks instead of trying to parse the raw response again
                chunks_response = SemanticChunksResponse(chunks=parsed_chunks)
                self.save_chunks_json(pdf_metadata, chunks_response, gemini_response)
                result['json_created'] = True
                logger.info(f"✅ JSON backup created successfully")
            except Exception as e:
                error_msg = f"Failed to save JSON backup: {e}"
                logger.error(f"❌ {error_msg}")
                
                # ROLLBACK: Move PDF back to ingestion directory since JSON creation failed
                logger.warning(f"🔄 Rolling back PDF move due to JSON creation failure")
                try:
                    if chunked_pdf_path.exists():
                        shutil.move(str(chunked_pdf_path), str(original_pdf_path))
                        result['rollback_performed'] = True
                        logger.info(f"✅ Successfully rolled back PDF to ingestion directory")
                    else:
                        logger.error(f"❌ Cannot rollback - chunked PDF not found: {chunked_pdf_path}")
                except Exception as rollback_error:
                    logger.error(f"❌ Failed to rollback PDF move: {rollback_error}")
                    # This is a critical error - we have a PDF in chunked directory but no JSON
                    result['error_message'] = f"CRITICAL: JSON creation failed AND rollback failed. PDF orphaned: {rollback_error}"
                    return result
                
                result['error_message'] = error_msg
                return result
            
            # Step 7: Final verification - ensure both JSON and PDF exist
            json_filename = f"{pdf_metadata['sanitized_filename'].replace('.pdf', '')}_chunks.json"
            json_path = Path("data/semantically-chunked/chunks") / json_filename
            
            if not json_path.exists():
                error_msg = f"JSON file not found after processing: {json_path}"
                logger.error(f"❌ {error_msg}")
                
                # ROLLBACK: Move PDF back to ingestion directory since JSON file is missing
                logger.warning(f"🔄 Rolling back PDF move due to missing JSON file")
                try:
                    if chunked_pdf_path.exists():
                        shutil.move(str(chunked_pdf_path), str(original_pdf_path))
                        result['rollback_performed'] = True
                        logger.info(f"✅ Successfully rolled back PDF to ingestion directory")
                    else:
                        logger.error(f"❌ Cannot rollback - chunked PDF not found: {chunked_pdf_path}")
                except Exception as rollback_error:
                    logger.error(f"❌ Failed to rollback PDF move: {rollback_error}")
                    result['error_message'] = f"CRITICAL: Missing JSON file AND rollback failed. PDF orphaned: {rollback_error}"
                    return result
                
                result['error_message'] = error_msg
                return result
            
            # Success!
            result['success'] = True
            result['chunks_count'] = chunks_count
            
            logger.info(f"✅ Successfully processed PDF: {pdf_metadata['original_filename']} ({chunks_count} chunks)")
            logger.info(f"📊 Final status - Moved: {result['moved_to_chunked']}, JSON: {result['json_created']}, Rollback: {result['rollback_performed']}")
            
        except Exception as e:
            result['error_message'] = f"Unexpected error: {e}"
            logger.error(f"❌ Error processing PDF {pdf_metadata['original_filename']}: {e}")
            
            # If PDF was moved but we hit an unexpected error, try to rollback
            if result['moved_to_chunked'] and not result['json_created']:
                logger.warning(f"🔄 Rolling back PDF move due to unexpected error")
                try:
                    if chunked_pdf_path.exists():
                        shutil.move(str(chunked_pdf_path), str(original_pdf_path))
                        result['rollback_performed'] = True
                        logger.info(f"✅ Successfully rolled back PDF to ingestion directory after unexpected error")
                    else:
                        logger.error(f"❌ Cannot rollback - chunked PDF not found: {chunked_pdf_path}")
                except Exception as rollback_error:
                    logger.error(f"❌ Failed to rollback PDF move after unexpected error: {rollback_error}")
        
        return result
    
    def process_pdf_parallel(self, pdf_metadata: Dict) -> Dict:
        """Process a single PDF in parallel (worker function)."""
        try:
            # Create a new chunker instance for this worker process
            worker_chunker = SemanticPDFChunker(
                self.db_path, 
                self.ingested_dir, 
                self.chunked_dir, 
                self.prompt_file
            )
            
            # Process the PDF
            result = worker_chunker.process_pdf(pdf_metadata)
            
            # Log the result
            if result['success']:
                logger.info(f"✅ Worker completed: {pdf_metadata['original_filename']} ({result['chunks_count']} chunks)")
            else:
                logger.error(f"❌ Worker failed: {pdf_metadata['original_filename']} - {result['error_message']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Worker error processing {pdf_metadata['original_filename']}: {e}")
            return {
                'success': False,
                'pdf_filename': pdf_metadata['original_filename'],
                'chunks_count': 0,
                'error_message': f"Worker error: {e}",
                'moved_to_chunked': False,
                'json_created': False,
                'rollback_performed': False
            }

    def run_chunking(self, max_files: int = 5, parallel_workers: int = 1) -> Dict:
        """Run the semantic chunking process with comprehensive safeguards."""
        logger.info("🚀 Starting semantic PDF chunking...")
        
        # Step 1: Get PDFs to process
        pdfs = self.get_pdfs_to_process(max_files)
        logger.info(f"📊 Found {len(pdfs)} PDFs to process")
        
        if not pdfs:
            logger.info("✅ No PDFs to process - all done!")
            return {'processed': 0, 'successful': 0, 'failed': 0, 'total_chunks': 0, 'rollbacks': 0}
        
        # Step 2: Process PDFs (sequentially or in parallel)
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        rollbacks = 0
        
        if parallel_workers > 1 and len(pdfs) > 1:
            logger.info(f"🔄 Processing {len(pdfs)} PDFs with {parallel_workers} parallel workers...")
            
            # Use multiprocessing for parallel processing
            with multiprocessing.Pool(processes=parallel_workers) as pool:
                # Process PDFs in parallel
                results = pool.map(self.process_pdf_parallel, pdfs)
                
                # Collect results
                for result in results:
                    if result['success']:
                        successful += 1
                        total_chunks += result['chunks_count']
                        logger.info(f"✅ Success: {result['chunks_count']} chunks")
                    else:
                        failed += 1
                        if result['rollback_performed']:
                            rollbacks += 1
                            logger.error(f"❌ Failed with rollback: {result['error_message']}")
                        else:
                            logger.error(f"❌ Failed: {result['error_message']}")
        else:
            logger.info(f"🔄 Processing {len(pdfs)} PDFs sequentially...")
            
            # Sequential processing (original method)
            for pdf_metadata in pdfs:
                logger.info(f"🔄 Processing: {pdf_metadata['original_filename']}")
                
                result = self.process_pdf(pdf_metadata)
                results.append(result)
                
                if result['success']:
                    successful += 1
                    total_chunks += result['chunks_count']
                    logger.info(f"✅ Success: {result['chunks_count']} chunks")
                else:
                    failed += 1
                    if result['rollback_performed']:
                        rollbacks += 1
                        logger.error(f"❌ Failed with rollback: {result['error_message']}")
                    else:
                        logger.error(f"❌ Failed: {result['error_message']}")
        
        # Step 3: Final verification
        if successful > 0:
            logger.info("🔍 Verifying file integrity...")
            json_count = len(list(Path("data/semantically-chunked/chunks").glob("*_chunks.json")))
            pdf_count = len(list(Path("data/semantically-chunked/pdfs").glob("*.pdf")))
            logger.info(f"   JSON files: {json_count}")
            logger.info(f"   PDF files: {pdf_count}")
            
            if json_count != pdf_count:
                logger.warning(f"⚠️  Mismatch detected: {json_count} JSON files vs {pdf_count} PDF files")
            else:
                logger.info("✅ File counts match - no orphaned files")
        
        # Summary
        summary = {
            'processed': len(pdfs),
            'successful': successful,
            'failed': failed,
            'total_chunks': total_chunks,
            'rollbacks': rollbacks,
            'results': results
        }
        
        logger.info("📋 Chunking Summary:")
        logger.info(f"   PDFs processed: {summary['processed']}")
        logger.info(f"   Successful: {summary['successful']}")
        logger.info(f"   Failed: {summary['failed']}")
        logger.info(f"   Rollbacks performed: {summary['rollbacks']}")
        logger.info(f"   Total chunks: {summary['total_chunks']}")
        
        return summary

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Semantic PDF chunking using Gemini Pro"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/sqlite/papers.db"),
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--ingested-dir",
        type=Path,
        default=Path("data/ingested/pdfs"),
        help="Directory containing PDFs to chunk"
    )
    parser.add_argument(
        "--chunked-dir",
        type=Path,
        default=Path("data/semantically-chunked/pdfs"),
        help="Directory to move processed PDFs to"
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("data/chunking_prompt.txt"),
        help="Path to the chunking prompt file"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of parallel workers (1 = sequential processing)"
    )
    
    args = parser.parse_args()
    
    try:
        chunker = SemanticPDFChunker(args.db_path, args.ingested_dir, args.chunked_dir, args.prompt_file)
        summary = chunker.run_chunking(max_files=args.max_files, parallel_workers=args.parallel_workers)
        
        print("✅ Semantic chunking completed!")
        print(f"📊 Processed: {summary['processed']} PDFs")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📝 Total chunks: {summary['total_chunks']}")
        
    except Exception as e:
        logger.error(f"❌ Error during chunking: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 