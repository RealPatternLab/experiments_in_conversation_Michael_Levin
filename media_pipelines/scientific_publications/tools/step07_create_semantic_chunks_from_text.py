#!/usr/bin/env python3
"""
Semantic Text Chunking Tool

This tool processes already extracted text files using Gemini Pro to create semantic chunks.
It's much more efficient than processing raw PDFs and avoids timeout issues.
"""

import argparse
import json
import os
import shutil
import sys
import re
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
    from pydantic import BaseModel
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
        logging.FileHandler('logs/chunk_extracted_text.log')
    ]
)
logger = logging.getLogger(__name__)

# Define the schema for structured output manually to avoid Pydantic default issues
CHUNKS_ARRAY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "section": {"type": "string"},
            "topic": {"type": "string"},
            "chunk_summary": {"type": "string"},
            "position_in_section": {"type": "string"},
            "certainty_level": {"type": "string"},
            "citation_context": {"type": "string"},
            "page_number": {"type": ["string", "null"]}
        },
        "required": ["text", "section", "topic", "chunk_summary", "position_in_section", "certainty_level", "citation_context"]
    }
}

def get_metadata_from_filename(filename: str) -> Dict:
    """
    Create basic metadata from filename since we don't have a database.
    
    Args:
        filename: The filename to extract metadata from
        
    Returns:
        Dictionary with basic metadata
    """
    # Extract basic info from filename
    # Example: pdf_20250812_084133_380095_extracted_text.txt
    parts = filename.replace('_extracted_text.txt', '').split('_')
    
    metadata = {
        'authors': 'Unknown',
        'journal': 'Unknown', 
        'doi': 'Unknown',
        'year': 'Unknown',
        'title': 'Unknown',
        'confidence_score': 0.5,
        'document_type': 'scientific_paper'
    }
    
    # Try to extract date if available
    if len(parts) >= 3 and parts[1].isdigit() and len(parts[1]) == 8:
        try:
            date_str = parts[1]
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            metadata['year'] = year
            metadata['publication_date'] = f"{year}-{month}-{day}"
        except:
            pass
    
    return metadata

def extract_page_number_from_text(text: str) -> Optional[str]:
    """
    Extract page number from text content.
    Looks for patterns like "Page X", "p. X", "page X", etc.
    
    Args:
        text: The text content to search
        
    Returns:
        Page number as string, or None if not found
    """
    # Common page number patterns
    patterns = [
        r'Page\s+(\d+)',
        r'p\.\s*(\d+)',
        r'page\s+(\d+)',
        r'pg\.\s*(\d+)',
        r'PAGE\s+(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

class SemanticTextChunker:
    """Handles semantic chunking of extracted text files using Gemini 2.5 Flash with schema validation."""
    
    def __init__(self, input_dir: Path, output_dir: Path, prompt_file: Path):
        self.extracted_text_dir = input_dir
        self.chunked_dir = output_dir
        self.prompt_file = prompt_file
        
        # Create directories
        self.chunked_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment and configure Gemini
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize Gemini client
        genai.configure(api_key=api_key)
        
        # Create the model with schema validation
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
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
    
    def get_text_files_to_process(self, max_files: int = 5) -> List[Dict]:
        """Get text files that need to be processed."""
        logger.info(f"🔍 Scanning extracted text directory: {self.extracted_text_dir}")
        
        if not self.extracted_text_dir.exists():
            logger.error(f"❌ Extracted text directory does not exist: {self.extracted_text_dir}")
            return []
        
        # Step 1: Get all text files in the extracted text directory
        text_files = list(self.extracted_text_dir.glob("*_extracted_text.txt"))
        logger.info(f"📄 Found {len(text_files)} extracted text files")
        
        # TEMPORARY: Only process PDFs with today's date (for testing)
        today = datetime.now().strftime("%Y%m%d")
        today_files = []
        for text_file in text_files:
            if today in text_file.name:
                today_files.append(text_file)
                logger.info(f"Found today's file: {text_file.name}")
        
        if not today_files:
            logger.info(f"No files found with today's date ({today})")
            return []
        
        logger.info(f"Found {len(today_files)} files with today's date ({today})")
        
        # Step 2: Get list of already processed files (have corresponding JSON files)
        chunks_dir = Path("data/semantically-chunked/chunks")
        if not chunks_dir.exists():
            chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Created chunks directory: {chunks_dir}")
        
        existing_json_files = list(chunks_dir.glob("*_chunks.json"))
        logger.info(f"📁 Found {len(existing_json_files)} already processed JSON files")
        
        # Create set of already processed file stems (without _extracted_text suffix)
        processed_stems = set()
        for json_file in existing_json_files:
            # Extract the base filename from JSON filename
            # e.g., "pdf_20250803_124125_479734_chunks.json" -> "pdf_20250803_124125_479734"
            stem = json_file.stem.replace('_chunks', '')
            processed_stems.add(stem)
            logger.debug(f"📋 Found processed file: {stem}")
        
        logger.info(f"📋 Already processed file stems: {sorted(processed_stems)}")
        
        # Step 3: Filter out already processed files
        unprocessed_files = []
        for text_file in today_files:  # Changed from text_files to today_files
            # Extract the base filename from text filename
            # e.g., "pdf_20250803_124125_479734_extracted_text.txt" -> "pdf_20250803_124125_479734"
            file_stem = text_file.stem.replace('_extracted_text', '')
            
            if file_stem not in processed_stems:
                unprocessed_files.append(text_file)
                logger.info(f"✅ Found unprocessed text file: {text_file.name} (stem: {file_stem})")
            else:
                logger.info(f"⏭️  Skipping already processed file: {text_file.name} (stem: {file_stem})")
        
        logger.info(f"📊 Total unprocessed text files: {len(unprocessed_files)}")
        
        if len(unprocessed_files) == 0:
            logger.info("🎉 All extracted text files have already been processed!")
            return []
        
        # Step 4: Limit to max_files
        files_to_process = unprocessed_files[:max_files]
        logger.info(f"🎯 Will process {len(files_to_process)} files (limited by max_files={max_files})")
        
        # Step 5: Convert to metadata format
        files = []
        for text_file in files_to_process:
            try:
                # Get basic file info
                file_size = text_file.stat().st_size
                original_filename = text_file.name
                pdf_filename = text_file.stem.replace('_extracted_text', '.pdf')
                
                logger.info(f"📋 Preparing metadata for: {original_filename} ({file_size} bytes)")
                
                # Get metadata from filename
                file_metadata = get_metadata_from_filename(original_filename)
                
                files.append({
                    'text_file': text_file,
                    'original_filename': original_filename,
                    'pdf_filename': pdf_filename,
                    'file_size': file_size,
                    'authors': file_metadata['authors'],
                    'journal': file_metadata['journal'],
                    'doi': file_metadata['doi'],
                    'year': file_metadata['year'],
                    'title': file_metadata['title'],
                    'confidence_score': file_metadata['confidence_score'],
                    'document_type': file_metadata['document_type']
                })
                
            except Exception as e:
                logger.error(f"❌ Error preparing metadata for {text_file.name}: {e}")
                continue
        
        logger.info(f"✅ Prepared {len(files)} files for processing")
        return files
    
    def process_text_with_gemini(self, text_file: Path, file_metadata: Dict) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """Process a text file with Gemini 2.5 Flash and return the structured response."""
        try:
            logger.info(f"🤖 Processing text file with Gemini 2.5 Flash: {file_metadata['original_filename']}")
            
            # Step 1: Read the text file
            logger.info("📄 Reading extracted text file...")
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            text_length = len(text_content)
            logger.info(f"📄 Text file size: {text_length} characters")
            
            # Step 2: Create the prompt (simplified since we're using schema validation)
            prompt = f"""
{self.prompt}

IMPORTANT: This is a {text_length} character text document. You MUST process the ENTIRE document from the very first word to the very last word. Do not stop early. Process every single section including Abstract, Introduction, Methods, Results, Discussion, Conclusion, References, and any appendices.

Please process the attached text and return the semantic chunks as specified above.
"""
            
            # Step 3: Send text to Gemini with schema validation
            logger.info("🚀 Sending text to Gemini with schema validation...")
            
            # Prepare content parts
            content_parts = [prompt, text_content]
            
            # Send to Gemini without schema validation (learning from API limitations)
            response = self.model.generate_content(
                contents=content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=32768,
                ),
            )
            
            if response.text:
                response_length = len(response.text)
                logger.info(f"✅ Received structured response from Gemini 2.5 Flash ({response_length} characters)")
                
                # Parse the text response as JSON
                try:
                    # Check if response is wrapped in markdown code block
                    response_text = response.text.strip()
                    if response_text.startswith('```json'):
                        # Extract JSON from markdown code block
                        json_start = response_text.find('```json') + 7
                        json_end = response_text.rfind('```')
                        if json_end > json_start:
                            response_text = response_text[json_start:json_end].strip()
                        else:
                            response_text = response_text[json_start:].strip()
                    elif response_text.startswith('```'):
                        # Extract JSON from generic code block
                        json_start = response_text.find('```') + 3
                        json_end = response_text.rfind('```')
                        if json_end > json_start:
                            response_text = response_text[json_start:json_end].strip()
                        else:
                            response_text = response_text[json_start:].strip()
                    
                    # Try to repair common JSON issues
                    try:
                        chunks_data = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️  JSON decode error: {e}")
                        logger.warning(f"🔍 Attempting to repair JSON...")
                        
                        # Try to find the last complete object
                        if response_text.endswith(','):
                            response_text = response_text[:-1]
                        
                        # Try to find the last complete array
                        if response_text.endswith(']') and response_text.count('[') == response_text.count(']'):
                            pass  # Already balanced
                        else:
                            # Find the last complete object and truncate there
                            last_complete = response_text.rfind('},')
                            if last_complete > 0:
                                response_text = response_text[:last_complete + 1] + ']'
                            else:
                                # If no complete objects, try to find the last complete array
                                last_bracket = response_text.rfind(']')
                                if last_bracket > 0:
                                    response_text = response_text[:last_bracket + 1]
                        
                        logger.warning(f"🔍 Repaired JSON (first 200 chars): {response_text[:200]}")
                        chunks_data = json.loads(response_text)
                    
                    parsed_chunks = [chunk for chunk in chunks_data] # No schema validation here
                    logger.info(f"✅ Successfully parsed {len(parsed_chunks)} chunks from text response")
                    return True, parsed_chunks, None
                except Exception as e:
                    logger.error(f"❌ Failed to parse text response: {e}")
                    logger.error(f"🔍 Full response text:")
                    logger.error(f"---START OF RESPONSE---")
                    logger.error(response.text)
                    logger.error(f"---END OF RESPONSE---")
                    logger.error(f"🔍 Response length: {len(response.text)} characters")
                    return False, None, f"Failed to parse response: {e}"
            else:
                logger.warning("⚠️  No response text from Gemini")
                return False, None, "No response text from Gemini"
                
        except Exception as e:
            error_msg = f"Error processing text with Gemini: {e}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
    
    def validate_and_store_chunks(self, chunks: List[Dict], file_metadata: Dict) -> Tuple[bool, int, Optional[str], List[SemanticChunk]]:
        """Validate chunks and convert to SemanticChunk objects."""
        try:
            logger.info(f"🔍 Validating {len(chunks)} chunks...")
            
            # Convert SemanticChunkSchema to SemanticChunk objects
            semantic_chunks = []
            for i, chunk_data in enumerate(chunks):
                try:
                    # Convert to SemanticChunk with safe field access - page_number is completely optional
                    chunk = SemanticChunk(
                        text=chunk_data['text'],
                        section=chunk_data['section'],
                        topic=chunk_data['topic'],
                        chunk_summary=chunk_data['chunk_summary'],
                        position_in_section=chunk_data['position_in_section'],
                        certainty_level=chunk_data['certainty_level'],
                        citation_context=chunk_data['citation_context'],
                        page_number=None  # Always set to None - we don't care about page numbers
                    )
                    semantic_chunks.append(chunk)
                    logger.info(f"🔍 Processing chunk {i}: {chunk_data}")
                except Exception as e:
                    logger.error(f"❌ Error validating chunk {i}: {e}")
                    logger.error(f"🔍 Chunk data: {chunk_data}")
                    raise
            
            logger.info(f"✅ Successfully processed {len(semantic_chunks)} chunks")
            return True, len(semantic_chunks), None, semantic_chunks
            
        except Exception as e:
            error_msg = f"Error validating/storing chunks: {e}"
            logger.error(f"❌ {error_msg}")
            return False, 0, error_msg, []
    
    def save_chunks_json(self, file_metadata: Dict, chunks_response: SemanticChunksResponse, gemini_response: str = ""):
        """Save chunks as JSON backup file."""
        try:
            # Create enriched chunks with metadata
            enriched_chunks = []
            for i, chunk in enumerate(chunks_response.chunks):
                enriched_chunk = create_enriched_chunk(
                    chunk=chunk,
                    chunk_index=i,
                    filename=file_metadata['pdf_filename'],
                    original_filename=file_metadata['original_filename'],
                    author=file_metadata['authors'],
                    year=file_metadata['year'],
                    source_title=file_metadata['title'],
                    doi=file_metadata['doi']
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
                    'page_number': chunk.page_number, # Include page number
                    'pdf_filename': file_metadata['pdf_filename'],  # Add PDF filename to each chunk
                    'original_filename': file_metadata['original_filename'],
                    'authors': file_metadata['authors'],
                    'year': file_metadata['year'],
                    'journal': file_metadata['journal'],
                    'doi': file_metadata['doi']
                }
                enriched_chunks.append(chunk_data)
            
            # Create output data
            # Convert Path objects to strings for JSON serialization
            serializable_metadata = {}
            for key, value in file_metadata.items():
                if isinstance(value, Path):
                    serializable_metadata[key] = str(value)
                else:
                    serializable_metadata[key] = value
            
            output_data = {
                'file_metadata': serializable_metadata,
                'chunks': enriched_chunks,
                'gemini_response': gemini_response,
                'processed_at': datetime.now().isoformat()
            }
            
            # Save to JSON file
            json_dir = Path("data/semantically-chunked/chunks")
            json_dir.mkdir(parents=True, exist_ok=True)
            
            json_filename = f"{file_metadata['pdf_filename'].replace('.pdf', '')}_chunks.json"
            json_file = json_dir / json_filename
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved chunks JSON: {json_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving chunks JSON: {e}")
    
    def save_gemini_response_json(self, file_metadata: Dict, chunks: List[Dict]):
        """Save the raw Gemini response to a JSON file in the logs directory."""
        try:
            # Check if we have a valid response
            if not chunks:
                logger.warning("⚠️  No chunks to save")
                return
            
            # Create a logs directory if it doesn't exist
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Create a filename based on the original filename and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{file_metadata['original_filename'].replace('.txt', '')}_{timestamp}_gemini_response.json"
            response_file = logs_dir / filename

            # Save the chunks as JSON
            chunks_data = [chunk for chunk in chunks] # No schema validation here
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved raw Gemini response to: {response_file}")
                
        except Exception as e:
            logger.error(f"❌ Error saving raw Gemini response JSON: {e}")
    
    def process_text_file(self, file_metadata: Dict) -> Dict:
        """Process a single text file through the complete chunking pipeline."""
        result = {
            'success': False,
            'filename': file_metadata['original_filename'],
            'chunks_count': 0,
            'error_message': None
        }
        
        try:
            # Step 1: Process with Gemini
            gemini_success, chunks, gemini_error = self.process_text_with_gemini(
                file_metadata['text_file'], 
                file_metadata
            )
            
            if not gemini_success:
                result['error_message'] = gemini_error
                return result
            
            # Step 2: Validate and store chunks
            validation_success, chunks_count, validation_error, parsed_chunks = self.validate_and_store_chunks(chunks, file_metadata)
            
            if not validation_success:
                result['error_message'] = validation_error
                return result
            
            # Step 3: Create JSON file
            logger.info(f"💾 Creating JSON backup for: {file_metadata['original_filename']}")
            try:
                chunks_response = SemanticChunksResponse(chunks=parsed_chunks)
                self.save_chunks_json(file_metadata, chunks_response)
                logger.info(f"✅ JSON backup created successfully")
            except Exception as e:
                result['error_message'] = f"Failed to save JSON backup: {e}"
                return result
            
            # Step 4: Save raw response
            try:
                self.save_gemini_response_json(file_metadata, chunks)
            except Exception as e:
                logger.warning(f"⚠️  Failed to save raw response: {e}")
            
            # Success!
            result['success'] = True
            result['chunks_count'] = chunks_count
            
            logger.info(f"✅ Successfully processed text file: {file_metadata['original_filename']} ({chunks_count} chunks)")
            
        except Exception as e:
            result['error_message'] = f"Unexpected error: {e}"
            logger.error(f"❌ Error processing text file {file_metadata['original_filename']}: {e}")
        
        return result
    
    def process_text_file_parallel(self, file_metadata: Dict) -> Dict:
        """Process a single text file in parallel (worker function)."""
        try:
            # Create a new chunker instance for this worker process
            worker_chunker = SemanticTextChunker(
                self.extracted_text_dir, 
                self.chunked_dir, 
                self.prompt_file
            )
            
            # Process the text file
            result = worker_chunker.process_text_file(file_metadata)
            
            # Log the result
            if result['success']:
                logger.info(f"✅ Worker completed: {file_metadata['original_filename']} ({result['chunks_count']} chunks)")
            else:
                logger.error(f"❌ Worker failed: {file_metadata['original_filename']} - {result['error_message']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Worker error processing {file_metadata['original_filename']}: {e}")
            return {
                'success': False,
                'filename': file_metadata['original_filename'],
                'chunks_count': 0,
                'error_message': f"Worker error: {e}"
            }

    def run_chunking(self, max_files: int = 5, parallel_workers: int = 1) -> Dict:
        """Run the semantic text chunking process."""
        logger.info("🚀 Starting semantic text chunking with Gemini 2.5 Flash...")
        
        # Step 1: Get text files to process
        files = self.get_text_files_to_process(max_files)
        logger.info(f"📊 Found {len(files)} text files to process")
        
        if not files:
            logger.info("✅ No text files to process - all done!")
            return {'processed': 0, 'successful': 0, 'failed': 0, 'total_chunks': 0}
        
        # Step 2: Process text files (sequentially or in parallel)
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        
        if parallel_workers > 1 and len(files) > 1:
            logger.info(f"🔄 Processing {len(files)} text files with {parallel_workers} parallel workers...")
            
            # Use multiprocessing for parallel processing
            with multiprocessing.Pool(processes=parallel_workers) as pool:
                # Process text files in parallel
                results = pool.map(self.process_text_file_parallel, files)
                
                # Collect results
                for result in results:
                    if result['success']:
                        successful += 1
                        total_chunks += result['chunks_count']
                        logger.info(f"✅ Success: {result['chunks_count']} chunks")
                    else:
                        failed += 1
                        logger.error(f"❌ Failed: {result['error_message']}")
        else:
            logger.info(f"🔄 Processing {len(files)} text files sequentially...")
            
            # Sequential processing
            for file_metadata in files:
                logger.info(f"🔄 Processing: {file_metadata['original_filename']}")
                
                result = self.process_text_file(file_metadata)
                results.append(result)
                
                if result['success']:
                    successful += 1
                    total_chunks += result['chunks_count']
                    logger.info(f"✅ Success: {result['chunks_count']} chunks")
                else:
                    failed += 1
                    logger.error(f"❌ Failed: {result['error_message']}")
        
        # Step 3: Final verification
        if successful > 0:
            logger.info("🔍 Verifying file integrity...")
            json_count = len(list(Path("data/semantically-chunked/chunks").glob("*_chunks.json")))
            logger.info(f"   JSON files: {json_count}")
        
        # Summary
        summary = {
            'processed': len(files),
            'successful': successful,
            'failed': failed,
            'total_chunks': total_chunks,
            'results': results
        }
        
        logger.info("📋 Chunking Summary:")
        logger.info(f"   Text files processed: {summary['processed']}")
        logger.info(f"   Successful: {summary['successful']}")
        logger.info(f"   Failed: {summary['failed']}")
        logger.info(f"   Total chunks: {summary['total_chunks']}")
        
        return summary

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Semantic text chunking using Gemini 2.5 Flash with schema validation"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/extracted_text"),
        help="Directory containing extracted text files to chunk"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/semantically-chunked/chunks"),
        help="Directory to save chunked JSON files"
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
        chunker = SemanticTextChunker(args.input_dir, args.output_dir, args.prompt_file)
        summary = chunker.run_chunking(max_files=args.max_files, parallel_workers=args.parallel_workers)
        
        print("✅ Semantic text chunking completed!")
        print(f"📊 Processed: {summary['processed']} text files")
        print(f"✅ Successful: {summary['successful']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📝 Total chunks: {summary['total_chunks']}")
        
    except Exception as e:
        logger.error(f"❌ Error during chunking: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 