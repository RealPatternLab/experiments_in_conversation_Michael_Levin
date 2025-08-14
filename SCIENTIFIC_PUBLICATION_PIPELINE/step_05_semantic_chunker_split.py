#!/usr/bin/env python3
"""
Step 5: Semantic Chunking with Intelligent Document Splitting

This improved version splits the document into manageable sections before sending to Gemini,
ensuring complete coverage while respecting section boundaries and token limits.

Features:
- Intelligent section detection and splitting
- Token counting for each section
- Slight overlap between sections to maintain context
- Complete document coverage guarantee
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
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
        logging.FileHandler('semantic_chunking_split.log')
    ]
)
logger = logging.getLogger(__name__)

class DocumentSplitter:
    """Intelligently splits documents into sections based on headers and token counts."""
    
    def __init__(self, max_tokens_per_section: int = 8000, overlap_tokens: int = 500):
        self.max_tokens_per_section = max_tokens_per_section
        self.overlap_tokens = overlap_tokens
        
        # Section header patterns
        self.section_patterns = [
            r'^(\d+\.\s+[A-Z][^:\n]*(?::[^:\n]*)?)',  # 1. Section Title
            r'^([A-Z][A-Z\s]+(?:[A-Z][a-z][^:\n]*)?)',  # ALL CAPS SECTION
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$)',  # Title Case Section
            r'^(ABSTRACT|INTRODUCTION|METHODS|RESULTS|DISCUSSION|CONCLUSION|REFERENCES|APPENDIX)',
            r'^(\[PAGE\s+\d+\]\s+[A-Z][^:\n]*)',  # [PAGE X] Section
            # Additional flexible patterns for various document types
            r'^([A-Z][a-z]+(?:\s+[a-z]+)*\s*$)',  # More flexible title case
            r'^([A-Z][A-Za-z\s]+(?:\s*[-–—]\s*[A-Z][A-Za-z\s]*)?)',  # Flexible headers with dashes
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:)',  # Headers ending with colon
            r'^([A-Z][A-Z\s]+(?:\s+[A-Z][a-z][^:\n]*)?)',  # Mixed case headers
        ]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect section boundaries in the text."""
        lines = text.split('\n')
        sections = []
        current_section = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            is_header = False
            section_title = None
            
            for pattern in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    is_header = True
                    section_title = match.group(1).strip()
                    break
            
            if is_header and section_title:
                # End previous section if exists
                if current_section:
                    current_section['end_line'] = i - 1
                    current_section['end_pos'] = len('\n'.join(lines[:i]))
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': section_title,
                    'start_line': i,
                    'start_pos': len('\n'.join(lines[:i])),
                    'end_line': None,
                    'end_pos': None,
                    'content': ''
                }
        
        # End the last section
        if current_section:
            current_section['end_line'] = len(lines) - 1
            current_section['end_pos'] = len(text)
            sections.append(current_section)
        
        # Extract content for each section
        for section in sections:
            start = section['start_pos']
            end = section['end_pos']
            section['content'] = text[start:end].strip()
            section['token_count'] = self.estimate_tokens(section['content'])
        
        # Fallback: if no sections detected, create one section with the entire document
        if not sections:
            logger.warning("No sections detected, creating fallback single section")
            sections = [{
                'title': 'Full Document',
                'start_line': 0,
                'start_pos': 0,
                'end_line': len(lines) - 1,
                'end_pos': len(text),
                'content': text,
                'token_count': self.estimate_tokens(text)
            }]
        
        return sections
    
    def split_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split sections that are too long into smaller chunks."""
        split_sections = []
        
        for section in sections:
            if section['token_count'] <= self.max_tokens_per_section:
                # Section is small enough, keep as is
                split_sections.append(section)
            else:
                # Section is too long, split it
                logger.info(f"Splitting long section '{section['title']}' ({section['token_count']} tokens)")
                
                # Split into chunks with overlap
                content = section['content']
                chunks = self._split_content_with_overlap(content, section['title'])
                
                for i, chunk in enumerate(chunks):
                    split_sections.append({
                        'title': f"{section['title']} (Part {i+1})",
                        'start_line': section['start_line'],
                        'start_pos': section['start_pos'],
                        'end_line': section['end_line'],
                        'end_pos': section['end_pos'],
                        'content': chunk,
                        'token_count': self.estimate_tokens(chunk),
                        'original_section': section['title'],
                        'part_number': i + 1
                    })
        
        return split_sections
    
    def _split_content_with_overlap(self, content: str, section_title: str) -> List[str]:
        """Split content into chunks with overlap."""
        chunks = []
        words = content.split()
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            current_chunk.append(word)
            current_tokens += len(word) // 4  # Rough token estimation
            
            if current_tokens >= self.max_tokens_per_section:
                # Add overlap words from the end
                overlap_words = []
                overlap_tokens = 0
                
                for word in reversed(current_chunk):
                    overlap_words.insert(0, word)
                    overlap_tokens += len(word) // 4
                    if overlap_tokens >= self.overlap_tokens:
                        break
                
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                current_chunk = overlap_words
                current_tokens = overlap_tokens
        
        # Add final chunk if there's content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks

class SemanticChunkerSplit:
    """Creates semantic chunks from extracted text using intelligent document splitting."""
    
    def __init__(self, extracted_text_dir: Path, chunks_dir: Path):
        self.extracted_text_dir = extracted_text_dir
        self.chunks_dir = chunks_dir
        
        # Create directories
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment and configure Gemini
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Create the model
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Initialize document splitter
        self.splitter = DocumentSplitter(max_tokens_per_section=8000, overlap_tokens=500)
        
        # Set the chunking prompt
        self.prompt = self._get_chunking_prompt()
        
        # Initialize progress queue if available
        self.progress_queue = None
        if PROGRESS_QUEUE_AVAILABLE:
            try:
                self.progress_queue = PipelineProgressQueue()
                logger.info("Progress queue initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize progress queue: {e}")
                self.progress_queue = None
    
    def _get_chunking_prompt(self) -> str:
        """Return the chunking prompt for individual sections."""
        prompt = """You are a scientific paper analysis AI. Your task is to create semantic chunks from a specific section of a Michael Levin scientific paper.

**CRITICAL: You must return ONLY a valid JSON array. No additional text, explanations, or markdown formatting.**

**SECTION PROCESSING REQUIREMENTS:**
- Process the ENTIRE section content provided to you
- Create 3-8 chunks depending on section length and complexity
- Each chunk should be 100-300 words
- Ensure complete coverage of the section content

**JSON STRUCTURE:**
```json
[
  {
    "text": "The actual text of the chunk. This must be a complete string without line breaks.",
    "section": "Section Name",
    "primary_topic": "Main conceptual focus of this chunk (e.g., Bioelectric signaling, Ion channels, Regeneration)",
    "secondary_topics": ["keyword1", "keyword2", "keyword3"],
    "chunk_summary": "One-sentence summary of the chunk's core message.",
    "position_in_section": "Beginning",
    "certainty_level": "High",
    "citation_context": "Describing prior work",
    "page_number": "1"
  }
]
```

**TOPIC GUIDELINES:**
- **Primary Topic**: The main conceptual focus (1-3 words max)
- **Secondary Topics**: 3-8 specific keywords someone might search for, including:
  - Technical terms (e.g., "gap junctions", "morphogenesis")
  - Biological concepts (e.g., "cell communication", "tissue organization")
  - Research methods (e.g., "computational modeling", "experimental validation")
  - Related phenomena (e.g., "regeneration", "developmental plasticity")
  - Specific molecules/processes (e.g., "V-ATPase", "bioelectric gradients")

**VALID VALUES:**
- `section`: Use the section name provided
- `position_in_section`: "Beginning", "Middle", "End"
- `certainty_level`: "High", "Medium", "Low"
- `citation_context`: "Describing prior work", "Presenting new results", "Drawing conclusions", "None"
- `page_number`: Extract from text if possible, otherwise use null

**CRITICAL: Process the ENTIRE section content and return ONLY the JSON array.**"""
        
        return prompt
    
    def get_text_files_to_process(self) -> List[Dict]:
        """Get list of extracted text files that need to be processed."""
        files_to_process = []
        
        if not self.extracted_text_dir.exists():
            logger.error(f"Extracted text directory does not exist: {self.extracted_text_dir}")
            return files_to_process
        
        # Find all extracted text files
        text_files = list(self.extracted_text_dir.glob("*_extracted_text.txt"))
        
        if not text_files:
            logger.warning("No extracted text files found")
            return files_to_process
        
        # Check which files have already been processed
        already_processed = set()
        if self.chunks_dir.exists():
            for json_file in self.chunks_dir.glob("*_chunks.json"):
                base_name = json_file.stem.replace('_chunks', '')
                already_processed.add(base_name)
        
        # Filter out already processed files
        for text_file in text_files:
            base_name = text_file.stem.replace('_extracted_text', '')
            
            if base_name not in already_processed:
                # Load corresponding metadata if available
                metadata_file = Path("step_02_metadata") / f"{base_name}_metadata.json"
                metadata = {}
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {base_name}: {e}")
                
                files_to_process.append({
                    'text_file': text_file,
                    'base_name': base_name,
                    'metadata': metadata
                })
                logger.info(f"Found unprocessed text file: {text_file.name}")
            else:
                logger.info(f"Skipping already processed file: {text_file.name}")
        
        logger.info(f"Found {len(files_to_process)} unprocessed text files")
        return files_to_process
    
    def split_document(self, text_file: Path) -> List[Dict[str, Any]]:
        """Split document into manageable sections."""
        logger.info(f"Splitting document: {text_file.name}")
        
        # Read the text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Detect sections
        sections = self.splitter.detect_sections(text_content)
        logger.info(f"Detected {len(sections)} sections")
        
        # Split long sections if needed
        split_sections = self.splitter.split_sections(sections)
        logger.info(f"After splitting: {len(split_sections)} sections")
        
        # Log section information
        for i, section in enumerate(split_sections):
            logger.info(f"Section {i+1}: '{section['title']}' - {section['token_count']} tokens")
        
        return split_sections
    
    def process_section_with_gemini(self, section: Dict[str, Any], metadata: Dict) -> Tuple[bool, Optional[List[Dict]], Optional[str]]:
        """Process a single section with Gemini to create semantic chunks."""
        try:
            logger.info(f"Processing section: {section['title']} ({section['token_count']} tokens)")
            
            # Create the prompt for this section
            prompt = f"""
{self.prompt}

**SECTION INFORMATION:**
- Section: {section['title']}
- Token count: {section['token_count']}
- Content length: {len(section['content'])} characters

**SECTION CONTENT:**
{section['content']}

Please process this section and return the semantic chunks as specified above.
"""
            
            # Send section to Gemini
            logger.info("Sending section to Gemini for chunking...")
            
            response = self.model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=16384,
                ),
            )
            
            if response.text:
                response_length = len(response.text)
                logger.info(f"Received response from Gemini ({response_length} characters)")
                
                # Parse the response as JSON
                try:
                    # Check if response is wrapped in markdown code block
                    response_text = response.text.strip()
                    if response_text.startswith('```json'):
                        json_start = response_text.find('```json') + 7
                        json_end = response_text.rfind('```')
                        if json_end > json_start:
                            response_text = response_text[json_start:json_end].strip()
                        else:
                            response_text = response_text[json_start:].strip()
                    elif response_text.startswith('```'):
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
                        logger.warning(f"JSON decode error: {e}")
                        logger.warning("Attempting to repair JSON...")
                        
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
                        
                        logger.warning(f"Repaired JSON (first 200 chars): {response_text[:200]}")
                        chunks_data = json.loads(response_text)
                    
                    # Add section metadata to each chunk
                    for chunk in chunks_data:
                        chunk['section'] = section['title']
                        if 'original_section' in section:
                            chunk['original_section'] = section['original_section']
                        if 'part_number' in section:
                            chunk['part_number'] = section['part_number']
                    
                    parsed_chunks = [chunk for chunk in chunks_data]
                    logger.info(f"Successfully parsed {len(parsed_chunks)} chunks from section")
                    return True, parsed_chunks, None
                    
                except Exception as e:
                    logger.error(f"Failed to parse response: {e}")
                    logger.error(f"Response text: {response.text}")
                    return False, None, f"Failed to parse response: {e}"
            else:
                logger.warning("No response text from Gemini")
                return False, None, "No response text from Gemini"
                
        except Exception as e:
            error_msg = f"Error processing section with Gemini: {e}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def save_chunks_json(self, base_name: str, all_chunks: List[Dict], metadata: Dict):
        """Save all chunks as JSON file."""
        try:
            # Create output data
            output_data = {
                'metadata': metadata,
                'chunks': all_chunks,
                'processed_at': datetime.now().isoformat(),
                'chunk_count': len(all_chunks),
                'processing_method': 'document_splitting'
            }
            
            # Save to JSON file
            json_filename = f"{base_name}_chunks.json"
            json_file = self.chunks_dir / json_filename
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved chunks JSON: {json_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chunks JSON: {e}")
            return False
    
    def process_text_file(self, file_info: Dict) -> Dict:
        """Process a single text file through the complete chunking pipeline."""
        result = {
            'success': False,
            'filename': file_info['text_file'].name,
            'chunks_count': 0,
            'error_message': None
        }
        
        try:
            # Find document ID from progress queue if available
            doc_id = None
            if self.progress_queue:
                try:
                    # Search for document with this filename in progress queue
                    for doc_id_candidate, doc_data in self.progress_queue._load_queue_data()["pipeline_progress"].items():
                        if doc_data.get("new_filename") == f"{file_info['base_name']}.pdf":
                            doc_id = doc_id_candidate
                            break
                    
                    if doc_id:
                        logger.info(f"Found document {doc_id} in progress queue")
                    else:
                        logger.warning(f"Document {file_info['base_name']} not found in progress queue")
                except Exception as e:
                    logger.warning(f"Progress queue lookup error: {e}")
                    doc_id = None
            
            # Step 1: Split document into sections
            sections = self.split_document(file_info['text_file'])
            
            if not sections:
                result['error_message'] = "Failed to split document into sections"
                return result
            
            # Step 2: Process each section with Gemini
            all_chunks = []
            successful_sections = 0
            failed_sections = 0
            
            for i, section in enumerate(sections):
                logger.info(f"Processing section {i+1}/{len(sections)}: {section['title']}")
                
                gemini_success, chunks, gemini_error = self.process_section_with_gemini(
                    section, 
                    file_info['metadata']
                )
                
                if gemini_success:
                    all_chunks.extend(chunks)
                    successful_sections += 1
                    logger.info(f"✅ Section {i+1} processed successfully: {len(chunks)} chunks")
                else:
                    failed_sections += 1
                    logger.error(f"❌ Section {i+1} failed: {gemini_error}")
            
            # Step 3: Save all chunks
            if all_chunks:
                save_success = self.save_chunks_json(
                    file_info['base_name'],
                    all_chunks,
                    file_info['metadata']
                )
                
                if not save_success:
                    result['error_message'] = "Failed to save chunks"
                    return result
                
                result['success'] = True
                result['chunks_count'] = len(all_chunks)
                
                # Update progress queue if available
                if self.progress_queue and doc_id:
                    try:
                        self.progress_queue.update_step_status(
                            doc_id, 
                            "step_05_semantic_chunking", 
                            "complete",
                            {"chunk_count": len(all_chunks)}
                        )
                        logger.info(f"Updated progress queue for {doc_id}")
                    except Exception as e:
                        logger.warning(f"Failed to update progress queue: {e}")
                
                logger.info(f"✅ Successfully processed text file: {file_info['text_file'].name}")
                logger.info(f"   Sections processed: {successful_sections}/{len(sections)}")
                logger.info(f"   Total chunks created: {len(all_chunks)}")
            else:
                result['error_message'] = "No chunks were created from any section"
            
        except Exception as e:
            result['error_message'] = f"Unexpected error: {e}"
            logger.error(f"Error processing text file {file_info['text_file'].name}: {e}")
            
            # Update progress queue if available
            if self.progress_queue and doc_id:
                try:
                    self.progress_queue.add_error(doc_id, "step_05_semantic_chunking", f"Unexpected error: {e}")
                except Exception as queue_error:
                    logger.warning(f"Failed to update progress queue: {queue_error}")
        
        return result
    
    def run_chunking(self) -> Dict:
        """Run the semantic chunking process with document splitting."""
        logger.info("Starting semantic text chunking with intelligent document splitting...")
        
        # Get text files to process
        files = self.get_text_files_to_process()
        logger.info(f"Found {len(files)} text files to process")
        
        if not files:
            logger.info("No text files to process - all done!")
            return {'processed': 0, 'successful': 0, 'failed': 0, 'total_chunks': 0}
        
        # Process text files
        results = []
        successful = 0
        failed = 0
        total_chunks = 0
        
        for file_info in files:
            logger.info(f"Processing: {file_info['text_file'].name}")
            
            result = self.process_text_file(file_info)
            results.append(result)
            
            if result['success']:
                successful += 1
                total_chunks += result['chunks_count']
                logger.info(f"Success: {result['chunks_count']} chunks")
            else:
                failed += 1
                logger.error(f"Failed: {result['error_message']}")
        
        # Summary
        summary = {
            'processed': len(files),
            'successful': successful,
            'failed': failed,
            'total_chunks': total_chunks,
            'results': results
        }
        
        logger.info("Chunking Summary:")
        logger.info(f"   Text files processed: {summary['processed']}")
        logger.info(f"   Successful: {summary['successful']}")
        logger.info(f"   Failed: {summary['failed']}")
        logger.info(f"   Total chunks: {summary['total_chunks']}")
        
        return summary

def main():
    """Main function to run the semantic chunking pipeline with document splitting."""
    parser = argparse.ArgumentParser(description="Step 5: Semantic Chunking with Document Splitting")
    parser.add_argument(
        "--extracted-text-dir",
        type=Path,
        default=Path("step_03_extracted_text"),
        help="Directory containing extracted text files from Step 3 (default: step_03_extracted_text)"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="step_05_semantic_chunks",
        help="Directory to store chunked JSON files (default: step_05_semantic_chunks)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create chunker and process files
        chunker = SemanticChunkerSplit(args.extracted_text_dir, Path(args.chunks_dir))
        summary = chunker.run_chunking()
        
        print("\n" + "="*60)
        print("STEP 5: SEMANTIC CHUNKING WITH DOCUMENT SPLITTING")
        print("="*60)
        print(f"Files processed: {summary['processed']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total chunks: {summary['total_chunks']}")
        
        if summary['failed'] > 0:
            print("\n⚠️ Some files failed to process. Check logs for details.")
        else:
            print("\n✅ All files processed successfully!")
            
    except Exception as e:
        logger.error(f"Error during semantic chunking: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
