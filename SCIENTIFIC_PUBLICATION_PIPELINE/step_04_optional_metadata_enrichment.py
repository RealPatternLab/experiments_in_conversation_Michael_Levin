#!/usr/bin/env python3
"""
Step 4: Optional Metadata Enrichment with CrossRef API

This enhanced version uses the CrossRef API to enrich metadata when DOIs are available,
providing much more reliable and comprehensive information than text analysis.

Features:
- CrossRef API integration for DOI-based lookups
- Fallback to text analysis for non-DOI documents
- Comprehensive metadata enrichment
- Rate limiting and error handling
"""

import argparse
import json
import os
import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import requests
from urllib.parse import quote

# Progress queue integration
try:
    from pipeline_progress_queue import PipelineProgressQueue
    PROGRESS_QUEUE_AVAILABLE = True
except ImportError:
    PROGRESS_QUEUE_AVAILABLE = False
    logging.warning("Progress queue not available. Pipeline tracking will be limited.")

try:
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: uv add python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('metadata_enrichment.log')
    ]
)
logger = logging.getLogger(__name__)

class CrossRefEnricher:
    """Enrich metadata using CrossRef API."""
    
    def __init__(self, email: Optional[str] = None):
        """Initialize the CrossRef enricher."""
        self.email = email
        if not self.email:
            # Try to get email from environment variable
            env_email = os.getenv('PIPELINE_EMAIL')
            self.email = env_email if env_email and env_email.strip() else None
        
        self.base_url = "https://api.crossref.org"
        
        if not self.email:
            logger.warning("No email provided for CrossRef API. Rate limits may apply.")
    
    def search_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Search CrossRef by DOI."""
        try:
            # Clean DOI - extract from URL if needed
            doi = doi.strip()
            if doi.startswith('https://doi.org/'):
                doi = doi.replace('https://doi.org/', '')
            elif doi.startswith('http://doi.org/'):
                doi = doi.replace('http://doi.org/', '')
            
            if not doi.startswith('10.'):
                logger.warning(f"Invalid DOI format: {doi}")
                return None
            
            logger.info(f"Searching CrossRef by DOI: {doi}")
            
            # Make request
            headers = {'User-Agent': 'MichaelLevinQAEngine/1.0'}
            if self.email:
                headers['User-Agent'] += f' (mailto:{self.email})'
            
            url = f"{self.base_url}/works/{quote(doi)}"
            logger.debug(f"Making request to: {url}")
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Found CrossRef data for DOI: {doi}")
                
                # CrossRef API returns data nested under 'message' key
                if 'message' in data:
                    work_data = data['message']
                    logger.debug(f"CrossRef response structure - keys: {list(data.keys())}")
                    logger.debug(f"Message keys: {list(work_data.keys())}")
                    return self._parse_crossref_response(work_data)
                else:
                    logger.warning("CrossRef response missing 'message' key")
                    return None
            else:
                logger.warning(f"CrossRef DOI lookup failed: {response.status_code} for DOI: {doi}")
                return None
                
        except Exception as e:
            logger.error(f"CrossRef DOI lookup error for {doi}: {e}")
            return None
    
    def _parse_crossref_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CrossRef API response into standardized metadata."""
        try:
            metadata = {}
            
            # Extract title
            if 'title' in data and data['title']:
                metadata['title'] = data['title'][0] if isinstance(data['title'], list) else data['title']
            
            # Extract authors
            if 'author' in data and data['author']:
                authors = []
                for author in data['author']:
                    if 'given' in author and 'family' in author:
                        authors.append(f"{author['given']} {author['family']}")
                    elif 'family' in author:
                        authors.append(author['family'])
                    elif 'name' in author:
                        authors.append(author['name'])
                if authors:
                    metadata['authors'] = authors
            else:
                metadata['authors'] = []
            
            # Debug logging for authors extraction
            logger.debug(f"Authors extraction - author data: {data.get('author')}, result: {metadata['authors']}")
            
            # Extract journal (try both container-title and short-container-title)
            if 'container-title' in data and data['container-title']:
                metadata['journal'] = data['container-title'][0] if isinstance(data['container-title'], list) else data['container-title']
            elif 'short-container-title' in data and data['short-container-title']:
                metadata['journal'] = data['short-container-title'][0] if isinstance(data['short-container-title'], list) else data['short-container-title']
            else:
                metadata['journal'] = None
            
            # Debug logging for journal extraction
            logger.debug(f"Journal extraction - container-title: {data.get('container-title')}, short-container-title: {data.get('short-container-title')}, result: {metadata['journal']}")
            
            # Extract publication date
            if 'published-print' in data and data['published-print']:
                date_parts = data['published-print']['date-parts'][0]
                if len(date_parts) >= 3:
                    metadata['publication_date'] = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 2:
                    metadata['publication_date'] = f"{date_parts[0]}-{date_parts[1]:02d}"
                elif len(date_parts) >= 1:
                    metadata['publication_date'] = str(date_parts[0])
            elif 'published-online' in data and data['published-online']:
                date_parts = data['published-online']['date-parts'][0]
                if len(date_parts) >= 3:
                    metadata['publication_date'] = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 2:
                    metadata['publication_date'] = f"{date_parts[0]}-{date_parts[1]:02d}"
                elif len(date_parts) >= 1:
                    metadata['publication_date'] = str(date_parts[0])
            else:
                metadata['publication_date'] = None
            
            # Extract DOI
            if 'DOI' in data:
                metadata['doi'] = data['DOI']
            
            # Extract abstract (CrossRef usually doesn't have abstracts)
            # We'll keep the original abstract from our extraction
            metadata['abstract'] = None  # CrossRef doesn't typically provide abstracts
            
            # Extract document type
            if 'type' in data:
                metadata['document_type'] = data['type']
            else:
                metadata['document_type'] = None
            
            # Extract page info (CrossRef uses 'page' field)
            if 'page' in data:
                metadata['page_range'] = data['page']
            else:
                metadata['page_range'] = None
            
            # Extract volume and issue
            if 'volume' in data:
                metadata['volume'] = data['volume']
            else:
                metadata['volume'] = None
            if 'issue' in data:
                metadata['issue'] = data['issue']
            else:
                metadata['issue'] = None
            
            # Extract reference count
            if 'reference-count' in data:
                metadata['references_count'] = data['reference-count']
            
            # Extract publisher
            if 'publisher' in data:
                metadata['publisher'] = data['publisher']
            
            # Extract ISSN
            if 'ISSN' in data and data['ISSN']:
                metadata['issn'] = data['ISSN'][0] if isinstance(data['ISSN'], list) else data['ISSN']
            
            # Add source info
            metadata['crossref_data'] = True
            metadata['enrichment_method'] = 'crossref_api'
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing CrossRef response: {e}")
            return {}
    
    def search_by_metadata(self, title: str, journal: str = None, authors: List[str] = None, year: int = None) -> Optional[Dict[str, Any]]:
        """Search CrossRef using multiple metadata fields for better accuracy."""
        try:
            if not title or len(title.strip()) < 10:
                logger.warning(f"Title too short for search: {title}")
                return None
            
            logger.info(f"Searching CrossRef by metadata: {title[:60]}...")
            
            # Build search query
            query_parts = [title]
            if journal:
                query_parts.append(f'"{journal}"')
            if authors and len(authors) > 0:
                # Add first author's last name
                first_author = authors[0].split()[-1] if ' ' in authors[0] else authors[0]
                query_parts.append(first_author)
            
            search_query = ' '.join(query_parts)
            logger.info(f"Search query: {search_query}")
            
            # Make request
            headers = {'User-Agent': 'MichaelLevinQAEngine/1.0'}
            if self.email:
                headers['User-Agent'] += f' (mailto:{self.email})'
            
            params = {
                'query': search_query,
                'rows': 5,  # Get more results to find the best match
                'sort': 'relevance'
            }
            
            if year:
                params['from-pub-date'] = f"{year-1}-01-01"
                params['until-pub-date'] = f"{year+1}-12-31"
            
            url = f"{self.base_url}/works"
            logger.debug(f"Making request to: {url}")
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                if items:
                    logger.info(f"✅ Found {len(items)} CrossRef results")
                    
                    # Find the best match
                    best_match = self._find_best_match(items, title, journal, authors, year)
                    if best_match:
                        logger.info(f"✅ Found best match with DOI: {best_match.get('DOI', 'Unknown')}")
                        return self._parse_crossref_response(best_match)
                    else:
                        logger.warning("No good match found among results")
                        return None
                else:
                    logger.warning(f"No CrossRef results found for metadata search")
                    return None
            else:
                logger.warning(f"CrossRef metadata search failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"CrossRef metadata search error: {e}")
            return None
    
    def _find_best_match(self, items: List[Dict], title: str, journal: str = None, authors: List[str] = None, year: int = None) -> Optional[Dict]:
        """Find the best matching item based on metadata similarity."""
        best_score = 0
        best_item = None
        
        for item in items:
            score = 0
            
            # Title similarity (exact match gets highest score)
            item_title = item.get('title', [''])[0] if item.get('title') else ''
            if title.lower() == item_title.lower():
                score += 100
            elif title.lower() in item_title.lower() or item_title.lower() in title.lower():
                score += 50
            
            # Journal similarity
            if journal:
                item_journal = item.get('container-title', [''])[0] if item.get('container-title') else ''
                if journal.lower() == item_journal.lower():
                    score += 30
                elif journal.lower() in item_journal.lower() or item_journal.lower() in journal.lower():
                    score += 15
            
            # Author similarity
            if authors and item.get('author'):
                item_authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get('author', [])]
                for author in authors[:2]:  # Check first 2 authors
                    if any(author.lower() in item_auth.lower() or item_auth.lower() in author.lower() for item_auth in item_authors):
                        score += 20
            
            # Year similarity
            if year and item.get('published-print'):
                item_year = item['published-print']['date-parts'][0][0]
                if abs(item_year - year) <= 1:  # Allow 1 year difference
                    score += 10
            
            # DOI presence (bonus points)
            if item.get('DOI'):
                score += 5
            
            if score > best_score:
                best_score = score
                best_item = item
        
        # Only return if we have a reasonable match
        if best_score >= 30:  # Minimum threshold
            logger.info(f"Best match score: {best_score}")
            return best_item
        
        return None

class TextBasedEnricher:
    """Fallback enricher using text analysis for documents without DOIs."""
    
    def __init__(self):
        """Initialize the text-based enricher."""
        pass
    
    def extract_affiliations(self, text: str) -> List[str]:
        """Extract institutional affiliations from text."""
        affiliations = []
        
        # Common affiliation patterns
        patterns = [
            r'(?:Department|Dept\.?|School|Institute|University|College|Center|Centre|Laboratory|Lab\.?)\s+of\s+[^,\n]+',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|College|Institute|School|Hospital)',
            r'(?:Department|Dept\.?|School|Institute|University|College|Center|Centre|Laboratory|Lab\.?)\s+[^,\n]+'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                if len(clean_match) > 10 and clean_match not in affiliations:
                    affiliations.append(clean_match)
        
        return affiliations[:5]  # Limit to 5 affiliations
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text."""
        keywords = []
        
        # Look for keyword patterns
        keyword_patterns = [
            r'Keywords?:\s*([^.\n]+)',
            r'Key\s+words?:\s*([^.\n]+)',
            r'Index\s+terms?:\s*([^.\n]+)'
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split by common separators
                parts = re.split(r'[,;]', match)
                for part in parts:
                    clean_part = part.strip()
                    if len(clean_part) > 2 and clean_part not in keywords:
                        keywords.append(clean_part)
        
        return keywords[:10]  # Limit to 10 keywords
    
    def extract_volume_issue(self, text: str) -> Dict[str, str]:
        """Extract volume and issue information."""
        result = {'volume': None, 'issue': None}
        
        # Volume patterns
        volume_patterns = [
            r'Volume\s+(\d+)',
            r'Vol\.?\s*(\d+)',
            r'V\.?\s*(\d+)'
        ]
        
        for pattern in volume_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['volume'] = match.group(1)
                break
        
        # Issue patterns
        issue_patterns = [
            r'Issue\s+(\d+)',
            r'No\.?\s*(\d+)',
            r'Number\s+(\d+)'
        ]
        
        for pattern in issue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result['issue'] = match.group(1)
                break
        
        return result
    
    def extract_page_range(self, text: str) -> Optional[str]:
        """Extract page range information."""
        page_patterns = [
            r'Pages?\s+(\d+[-–]\d+)',
            r'pp\.?\s*(\d+[-–]\d+)',
            r'(\d+[-–]\d+)'
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def extract_reference_count(self, text: str) -> Optional[int]:
        """Extract reference count from text."""
        ref_patterns = [
            r'References?\s*\((\d+)\)',
            r'(\d+)\s+references?',
            r'References?\s*(\d+)'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None

class MetadataEnricher:
    """Main class for enriching metadata using CrossRef API and text analysis."""
    
    def __init__(self, metadata_dir: Path, extracted_text_dir: Path):
        """Initialize the metadata enricher."""
        self.metadata_dir = metadata_dir
        self.extracted_text_dir = extracted_text_dir
        
        # Initialize enrichers
        self.crossref_enricher = CrossRefEnricher()
        self.text_enricher = TextBasedEnricher()
        
        # Rate limiting for CrossRef API
        self.last_api_call = 0
        self.min_delay = 1.0  # 1 second between calls
        
        # Initialize progress queue if available
        self.progress_queue = None
        if PROGRESS_QUEUE_AVAILABLE:
            try:
                self.progress_queue = PipelineProgressQueue()
                logger.info("Progress queue initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize progress queue: {e}")
                self.progress_queue = None
    
    def get_metadata_files_to_process(self) -> List[Dict]:
        """Get list of metadata files that need enrichment."""
        files_to_process = []
        
        if not self.metadata_dir.exists():
            logger.error(f"Metadata directory does not exist: {self.metadata_dir}")
            return files_to_process
        
        # Find all metadata files
        metadata_files = list(self.metadata_dir.glob("*_metadata.json"))
        
        if not metadata_files:
            logger.warning("No metadata files found")
            return files_to_process
        
        # Check which files need enrichment
        for metadata_file in metadata_files:
            base_name = metadata_file.stem.replace('_metadata', '')
            
            # Load metadata to check enrichment status and what's missing
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Check if already enriched
                if metadata.get('enrichment_timestamp'):
                    logger.info(f"Skipping already enriched file: {metadata_file.name}")
                    continue
                
                # Check if enrichment is needed
                if self._needs_enrichment(metadata):
                    # Check if corresponding text file exists
                    text_file = self.extracted_text_dir / f"{base_name}_extracted_text.txt"
                    if text_file.exists():
                        files_to_process.append({
                            'metadata_file': metadata_file,
                            'text_file': text_file,
                            'base_name': base_name,
                            'metadata': metadata
                        })
                        logger.info(f"Found file needing enrichment: {metadata_file.name}")
                    else:
                        logger.warning(f"No corresponding text file for {metadata_file.name}")
                else:
                    logger.info(f"File already complete: {metadata_file.name}")
                    
            except Exception as e:
                logger.error(f"Error reading metadata file {metadata_file.name}: {e}")
        
        logger.info(f"Found {len(files_to_process)} files needing enrichment")
        return files_to_process
    
    def _needs_enrichment(self, metadata: Dict) -> bool:
        """Check if metadata needs enrichment."""
        # Check for missing or incomplete fields
        missing_fields = []
        
        if not metadata.get('publication_year') or metadata.get('publication_year') == 'null':
            missing_fields.append('publication_year')
        
        if not metadata.get('journal') or metadata.get('journal') == 'null':
            missing_fields.append('journal')
        
        if not metadata.get('authors') or len(metadata.get('authors', [])) == 0:
            missing_fields.append('authors')
        
        if not metadata.get('doi') or metadata.get('doi') == 'null':
            missing_fields.append('doi')
        
        # Check for additional fields that could be enriched
        enrichment_fields = ['volume', 'issue', 'page_range', 'abstract', 'keywords', 'affiliations']
        for field in enrichment_fields:
            if not metadata.get(field) or metadata.get(field) == 'null':
                missing_fields.append(field)
        
        if missing_fields:
            logger.info(f"Missing fields: {missing_fields}")
            return True
        
        return False
    
    def enrich_metadata(self, file_info: Dict) -> bool:
        """Enrich metadata for a single file."""
        try:
            metadata_file = file_info['metadata_file']
            text_file = file_info['text_file']
            base_name = file_info['base_name']
            metadata = file_info['metadata']
            
            logger.info(f"Enriching metadata: {metadata_file.name}")
            
            # Find document ID from progress queue if available
            doc_id = None
            if self.progress_queue:
                try:
                    # Search for document with this filename in progress queue
                    for doc_id_candidate, doc_data in self.progress_queue._load_queue_data()["pipeline_progress"].items():
                        if doc_data.get("new_filename") == f"{base_name}.pdf":
                            doc_id = doc_id_candidate
                            break
                    
                    if doc_id:
                        logger.info(f"Found document {doc_id} in progress queue")
                    else:
                        logger.warning(f"Document {base_name} not found in progress queue")
                except Exception as e:
                    logger.warning(f"Progress queue lookup error: {e}")
                    doc_id = None
            
            # Try CrossRef API first if DOI is available
            doi = metadata.get('doi')
            if doi and doi != 'null':
                logger.info(f"Attempting CrossRef enrichment for DOI: {doi}")
                
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_api_call < self.min_delay:
                    sleep_time = self.min_delay - (current_time - self.last_api_call)
                    logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                
                crossref_data = self.crossref_enricher.search_by_doi(doi)
                if crossref_data:
                    logger.info(f"CrossRef data received: {list(crossref_data.keys())}")
                    
                    # Update metadata with CrossRef data
                    original_metadata = metadata.copy()
                    metadata.update(crossref_data)
                    metadata['crossref_enriched'] = True
                    metadata['enrichment_timestamp'] = datetime.now().isoformat()
                    metadata['enrichment_method'] = 'crossref_api'
                    
                    logger.info(f"✅ Successfully enriched with CrossRef data")
                    
                    # Log what was updated
                    updated_fields = []
                    for key, value in crossref_data.items():
                        if key in original_metadata and original_metadata[key] != value:
                            updated_fields.append(f"{key}: {original_metadata[key]} → {value}")
                    
                    if updated_fields:
                        logger.info(f"Updated fields: {updated_fields}")
                    
                    self.last_api_call = time.time()
                else:
                    logger.warning(f"CrossRef enrichment failed for DOI: {doi}")
            
            # If DOI enrichment failed or DOI is missing, try metadata search
            if not crossref_data and (not doi or doi == 'null' or self._needs_enrichment(metadata)):
                logger.info("Attempting CrossRef metadata search as fallback")
                
                # Rate limiting
                current_time = time.time()
                if current_time - self.last_api_call < self.min_delay:
                    sleep_time = self.min_delay - (current_time - self.last_api_call)
                    logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                
                # Try to find the paper using metadata
                title = metadata.get('title')
                journal = metadata.get('journal')
                authors = metadata.get('authors', [])
                year = metadata.get('publication_year')
                
                if title and journal:
                    crossref_data = self.crossref_enricher.search_by_metadata(
                        title=title,
                        journal=journal,
                        authors=authors,
                        year=year
                    )
                    
                    if crossref_data:
                        logger.info(f"✅ Found paper via metadata search with DOI: {crossref_data.get('doi', 'Unknown')}")
                        
                        # Update metadata with CrossRef data
                        original_metadata = metadata.copy()
                        metadata.update(crossref_data)
                        metadata['crossref_enriched'] = True
                        metadata['enrichment_timestamp'] = datetime.now().isoformat()
                        metadata['enrichment_method'] = 'crossref_metadata_search'
                        
                        # Log what was found
                        found_fields = []
                        for key, value in crossref_data.items():
                            if value and value != 'null':
                                found_fields.append(key)
                        
                        if found_fields:
                            logger.info(f"Found fields via metadata search: {found_fields}")
                        
                        self.last_api_call = time.time()
                    else:
                        logger.warning("CrossRef metadata search failed")
                else:
                    logger.warning("Insufficient metadata for CrossRef search (need title and journal)")
            
            # Fallback to text analysis for remaining missing fields
            if self._needs_enrichment(metadata):
                logger.info("Attempting text-based enrichment for remaining fields")
                
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # Extract additional information from text
                    text_updates = {}
                    
                    # Extract affiliations
                    affiliations = self.text_enricher.extract_affiliations(text_content)
                    if affiliations and not metadata.get('affiliations'):
                        text_updates['affiliations'] = affiliations
                    
                    # Extract keywords
                    keywords = self.text_enricher.extract_keywords(text_content)
                    if keywords and not metadata.get('keywords'):
                        text_updates['keywords'] = keywords
                    
                    # Extract volume/issue
                    volume_issue = self.text_enricher.extract_volume_issue(text_content)
                    if volume_issue['volume'] and not metadata.get('volume'):
                        text_updates['volume'] = volume_issue['volume']
                    if volume_issue['issue'] and not metadata.get('issue'):
                        text_updates['issue'] = volume_issue['issue']
                    
                    # Extract page range
                    page_range = self.text_enricher.extract_page_range(text_content)
                    if page_range and not metadata.get('page_range'):
                        text_updates['page_range'] = page_range
                    
                    # Extract reference count
                    ref_count = self.text_enricher.extract_reference_count(text_content)
                    if ref_count and not metadata.get('reference_count'):
                        text_updates['reference_count'] = ref_count
                    
                    # Update metadata with text-based findings
                    if text_updates:
                        metadata.update(text_updates)
                        metadata['text_enriched'] = True
                        metadata['enrichment_method'] = metadata.get('enrichment_method', 'text_analysis')
                        if 'crossref_enriched' not in metadata:
                            metadata['enrichment_timestamp'] = datetime.now().isoformat()
                        
                        logger.info(f"✅ Text-based enrichment added: {list(text_updates.keys())}")
                    
                except Exception as e:
                    logger.error(f"Error during text-based enrichment: {e}")
            
            # Update the original metadata file directly
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Original metadata file updated: {metadata_file.name}")
            
            # Update progress queue if available
            if self.progress_queue and doc_id:
                try:
                    self.progress_queue.update_step_status(
                        doc_id, 
                        "step_04_metadata_enrichment", 
                        "complete"
                    )
                    logger.info(f"Updated progress queue for {doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to update progress queue: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error enriching metadata for {file_info['metadata_file'].name}: {e}")
            
            # Update progress queue if available
            if self.progress_queue and doc_id:
                try:
                    self.progress_queue.add_error(doc_id, "step_04_metadata_enrichment", f"Error: {e}")
                except Exception as queue_error:
                    logger.warning(f"Failed to update progress queue: {queue_error}")
            
            return False
    
    def run_enrichment(self) -> Dict:
        """Run the metadata enrichment process."""
        logger.info("Starting metadata enrichment process...")
        
        # Get files to process
        files = self.get_metadata_files_to_process()
        
        if not files:
            logger.info("No files need enrichment - all done!")
            return {'total': 0, 'enriched': 0, 'failed': 0}
        
        # Process files
        enriched = 0
        failed = 0
        
        for file_info in files:
            if self.enrich_metadata(file_info):
                enriched += 1
            else:
                failed += 1
        
        # Summary
        summary = {
            'total': len(files),
            'enriched': enriched,
            'failed': failed
        }
        
        logger.info("Enrichment Summary:")
        logger.info(f"   Files processed: {summary['total']}")
        logger.info(f"   Successfully enriched: {summary['enriched']}")
        logger.info(f"   Failed: {summary['failed']}")
        
        return summary

def main():
    """Main function to run the metadata enrichment pipeline."""
    parser = argparse.ArgumentParser(description="Step 4: Optional Metadata Enrichment with CrossRef API")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("step_02_metadata"),
        help="Directory containing metadata files from Step 2 (default: step_02_metadata)"
    )
    parser.add_argument(
        "--extracted-text-dir",
        type=Path,
        default=Path("step_03_extracted_text"),
        help="Directory containing extracted text files from Step 3 (default: step_03_extracted_text)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create enricher and process files
        enricher = MetadataEnricher(args.metadata_dir, args.extracted_text_dir)
        summary = enricher.run_enrichment()
        
        print("\n" + "="*60)
        print("STEP 4: OPTIONAL METADATA ENRICHMENT WITH CROSSREF API")
        print("="*60)
        print(f"Files processed: {summary['total']}")
        print(f"Successfully enriched: {summary['enriched']}")
        print(f"Failed: {summary['failed']}")
        
        if summary['failed'] > 0:
            print("\n⚠️ Some files failed to enrich. Check logs for details.")
        else:
            print("\n✅ All files processed successfully!")
            
    except Exception as e:
        logger.error(f"Error during metadata enrichment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
