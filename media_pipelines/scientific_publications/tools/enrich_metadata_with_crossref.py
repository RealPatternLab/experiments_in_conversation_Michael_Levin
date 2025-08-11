#!/usr/bin/env python3
"""
Enrich PDF Metadata with CrossRef and Unpaywall APIs

This tool queries external APIs to enrich metadata extracted from PDFs.
It uses CrossRef for DOI-based lookups and Unpaywall for additional metadata.

Updated for the new scientific publications pipeline structure.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging
import time
import requests
from urllib.parse import quote

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/enrich_metadata_with_crossref.log')
    ]
)
logger = logging.getLogger(__name__)


class CrossRefEnricher:
    """
    Enrich metadata using CrossRef API.
    """
    
    def __init__(self, email: Optional[str] = None):
        """
        Initialize the CrossRef enricher.
        
        Args:
            email: Email for CrossRef API (recommended for higher rate limits)
        """
        self.email = email
        self.base_url = "https://api.crossref.org"
        self.logger = logging.getLogger(__name__)
    
    def search_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Search CrossRef by DOI.
        
        Args:
            doi: DOI to search for
            
        Returns:
            CrossRef metadata or None
        """
        try:
            # Clean DOI
            doi = doi.strip()
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
                return self._parse_crossref_response(data)
            else:
                logger.warning(f"CrossRef DOI lookup failed: {response.status_code} for DOI: {doi}")
                return None
                
        except Exception as e:
            logger.error(f"CrossRef DOI lookup error for {doi}: {e}")
            return None
    
    def search_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Search CrossRef by title.
        
        Args:
            title: Title to search for
            
        Returns:
            CrossRef metadata or None
        """
        try:
            if not title or len(title.strip()) < 10:
                logger.warning(f"Title too short for search: {title}")
                return None
            
            logger.info(f"Searching CrossRef by title: {title[:60]}...")
            
            # Make request
            headers = {'User-Agent': 'MichaelLevinQAEngine/1.0'}
            if self.email:
                headers['User-Agent'] += f' (mailto:{self.email})'
            
            params = {
                'query': title,
                'rows': 1,
                'sort': 'relevance'
            }
            
            url = f"{self.base_url}/works"
            logger.debug(f"Making request to: {url}")
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('message', {}).get('items', [])
                
                if items:
                    logger.info(f"✅ Found CrossRef data by title")
                    return self._parse_crossref_response(items[0])
                else:
                    logger.warning(f"No CrossRef results found for title: {title[:60]}...")
                    return None
            else:
                logger.warning(f"CrossRef title search failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"CrossRef title search error: {e}")
            return None
    
    def _parse_crossref_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse CrossRef API response into standardized metadata.
        
        Args:
            data: Raw CrossRef API response
            
        Returns:
            Standardized metadata dictionary
        """
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
            
            # Extract journal
            if 'container-title' in data and data['container-title']:
                metadata['journal'] = data['container-title'][0] if isinstance(data['container-title'], list) else data['container-title']
            
            # Extract publication date
            if 'published-print' in data and data['published-print']:
                date_parts = data['published-print']['date-parts'][0]
                if len(date_parts) >= 1:
                    metadata['publication_date'] = str(date_parts[0])
            elif 'published-online' in data and data['published-online']:
                date_parts = data['published-online']['date-parts'][0]
                if len(date_parts) >= 1:
                    metadata['publication_date'] = str(date_parts[0])
            
            # Extract DOI
            if 'DOI' in data:
                metadata['doi'] = data['DOI']
            
            # Extract abstract
            if 'abstract' in data and data['abstract']:
                metadata['abstract'] = data['abstract']
            
            # Extract document type
            if 'type' in data:
                metadata['document_type'] = data['type']
            
            # Extract page info
            if 'page' in data:
                metadata['page_range'] = data['page']
            
            # Extract volume and issue
            if 'volume' in data:
                metadata['volume'] = data['volume']
            if 'issue' in data:
                metadata['issue'] = data['issue']
            
            # Add source info
            metadata['crossref_data'] = True
            metadata['enrichment_method'] = 'crossref_api'
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing CrossRef response: {e}")
            return {}


class UnpaywallEnricher:
    """
    Enrich metadata using Unpaywall API.
    """
    
    def __init__(self, email: Optional[str] = None):
        """
        Initialize the Unpaywall enricher.
        
        Args:
            email: Email for Unpaywall API (required)
        """
        self.email = email
        self.base_url = "https://api.unpaywall.org/v2"
        self.logger = logging.getLogger(__name__)
        
        if not email:
            logger.warning("No email provided for Unpaywall API. Rate limits may apply.")
    
    def search_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Search Unpaywall by DOI.
        
        Args:
            doi: DOI to search for
            
        Returns:
            Unpaywall metadata or None
        """
        try:
            if not self.email:
                logger.warning("Email required for Unpaywall API")
                return None
            
            # Clean DOI
            doi = doi.strip()
            if not doi.startswith('10.'):
                logger.warning(f"Invalid DOI format: {doi}")
                return None
            
            logger.info(f"Searching Unpaywall by DOI: {doi}")
            
            # Make request
            url = f"{self.base_url}/{quote(doi)}"
            params = {'email': self.email}
            
            logger.debug(f"Making request to: {url}")
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Found Unpaywall data for DOI: {doi}")
                return self._parse_unpaywall_response(data)
            else:
                logger.warning(f"Unpaywall DOI lookup failed: {response.status_code} for DOI: {doi}")
                return None
                
        except Exception as e:
            logger.error(f"Unpaywall DOI lookup error for {doi}: {e}")
            return None
    
    def _parse_unpaywall_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Unpaywall API response into standardized metadata.
        
        Args:
            data: Raw Unpaywall API response
            
        Returns:
            Standardized metadata dictionary
        """
        try:
            metadata = {}
            
            # Extract open access status
            if 'is_oa' in data:
                metadata['is_open_access'] = data['is_oa']
            
            # Extract best open access location
            if 'best_oa_location' in data and data['best_oa_location']:
                best_oa = data['best_oa_location']
                if 'url_for_pdf' in best_oa:
                    metadata['pdf_url'] = best_oa['url_for_pdf']
                if 'url_for_landing_page' in best_oa:
                    metadata['landing_page_url'] = best_oa['url_for_landing_page']
                if 'host_type' in best_oa:
                    metadata['host_type'] = best_oa['host_type']
            
            # Extract publisher
            if 'publisher' in data:
                metadata['publisher'] = data['publisher']
            
            # Extract journal name
            if 'journal_name' in data:
                metadata['journal_name'] = data['journal_name']
            
            # Extract publication date
            if 'published_date' in data:
                metadata['published_date'] = data['published_date']
            
            # Extract license
            if 'license' in data:
                metadata['license'] = data['license']
            
            # Add source info
            metadata['unpaywall_data'] = True
            metadata['enrichment_method'] = 'unpaywall_api'
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing Unpaywall response: {e}")
            return {}


def enrich_metadata_with_apis(input_dir: Path, output_dir: Path, email: Optional[str] = None) -> None:
    """
    Enrich metadata using CrossRef and Unpaywall APIs.
    
    Args:
        input_dir: Directory containing chunk JSON files
        output_dir: Directory to save enriched metadata
        email: Email for API access (recommended for higher rate limits)
    """
    if not input_dir.exists():
        logger.error(f"❌ Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find chunk files
    chunk_files = list(input_dir.glob("*_chunks.json"))
    if not chunk_files:
        logger.info("ℹ️  No chunk files found")
        return
    
    logger.info(f"🔍 Found {len(chunk_files)} chunk files to enrich")
    
    # Check for already processed files
    already_processed = set()
    if output_dir.exists():
        for enriched_file in output_dir.glob("*_enriched.json"):
            # Parse filename: pdf_20250803_124125_152265_chunks_enriched.json -> pdf_20250803_124125_152265_chunks.json
            base_name = enriched_file.stem.replace('_chunks_enriched', '')
            chunk_filename = f"{base_name}_chunks.json"
            already_processed.add(chunk_filename)
    
    logger.info(f"📊 Found {len(already_processed)} already processed files")
    
    # Filter out already processed files
    unprocessed_files = []
    for chunk_file in chunk_files:
        if chunk_file.name not in already_processed:
            unprocessed_files.append(chunk_file)
            logger.info(f"✅ Found unprocessed chunk file: {chunk_file.name}")
        else:
            logger.info(f"⏭️  Skipping already processed file: {chunk_file.name}")
    
    if not unprocessed_files:
        logger.info("🎉 All chunk files have already been processed!")
        return
    
    logger.info(f"🔄 Processing {len(unprocessed_files)} unprocessed files")
    
    # Initialize enrichers
    crossref_enricher = CrossRefEnricher(email)
    unpaywall_enricher = UnpaywallEnricher(email)
    
    successful_enrichments = 0
    failed_enrichments = 0
    
    for chunk_file in unprocessed_files:
        try:
            # Load chunk data
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            logger.info(f"🔄 Enriching {chunk_file.name}")
            
            # Extract metadata from chunk file
            file_metadata = chunk_data.get('file_metadata', {})
            
            # Extract key fields for enrichment
            doi = file_metadata.get('doi')
            title = file_metadata.get('title')
            
            # If no title in metadata, try to extract from filename
            if not title or title == 'Unknown':
                title = chunk_file.stem.replace('_chunks', '')
            
            enriched_data = {}
            enrichment_methods = []
            
            # Try CrossRef by DOI first
            if doi:
                logger.info(f"  🔍 Searching CrossRef by DOI: {doi}")
                crossref_data = crossref_enricher.search_by_doi(doi)
                if crossref_data:
                    enriched_data.update(crossref_data)
                    enrichment_methods.append('crossref')
                    logger.info(f"    ✅ Found CrossRef data")
                time.sleep(1)  # Rate limiting
            
            # Try CrossRef by title if no DOI or no CrossRef data
            if not doi or not enriched_data:
                if title and title != 'Unknown':
                    logger.info(f"  🔍 Searching CrossRef by title: {title[:60]}...")
                    crossref_data = crossref_enricher.search_by_title(title)
                    if crossref_data:
                        enriched_data.update(crossref_data)
                        enrichment_methods.append('crossref_title')
                        logger.info(f"    ✅ Found CrossRef data by title")
                    time.sleep(1)  # Rate limiting
            
            # Try Unpaywall if we have DOI
            if doi:
                logger.info(f"  🔍 Searching Unpaywall by DOI: {doi}")
                unpaywall_data = unpaywall_enricher.search_by_doi(doi)
                if unpaywall_data:
                    enriched_data.update(unpaywall_data)
                    enrichment_methods.append('unpaywall')
                    logger.info(f"    ✅ Found Unpaywall data")
                time.sleep(1)  # Rate limiting
            
            # Save enriched metadata
            if enriched_data:
                # Create enriched chunk data
                enriched_chunk_data = chunk_data.copy()
                
                # Update file metadata with enriched data
                if 'file_metadata' not in enriched_chunk_data:
                    enriched_chunk_data['file_metadata'] = {}
                enriched_chunk_data['file_metadata'].update(enriched_data)
                
                # Add enrichment info
                enriched_chunk_data['enriched_at'] = datetime.now().isoformat()
                enriched_chunk_data['enrichment_methods'] = enrichment_methods
                
                # Save enriched chunk data
                enriched_filename = f"{chunk_file.stem}_enriched.json"
                enriched_file = output_dir / enriched_filename
                
                with open(enriched_file, 'w', encoding='utf-8') as f:
                    json.dump(enriched_chunk_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"  ✅ Enriched with: {', '.join(enrichment_methods)}")
                if enriched_data.get('title'):
                    logger.info(f"     Title: {enriched_data['title'][:60]}...")
                if enriched_data.get('authors'):
                    logger.info(f"     Authors: {', '.join(enriched_data['authors'][:3])}")
                if enriched_data.get('doi'):
                    logger.info(f"     DOI: {enriched_data['doi']}")
                if enriched_data.get('is_open_access'):
                    logger.info(f"     Open Access: {enriched_data['is_open_access']}")
                logger.info(f"     Saved to: {enriched_file}")
                
                successful_enrichments += 1
            else:
                logger.warning(f"  ⚠️  No enrichment data found")
                failed_enrichments += 1
        
        except Exception as e:
            logger.error(f"  ❌ Error enriching {chunk_file.name}: {e}")
            failed_enrichments += 1
    
    logger.info(f"\n📊 API Enrichment Summary:")
    logger.info(f"  ✅ Successful: {successful_enrichments}")
    logger.info(f"  ❌ Failed: {failed_enrichments}")
    logger.info(f"  ⏭️  Skipped: {len(chunk_files) - len(unprocessed_files)}")
    logger.info(f"  📄 Total processed: {len(unprocessed_files)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Enrich semantic chunk metadata using CrossRef and Unpaywall APIs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/transformed_data/semantic_chunking"),
        help="Directory containing chunk JSON files (default: data/transformed_data/semantic_chunking)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/transformed_data/metadata_enrichment"),
        help="Directory to save enriched metadata (default: data/transformed_data/metadata_enrichment)"
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Email for API access (recommended for higher rate limits)"
    )
    
    args = parser.parse_args()
    
    try:
        enrich_metadata_with_apis(args.input_dir, args.output_dir, args.email)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 