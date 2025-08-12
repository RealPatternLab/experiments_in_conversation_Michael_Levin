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
        if not self.email:
            # Try to get email from environment variable
            import os
            env_email = os.getenv('PIPELINE_EMAIL')
            self.email = env_email if env_email and env_email.strip() else None
        
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
            else:
                metadata['authors'] = []
            
            # Extract journal
            if 'container-title' in data and data['container-title']:
                metadata['journal'] = data['container-title'][0] if isinstance(data['container-title'], list) else data['container-title']
            else:
                metadata['journal'] = None
            
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
            
            # Extract abstract
            if 'abstract' in data and data['abstract']:
                metadata['abstract'] = data['abstract']
            else:
                metadata['abstract'] = None
            
            # Extract document type
            if 'type' in data:
                metadata['document_type'] = data['type']
            else:
                metadata['document_type'] = None
            
            # Extract page info
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
        if not self.email:
            # Try to get email from environment variable
            import os
            env_email = os.getenv('PIPELINE_EMAIL')
            self.email = env_email if env_email and env_email.strip() else None
        
        self.base_url = "https://api.unpaywall.org/v2"
        self.logger = logging.getLogger(__name__)
        
        if not self.email:
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


class MetadataEnricher:
    """
    Main class for enriching metadata using both CrossRef and Unpaywall APIs.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize the metadata enricher.
        
        Args:
            input_dir: Directory containing metadata files to enrich
            output_dir: Directory to save enriched metadata files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Get email from environment if available
        import os
        env_email = os.getenv('PIPELINE_EMAIL')
        
        # Initialize enrichers
        self.crossref_enricher = CrossRefEnricher(email=env_email)
        self.unpaywall_enricher = UnpaywallEnricher(email=env_email)
        
        self.logger = logging.getLogger(__name__)
    
    def get_metadata_files_to_process(self, max_files: Optional[int] = None) -> List[Path]:
        """Get list of metadata files to process."""
        # Look for quick metadata files since they contain the initial metadata
        metadata_files = []
        
        for file_path in self.input_dir.glob("*_quick_metadata.json"):
            if file_path.is_file():
                metadata_files.append(file_path)
        
        if not metadata_files:
            self.logger.info(f"No quick metadata files found in {self.input_dir}")
            return []
        
        self.logger.info(f"Found {len(metadata_files)} quick metadata files to process")
        
        # Sort by modification time (newest first)
        metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Apply max_files limit
        if max_files:
            metadata_files = metadata_files[:max_files]
        
        return metadata_files
    
    def enrich_metadata(self, metadata_file: Path) -> bool:
        """Enrich a single metadata file."""
        try:
            self.logger.info(f"Enriching metadata: {metadata_file.name}")
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check if we have a DOI to work with
            doi = metadata.get('doi')
            if not doi:
                self.logger.info(f"⏭️  Skipping {metadata_file.name} - no DOI available for enrichment")
                return False
            
            # Enrich with CrossRef
            crossref_data = self.crossref_enricher.search_by_doi(doi)
            if crossref_data:
                metadata.update(crossref_data)
                metadata['crossref_enriched'] = True
                self.logger.info(f"✅ Enriched with CrossRef data")
            
            # Enrich with Unpaywall
            unpaywall_data = self.unpaywall_enricher.search_by_doi(doi)
            if unpaywall_data:
                metadata.update(unpaywall_data)
                metadata['unpaywall_enriched'] = True
                self.logger.info(f"✅ Enriched with Unpaywall data")
            
            # Add enrichment timestamp
            metadata['enriched_at'] = datetime.now().isoformat()
            
            # Save enriched metadata
            base_name = metadata_file.stem.replace('_metadata', '')
            output_file = self.output_dir / f"{base_name}_metadata_enriched.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Enriched metadata saved: {output_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error enriching {metadata_file.name}: {e}")
            return False
    
    def process_files(self, max_files: Optional[int] = None) -> Dict:
        """Process multiple metadata files."""
        metadata_files = self.get_metadata_files_to_process(max_files)
        
        if not metadata_files:
            self.logger.info("No files to process")
            return {"total": 0, "processed": 0, "failed": 0}
        
        self.logger.info(f"🔍 Processing {len(metadata_files)} metadata files for enrichment...")
        
        processed = 0
        failed = 0
        
        for metadata_file in metadata_files:
            if self.enrich_metadata(metadata_file):
                processed += 1
            else:
                failed += 1
        
        results = {
            "total": len(metadata_files),
            "processed": processed,
            "failed": failed
        }
        
        self.logger.info(f"📊 Processing complete: {processed} successfully enriched, {failed} could not be enriched")
        return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enrich metadata with CrossRef and Unpaywall APIs")
    parser.add_argument("--input-dir", required=True, help="Directory containing metadata files")
    parser.add_argument("--output-dir", required=True, help="Directory to save enriched metadata files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    try:
        # Create enricher
        enricher = MetadataEnricher(input_dir, output_dir)
        
        # Process files
        results = enricher.process_files(args.max_files)
        
        # Log summary
        logger.info(f"📊 Processing Summary:")
        logger.info(f"   Total files: {results['total']}")
        logger.info(f"   Successfully enriched: {results['processed']}")
        logger.info(f"   Could not enrich (no DOI, etc.): {results['failed']}")
        
        # Exit with success (0) as long as the tool itself worked correctly
        # Individual file enrichment failures are expected and not tool errors
        if results['failed'] > 0:
            logger.info(f"ℹ️  Note: {results['failed']} files could not be enriched (missing DOIs, etc.) - this is normal")
        
        logger.info("✅ Tool completed successfully - individual file enrichment statuses are expected behavior")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Tool encountered an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 