#!/usr/bin/env python3
"""
Pydantic models for semantic chunk validation.
These models define the expected structure of chunk data from Gemini Pro.
"""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime

class SemanticChunk(BaseModel):
    """Model for a single semantic chunk from a PDF."""
    
    text: str = Field(..., description="The semantically meaningful paragraph or group of sentences (ideally 100–300 words)")
    section: str = Field(..., description="Which part of the paper the chunk is from (e.g., Abstract, Introduction, Methods, Results, Discussion, Conclusion)")
    topic: str = Field(..., description="A concise topic or concept that this chunk is about")
    chunk_summary: str = Field(..., description="A one-sentence summary of the chunk's core message")
    position_in_section: Literal["Beginning", "Middle", "End"] = Field(..., description="Indicate if it appears at the beginning, middle, or end of the section")
    certainty_level: Literal["High", "Medium", "Low"] = Field(..., description="Your confidence that the chunk expresses Levin's core view or a key claim")
    citation_context: str = Field(..., description="If relevant, describe whether the chunk is referring to prior work, presenting new results, or drawing conclusions")
    page_number: Optional[Union[str, int]] = Field(None, description="The page number where this chunk appears (if available)")
    
    @validator('text')
    def validate_text_length(cls, v):
        """Ensure text is not empty and has reasonable length."""
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v.strip()) < 50:
            raise ValueError('Text should be at least 50 characters')
        if len(v.strip()) > 10000:
            raise ValueError('Text should not exceed 10000 characters')
        return v.strip()
    
    @validator('section')
    def validate_section(cls, v):
        """Ensure section is a valid paper section."""
        valid_sections = [
            "Abstract", "Introduction", "Methods", "Results", "Discussion", 
            "Conclusion", "Materials and Methods", "Experimental", "Background",
            "Literature Review", "Analysis", "Summary", "Future Work"
        ]
        if v not in valid_sections:
            # Allow custom sections but warn
            print(f"⚠️  Warning: Unknown section '{v}' - this may be valid for this paper")
        return v
    
    @validator('topic')
    def validate_topic(cls, v):
        """Ensure topic is not empty."""
        if not v or not v.strip():
            raise ValueError('Topic cannot be empty')
        return v.strip()
    
    @validator('chunk_summary')
    def validate_summary(cls, v):
        """Ensure summary is not empty and reasonable length."""
        if not v or not v.strip():
            raise ValueError('Chunk summary cannot be empty')
        if len(v.strip()) > 500:
            raise ValueError('Chunk summary should not exceed 500 characters')
        return v.strip()
    
    @validator('citation_context')
    def validate_citation_context(cls, v):
        """Ensure citation context is not empty."""
        if not v or not v.strip():
            raise ValueError('Citation context cannot be empty')
        return v.strip()

class SemanticChunksResponse(BaseModel):
    """Model for the complete response from Gemini containing multiple chunks."""
    
    chunks: List[SemanticChunk] = Field(..., description="Array of semantic chunks")
    
    @validator('chunks')
    def validate_chunks(cls, v):
        """Ensure we have at least one chunk."""
        if not v:
            raise ValueError('Response must contain at least one chunk')
        if len(v) > 100:
            raise ValueError('Response should not contain more than 100 chunks')
        return v

class ChunkProcessingResult(BaseModel):
    """Model for tracking the processing result of a PDF chunking operation."""
    
    sanitized_file_id: int
    original_filename: str
    sanitized_filename: str
    processing_status: Literal["pending", "processing", "completed", "failed"] = "pending"
    chunks_count: int = 0
    total_chunks: int = 0
    error_message: Optional[str] = None
    gemini_response_json: Optional[str] = None
    processed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class ChunkMetadata(BaseModel):
    """Model for storing additional metadata with each chunk."""
    
    # File information
    filename: str
    original_filename: str
    file_size: Optional[int] = None
    
    # Document classification
    document_type: str = "research_paper"  # research_paper, review, book_chapter, patent, etc.
    
    # Author and publication information (from metadata extraction)
    author: Optional[str] = None
    authors: Optional[str] = None  # Full authors string
    year: Optional[str] = None  # Changed from int to str to handle "unknown"
    source_title: Optional[str] = None  # e.g., "Journal of Regenerative Biology"
    journal: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[str] = None
    
    # Processing and quality information
    chunk_index: int
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: Optional[float] = None  # From metadata extraction
    extraction_method: Optional[str] = None  # How metadata was extracted
    
    # Enrichment information
    enrichment_sources: Optional[str] = None  # JSON string of enrichment sources used
    cross_reference_match: Optional[str] = None  # Whether this matched reference data
    
    # Content analysis
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    
    # Pipeline tracking
    sanitized_file_id: Optional[int] = None
    metadata_extraction_id: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class EnrichedChunk(BaseModel):
    """Model for a chunk with additional metadata."""
    
    # Core chunk data
    chunk: SemanticChunk
    
    # Additional metadata
    metadata: ChunkMetadata
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

def validate_gemini_response(response_text: str) -> SemanticChunksResponse:
    """
    Validate a Gemini response and return structured chunks.
    
    Args:
        response_text: Raw text response from Gemini
        
    Returns:
        SemanticChunksResponse: Validated chunks
        
    Raises:
        ValueError: If response is invalid
    """
    try:
        # Try to parse as JSON array directly
        import json
        data = json.loads(response_text)
        
        # If it's a list, wrap it in the expected structure
        if isinstance(data, list):
            return SemanticChunksResponse(chunks=data)
        else:
            raise ValueError("Response should be a JSON array")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")
    except Exception as e:
        raise ValueError(f"Validation error: {e}")

def create_enriched_chunk(
    chunk: SemanticChunk, 
    chunk_index: int,
    filename: str,
    original_filename: str,
    author: Optional[str] = None,
    authors: Optional[str] = None,
    year: Optional[str] = None,  # Changed from int to str
    source_title: Optional[str] = None,
    journal: Optional[str] = None,
    doi: Optional[str] = None,
    publication_date: Optional[str] = None,
    confidence_score: Optional[float] = None,
    extraction_method: Optional[str] = None,
    document_type: Optional[str] = None,
    file_size: Optional[int] = None,
    sanitized_file_id: Optional[int] = None,
    metadata_extraction_id: Optional[int] = None,
    enrichment_sources: Optional[str] = None,
    cross_reference_match: Optional[str] = None,
    word_count: Optional[int] = None,
    character_count: Optional[int] = None
) -> EnrichedChunk:
    """
    Create an enriched chunk with metadata.
    
    Args:
        chunk: The semantic chunk
        chunk_index: Index of the chunk in the document
        filename: Current filename
        original_filename: Original filename
        author: Author name
        authors: Full authors string
        year: Publication year
        source_title: Journal or source title
        journal: Journal name
        doi: DOI if available
        publication_date: Publication date
        confidence_score: Confidence score from metadata extraction
        extraction_method: Method used for metadata extraction
        document_type: Type of document
        file_size: File size in bytes
        sanitized_file_id: ID of the sanitized file
        metadata_extraction_id: ID of the metadata extraction record
        enrichment_sources: JSON string of enrichment sources used
        cross_reference_match: Whether this matched reference data
        word_count: Number of words in the chunk
        character_count: Number of characters in the chunk
        
    Returns:
        EnrichedChunk: Chunk with metadata
    """
    try:
        # Debug: Log the parameters
        print(f"🔍 Creating enriched chunk with parameters:")
        print(f"  chunk_index: {chunk_index}")
        print(f"  filename: {filename}")
        print(f"  original_filename: {original_filename}")
        print(f"  author: {author}")
        print(f"  authors: {authors}")
        print(f"  year: {year}")
        print(f"  source_title: {source_title}")
        print(f"  journal: {journal}")
        print(f"  doi: {doi}")
        print(f"  publication_date: {publication_date}")
        print(f"  confidence_score: {confidence_score}")
        print(f"  extraction_method: {extraction_method}")
        print(f"  document_type: {document_type}")
        print(f"  file_size: {file_size}")
        print(f"  sanitized_file_id: {sanitized_file_id}")
        print(f"  metadata_extraction_id: {metadata_extraction_id}")
        print(f"  enrichment_sources: {enrichment_sources}")
        print(f"  cross_reference_match: {cross_reference_match}")
        print(f"  word_count: {word_count}")
        print(f"  character_count: {character_count}")
        
        metadata = ChunkMetadata(
            filename=filename,
            original_filename=original_filename,
            file_size=file_size,
            document_type=document_type or "research_paper",
            author=author,
            authors=authors,
            year=year,
            source_title=source_title,
            journal=journal,
            doi=doi,
            publication_date=publication_date,
            chunk_index=chunk_index,
            confidence_score=confidence_score,
            extraction_method=extraction_method,
            enrichment_sources=enrichment_sources,
            cross_reference_match=cross_reference_match,
            word_count=word_count,
            character_count=character_count,
            sanitized_file_id=sanitized_file_id,
            metadata_extraction_id=metadata_extraction_id
        )
        
        return EnrichedChunk(chunk=chunk, metadata=metadata)
    except Exception as e:
        print(f"❌ Error in create_enriched_chunk: {e}")
        raise 