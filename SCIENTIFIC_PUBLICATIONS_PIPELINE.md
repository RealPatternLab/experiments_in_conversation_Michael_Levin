# Scientific Publications Pipeline - Design Document

## Overview

This document outlines the design for the first pipeline in the Virtual Person Pipeline system, specifically focused on processing scientific publications related to Michael Levin's research. This pipeline will serve as the foundation for building a RAG system based on his published work.

## Pipeline Architecture

### 1. File Ingestion & Uniqueness Validation
- **Input**: Scientific publication files (PDFs, primarily)
- **Output**: Unique files ready for processing or deletion with explanation

#### Step 1.1: Hash Generation & Duplicate Detection
```python
class HashValidator:
    def generate_file_hash(self, file_path: str) -> str
    def check_hash_exists(self, file_hash: str) -> bool
    def record_hash(self, file_hash: str, metadata: dict)
```

**Process Flow:**
1. Generate SHA-256 hash of file content
2. Check hash against database of processed files
3. If hash exists: record duplicate, delete file, exit pipeline
4. If hash is unique: proceed to next step

#### Step 1.2: Metadata Extraction & Duplicate Detection
```python
class MetadataExtractor:
    def extract_publication_metadata(self, file_path: str) -> PublicationMetadata
    def check_metadata_duplicate(self, metadata: PublicationMetadata) -> bool
```

**Metadata Fields to Extract:**
- Publication title
- Publication year
- Authors (full list)
- DOI (if available)
- Journal/conference name
- Abstract (if available)

**Duplicate Detection Logic:**
- If metadata matches existing record: delete file, record reason, exit pipeline
- If metadata is unique: proceed to text extraction

### 2. Text Extraction
- **Input**: Validated unique publication files
- **Output**: Extracted text content ready for semantic analysis

```python
class TextExtractor:
    def extract_text_from_pdf(self, file_path: str) -> str
    def clean_extracted_text(self, raw_text: str) -> str
    def validate_text_quality(self, text: str) -> bool
```

**Text Processing Steps:**
1. Extract raw text from PDF using PyPDF2 or similar
2. Clean text (remove headers, footers, page numbers)
3. Validate text quality (minimum length, readable content)
4. Store extracted text with file reference

### 3. Semantic Chunking
- **Input**: Clean extracted text
- **Output**: Semantically meaningful text chunks

```python
class SemanticChunker:
    def chunk_by_semantic_units(self, text: str) -> List[TextChunk]
    def chunk_by_research_topics(self, text: str) -> List[TextChunk]
    def chunk_by_methodology(self, text: str) -> List[TextChunk]
```

**Chunking Strategy:**
- **Research Topic Chunks**: Group text by research themes (e.g., bioelectricity, regeneration, developmental biology)
- **Methodology Chunks**: Group by experimental methods and techniques
- **Result Chunks**: Group by findings and conclusions
- **Context Chunks**: Group by background and literature review

**Chunk Metadata:**
- Source file reference
- Chunk type (topic, method, result, context)
- Chunk position in document
- Related research areas
- Key terms and concepts

### 4. FAISS Embedding Generation
- **Input**: Semantic text chunks
- **Output**: Vector embeddings stored in FAISS index

```python
class EmbeddingGenerator:
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[Vector]
    def store_in_faiss(self, embeddings: List[Vector], metadata: List[dict])
    def update_faiss_index(self, new_embeddings: List[Vector])
```

**Embedding Process:**
1. Generate embeddings using sentence-transformers model
2. Store embeddings in FAISS index with metadata
3. Maintain index versioning for updates
4. Optimize index for similarity search

## Data Models

### PublicationMetadata
```python
@dataclass
class PublicationMetadata:
    file_hash: str
    filename: str
    title: str
    authors: List[str]
    publication_year: int
    doi: Optional[str]
    journal: str
    abstract: Optional[str]
    processing_timestamp: datetime
    is_duplicate: bool
    duplicate_reason: Optional[str]
```

### TextChunk
```python
@dataclass
class TextChunk:
    chunk_id: str
    source_file_hash: str
    chunk_text: str
    chunk_type: ChunkType
    chunk_position: int
    research_topics: List[str]
    key_terms: List[str]
    embedding_vector: Optional[List[float]]
    created_timestamp: datetime
```

### ProcessingRecord
```python
@dataclass
class ProcessingRecord:
    file_hash: str
    filename: str
    processing_status: ProcessingStatus
    processing_steps: List[ProcessingStep]
    error_messages: List[str]
    processing_duration: float
    created_timestamp: datetime
    updated_timestamp: datetime
```

## Database Schema

### Tables
1. **processed_files**: Track all processed files and their status
2. **file_hashes**: Store file hashes for duplicate detection
3. **publication_metadata**: Store extracted publication information
4. **text_chunks**: Store processed text chunks
5. **embeddings**: Store FAISS embeddings with metadata
6. **processing_logs**: Track processing steps and errors

## Pipeline Flow Diagram

```
File Upload → Hash Generation → Hash Check → Metadata Extraction → Metadata Check → Text Extraction → Semantic Chunking → FAISS Embedding → Complete
     ↓              ↓              ↓              ↓                ↓              ↓                ↓                ↓
   Validate    Generate SHA-256  Check DB    Extract Title,     Check for     Extract PDF     Create Topic-    Store Vectors
   File Type   Hash of Content   for Match   Authors, Year,    Metadata      Text Content    Based Chunks    in FAISS Index
              ↓              ↓              ↓              ↓                ↓              ↓                ↓
            Hash Exists?   No → Continue   Extract DOI,      Duplicate?     Clean Text     Group by        Update Search
            Yes → Delete                    Journal, etc.     Yes → Delete   Content        Research Areas   Index
```

## Error Handling & Logging

### Error Scenarios
1. **File Corruption**: Invalid PDF, corrupted content
2. **Text Extraction Failure**: Unreadable text, OCR required
3. **Metadata Extraction Failure**: Missing or malformed metadata
4. **Chunking Failure**: Insufficient text for meaningful chunks
5. **Embedding Generation Failure**: Model errors, memory issues

### Logging Strategy
- **Processing Logs**: Track each step completion
- **Error Logs**: Detailed error information with context
- **Performance Logs**: Processing time for each step
- **Audit Logs**: File processing decisions and reasons

## Testing Strategy

### Unit Tests
- Hash generation and validation
- Metadata extraction accuracy
- Text cleaning and validation
- Chunking algorithm quality
- Embedding generation consistency

### Integration Tests
- End-to-end pipeline execution
- Database operations
- FAISS index updates
- Error handling scenarios

### Validation Tests
- Duplicate detection accuracy
- Text quality assessment
- Chunk semantic coherence
- Embedding similarity relevance

## Performance Considerations

### Optimization Targets
- **Processing Speed**: Target < 30 seconds per publication
- **Memory Usage**: Efficient chunking to avoid memory overflow
- **Storage Efficiency**: Compressed embeddings and metadata
- **Search Performance**: FAISS index optimization for fast retrieval

### Scalability
- **Batch Processing**: Process multiple files concurrently
- **Incremental Updates**: Add new publications without reprocessing
- **Index Maintenance**: Regular FAISS index optimization

## Next Steps

1. **Phase 1**: Implement hash validation and duplicate detection
2. **Phase 2**: Build metadata extraction system
3. **Phase 3**: Develop text extraction and cleaning
4. **Phase 4**: Implement semantic chunking algorithms
5. **Phase 5**: Integrate FAISS embedding storage
6. **Phase 6**: End-to-end testing and optimization

## Success Metrics

- **Duplicate Detection Accuracy**: > 99% correct identification
- **Metadata Extraction Accuracy**: > 95% correct field extraction
- **Text Extraction Quality**: > 90% readable content
- **Chunking Relevance**: > 85% semantically coherent chunks
- **Processing Speed**: < 30 seconds per publication
- **Storage Efficiency**: < 1MB per publication processed
