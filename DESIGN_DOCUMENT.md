# Michael Levin Scientific Publications RAG System - Design Document

## Project Overview

The Michael Levin Scientific Publications RAG System is an AI-powered system that creates a conversational interface with Michael Levin using Retrieval-Augmented Generation (RAG) based on his scientific publications. The system processes PDFs through a comprehensive 7-step pipeline to build a knowledge base, then provides intelligent responses with source citations.

## System Architecture

### 1. Document Processing Pipeline
- **PDF Input**: Scientific publications in PDF format
- **Duplicate Detection**: SHA-256 hash validation to prevent reprocessing
- **Metadata Extraction**: Title, authors, DOI, publication year, journal
- **Text Extraction**: Full text extraction using PyPDF2 and Gemini
- **Metadata Enrichment**: Crossref API integration for additional data
- **Semantic Chunking**: Intelligent text segmentation by research topics
- **Vector Embedding**: OpenAI text-embedding-3-large for semantic search
- **Archive Management**: Organized storage of processed documents

### 2. RAG System
- **Vector Database**: FAISS index for fast similarity search
- **Retrieval Engine**: Semantic search across all embedded chunks
- **Response Generation**: OpenAI GPT models with retrieved context
- **Source Attribution**: PDF citations and DOI links for every response

### 3. User Interface
- **Streamlit Web Application**: Interactive chat interface
- **Conversation Management**: Chat history and context tracking
- **Response Display**: Formatted responses with expandable source sections
- **Visual Design**: Dark theme with comprehensive styling

## Technical Implementation

### Core Technologies
- **Python 3.9+**: Primary programming language
- **OpenAI API**: GPT models and text embeddings
- **FAISS**: Vector similarity search and storage
- **Streamlit**: Web application framework
- **PyPDF2**: PDF text extraction
- **Google Gemini**: Alternative text extraction method
- **Crossref API**: Scientific publication metadata enrichment

### Pipeline Implementation
```python
# Pipeline step execution
class ScientificPublicationPipeline:
    def step_01_hash_validation(self)      # Duplicate detection
    def step_02_metadata_extraction(self)  # Extract publication info
    def step_03_text_extraction(self)      # PDF to text conversion
    def step_04_metadata_enrichment(self)  # Crossref API integration
    def step_05_semantic_chunking(self)    # Intelligent text splitting
    def step_06_consolidated_embedding(self) # Vector index creation
    def step_07_archive_management(self)   # File organization
```

### RAG Implementation
```python
class FAISSRetriever:
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = embeddings_dir
        self.indices = {}
        self.metadata = {}
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5):
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # FAISS similarity search
        similarity_scores, indices = index.search(query_embedding, top_k)
        
        # Return relevant chunks with metadata
        return self.format_results(similarity_scores, indices)
    
    def generate_embedding(self, text: str) -> List[float]:
        # OpenAI text-embedding-3-large
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
```

### Semantic Chunking Strategy
```python
class SemanticChunker:
    def chunk_by_semantic_units(self, text: str) -> List[TextChunk]:
        # Research topic-based chunking
        sections = self.detect_research_sections(text)
        
        # Fallback to single chunk if no sections detected
        if not sections:
            sections = [{
                'title': 'Full Document',
                'text': text,
                'start_pos': 0,
                'end_pos': len(text)
            }]
        
        return sections
    
    def detect_research_sections(self, text: str) -> List[Dict]:
        # Multiple regex patterns for section detection
        patterns = [
            r'^(\d+\.\s+[A-Z][^:\n]*(?::[^:\n]*)?)',  # 1. Section Title
            r'^([A-Z][A-Z\s]+(?:[A-Z][a-z][^:\n]*)?)', # ALL CAPS SECTION
            r'^(Abstract|Introduction|Methods|Results|Discussion|Conclusion)',
            r'^(Background|Materials|Experimental|Analysis|Summary)'
        ]
        # Apply patterns and extract sections
```

## Data Flow

### 1. Document Ingestion
```
PDF Upload → Hash Generation → Duplicate Check → Pipeline Entry
     ↓              ↓              ↓              ↓
  File Copy    SHA-256 Hash   Database Query   Status Update
```

### 2. Processing Pipeline
```
Step 01 → Step 02 → Step 03 → Step 04 → Step 05 → Step 06 → Step 07
  Hash      Meta      Text      Enrich    Chunk     Embed     Archive
  Valid     Extract   Extract   Meta      Text      Vector    Files
```

### 3. RAG System
```
User Query → Embedding → FAISS Search → Context Retrieval → GPT Response → Source Citations
     ↓           ↓           ↓              ↓              ↓              ↓
  Natural    Vector     Similarity     Relevant      Generated      PDF Links
  Language   Convert    Search        Chunks        Response       & DOIs
```

## Project Structure

```
michael-levin-qa-engine-neu1/
├── SCIENTIFIC_PUBLICATION_PIPELINE/           # Core processing pipeline
│   ├── step_01_raw/                           # Input PDF directory
│   ├── step_01_unique_hashcode_validator.py   # Hash validation
│   ├── step_02_metadata/                      # Extracted metadata
│   ├── step_02_metadata_extractor.py          # Metadata extraction
│   ├── step_03_extracted_text/                # Raw text from PDFs
│   ├── step_03_text_extractor.py              # Text extraction
│   ├── step_04_enriched_metadata/             # Enhanced metadata
│   ├── step_04_optional_metadata_enrichment.py # Crossref enrichment
│   ├── step_05_semantic_chunks/               # Semantic text chunks
│   ├── step_05_semantic_chunker_split.py      # Text chunking
│   ├── step_06_faiss_embeddings/              # FAISS vector indices
│   ├── step_06_consolidated_embedding.py      # Vector embedding
│   ├── step_07_archive/                       # Processed PDFs
│   ├── step_07_move_to_archive.py             # Archive management
│   ├── pipeline_progress_queue.py              # Pipeline state tracking
│   └── restore_progress_queue.py               # State recovery
├── streamlit_app.py                            # Main web interface
├── pyproject.toml                              # Project configuration
├── .env                                        # Environment variables
└── thinking_box.gif                            # UI assets
```

## Data Models

### PublicationMetadata
```python
@dataclass
class PublicationMetadata:
    file_hash: str                    # SHA-256 hash for uniqueness
    filename: str                     # Original PDF filename
    title: str                        # Publication title
    authors: List[str]                # Author list
    publication_year: int             # Year of publication
    doi: Optional[str]                # Digital Object Identifier
    journal: str                      # Journal/conference name
    abstract: Optional[str]           # Publication abstract
    processing_timestamp: datetime    # When processed
    crossref_data: Optional[Dict]    # Enhanced metadata from Crossref
```

### TextChunk
```python
@dataclass
class TextChunk:
    chunk_id: str                     # Unique chunk identifier
    source_file: str                  # Source PDF filename
    chunk_text: str                   # Actual text content
    chunk_type: str                   # Type (topic, method, result, etc.)
    chunk_position: int               # Position in document
    research_topics: List[str]        # Related research areas
    key_terms: List[str]              # Important terms and concepts
    embedding_vector: List[float]     # Vector representation
    created_timestamp: datetime       # When created
```

### ProcessingStatus
```python
@dataclass
class ProcessingStatus:
    doc_id: str                       # Document identifier
    current_step: str                 # Current pipeline step
    status: str                       # Status (pending, processing, complete, failed)
    step_results: Dict[str, Any]      # Results from each step
    error_messages: List[str]         # Any error messages
    processing_duration: float        # Total processing time
    created_timestamp: datetime       # When processing started
    updated_timestamp: datetime       # Last update time
```

## Pipeline Flow Diagram

```
PDF Upload → Hash Validation → Metadata Extraction → Text Extraction → Metadata Enrichment → Semantic Chunking → Vector Embedding → Archive
     ↓              ↓                ↓                ↓                ↓                ↓                ↓                ↓
  Copy to      Generate SHA-256   Extract Title,   Extract PDF      Crossref API    Split by         OpenAI           Move to
  step_01      Hash & Check      Authors, Year,   Text Content     Integration     Research          Embeddings      step_07
              Database for       DOI, Journal     Using PyPDF2     for Enhanced    Topics &          & FAISS         Archive
              Duplicates         & Abstract       & Gemini         Metadata        Methods           Index Creation   Directory
```

## Error Handling & Logging

### Error Scenarios
1. **File Corruption**: Invalid PDF, corrupted content
2. **Text Extraction Failure**: Unreadable text, OCR required
3. **Metadata Extraction Failure**: Missing or malformed metadata
4. **Chunking Failure**: Insufficient text for meaningful chunks
5. **Embedding Generation Failure**: OpenAI API errors, rate limits
6. **Pipeline State Corruption**: Progress queue issues

### Logging Strategy
- **Processing Logs**: Track each step completion with timestamps
- **Error Logs**: Detailed error information with context and stack traces
- **Performance Logs**: Processing time for each step and overall pipeline
- **Audit Logs**: File processing decisions, duplicate detection results
- **Recovery Logs**: State restoration and error recovery attempts

### Recovery Mechanisms
```python
# Progress queue restoration
class ProgressQueueRestorer:
    def restore_from_processed_files(self):
        # Scan existing output directories
        # Reconstruct pipeline state
        # Update progress queue
        
    def validate_pipeline_state(self):
        # Check consistency between files and state
        # Identify missing or corrupted data
        # Suggest recovery actions
```

## Testing Strategy

### Unit Tests
- Hash generation and validation accuracy
- Metadata extraction from various PDF formats
- Text cleaning and validation quality
- Chunking algorithm semantic coherence
- Embedding generation consistency
- FAISS search relevance

### Integration Tests
- End-to-end pipeline execution
- Progress queue state management
- Error handling and recovery scenarios
- Cross-step data consistency

### Validation Tests
- Duplicate detection accuracy
- Text quality assessment
- Chunk semantic coherence
- Embedding similarity relevance
- RAG response quality and citation accuracy

## Performance Considerations

### Optimization Targets
- **Processing Speed**: Target < 60 seconds per publication
- **Memory Usage**: Efficient chunking to avoid memory overflow
- **Storage Efficiency**: Compressed embeddings and metadata
- **Search Performance**: FAISS index optimization for fast retrieval
- **API Efficiency**: Batch processing and rate limit management

### Scalability
- **Batch Processing**: Process multiple files concurrently where possible
- **Incremental Updates**: Add new publications without reprocessing
- **Index Maintenance**: Regular FAISS index optimization
- **State Management**: Robust progress tracking and recovery

## Success Metrics

- **Duplicate Detection Accuracy**: > 99% correct identification
- **Metadata Extraction Accuracy**: > 95% correct field extraction
- **Text Extraction Quality**: > 90% readable content
- **Chunking Relevance**: > 85% semantically coherent chunks
- **Processing Speed**: < 60 seconds per publication
- **Storage Efficiency**: < 2MB per publication processed
- **RAG Response Quality**: Human evaluation scores (1-5 scale)
- **Citation Accuracy**: > 95% correct source attribution

## Current Implementation Status

### ✅ Completed Features
- **7-step processing pipeline** with comprehensive error handling
- **Consolidated embedding system** replacing fragmented approach
- **FAISS vector search** with semantic similarity
- **Streamlit web interface** with dark theme and responsive design
- **Source citation system** with PDF links and DOI references
- **Progress tracking** with state recovery mechanisms
- **Comprehensive logging** and error reporting

### 🔄 Recent Improvements
- **Consolidated embeddings**: Single comprehensive index instead of fragmented directories
- **Enhanced UI**: Black theme, improved sidebar, better text visibility
- **Robust error handling**: Progress queue restoration and validation
- **Performance optimization**: FAISS search implementation and similarity thresholds

### 🚧 Areas for Enhancement
- **Batch processing**: Parallel execution of pipeline steps
- **Advanced chunking**: Machine learning-based semantic segmentation
- **Multi-modal support**: Image and diagram processing
- **Real-time updates**: Live pipeline monitoring and status updates

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd michael-levin-qa-engine-neu1
   ```

2. **Set up environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

3. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add OpenAI API key
   ```

4. **Process publications**
   ```bash
   # Add PDFs to step_01_raw/
   # Run pipeline steps sequentially
   ```

5. **Start the interface**
   ```bash
   uv run streamlit run streamlit_app.py
   ```

## Contributing

- Follow PEP 8 coding standards
- Write comprehensive tests for new features
- Update documentation for API changes
- Use conventional commit messages
- Test pipeline changes thoroughly before committing

---

*This document reflects the current implementation of the Michael Levin Scientific Publications RAG System as of the latest update.* 