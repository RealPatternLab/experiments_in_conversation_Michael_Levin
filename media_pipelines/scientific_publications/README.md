# Scientific Publications Pipeline

This pipeline processes research papers, PDFs, and academic publications to create RAG-ready semantic chunks and Q&A pairs for fine-tuning.

## Purpose
Transform academic publications into structured, searchable knowledge that can be used for:
- Retrieval-Augmented Generation (RAG) applications
- Fine-tuning language models with Q&A pairs
- Academic research and citation analysis

## Data Flow
```
Raw PDFs → Text Extraction → Metadata Extraction → Semantic Chunking → Vector Embeddings → RAG Index
```

## Input Requirements
- **File Types**: PDF files containing research papers
- **Content**: Academic publications, research articles, conference papers
- **Metadata**: Author information, publication dates, DOIs, journal names

## Processing Steps
1. **Text Extraction**: Extract raw text from PDF files
2. **Metadata Extraction**: Parse author, title, DOI, publication date, journal
3. **Semantic Chunking**: Break text into meaningful, contextually coherent chunks
4. **Vector Embeddings**: Generate embeddings for each chunk using OpenAI text-embedding-3-large
5. **Index Creation**: Store chunks and embeddings in FAISS vector database

## Output Structure
- **Semantic Chunks**: Contextually coherent text segments
- **Metadata**: Rich metadata for each chunk
- **Vector Embeddings**: 3072-dimensional vectors for similarity search
- **RAG Index**: FAISS index for fast retrieval

## Tools Required
- PDF text extraction tools
- Metadata parsing utilities
- Semantic chunking algorithms
- Vector embedding generation
- FAISS index management

## Directory Structure

```
scientific_publications/
├── pipeline/                                    # Pipeline orchestration scripts
│   ├── process_scientific_publications.py      # Main pipeline script
│   └── logs/                                   # Pipeline execution logs
├── tools/                                       # Processing tools
│   ├── sort_by_file_type.py                    # File organization
│   ├── sanitize_files.py                       # File sanitization
│   ├── extract_text_from_pdf.py                # Text extraction
│   ├── extract_metadata_from_pdf.py            # Metadata extraction
│   ├── enrich_metadata_with_gemini.py          # AI metadata enrichment
│   ├── enrich_metadata_with_crossref.py        # API metadata enrichment
│   ├── chunk_extracted_text.py                 # Semantic chunking
│   └── embed_semantic_chunks_faiss.py         # Vector embeddings
└── data/                                        # Data directories
    ├── source_data/                             # Input and intermediate data
    │   ├── raw/                                 # Raw PDF files
    │   ├── archive/                             # Archived raw files
    │   ├── DLQ/                                 # Dead letter queue (non-PDFs)
    │   └── preprocessed/                        # Processed files
    │       └── sanitized/                       # Sanitized files
    │           └── pdfs/                        # Sanitized PDFs
    └── transformed_data/                        # Final processed data
        ├── text_extraction/                     # Extracted text files
        ├── metadata_extraction/                 # Metadata files
        ├── semantic_chunking/                   # Semantic chunk files
        └── vector_embeddings/                   # FAISS index & embeddings
```

## Configuration
- Chunk size parameters
- Embedding model settings
- Index configuration
- Quality thresholds 