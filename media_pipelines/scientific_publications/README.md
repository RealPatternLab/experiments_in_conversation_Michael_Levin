# Scientific Publications Pipeline

This pipeline processes research papers, PDFs, and academic publications to create RAG-ready semantic chunks and Q&A pairs for fine-tuning.

## Purpose
Transform academic publications into structured, searchable knowledge that can be used for:
- Retrieval-Augmented Generation (RAG) applications
- Fine-tuning language models with Q&A pairs
- Academic research and citation analysis

## Pipeline Overview

The Scientific Publications Pipeline processes PDF scientific papers through a structured workflow with **early deduplication** to prevent processing duplicate papers:

### **Pipeline Flow:**

1. **File Organization & Archiving** (`01_sort_and_archive_incoming_files.py`)
   - Sorts incoming files by type
   - Archives all files with timestamps
   - Routes non-PDFs to DLQ

2. **File Sanitization** (`02_detect_corruption_and_sanitize_pdfs.py`)
   - Detects corrupted PDFs
   - Sanitizes filenames
   - Moves sanitized PDFs to preprocessed directory

3. **Early Deduplication** (Critical Cost-Saving Step)
   - **Quick Metadata Extraction** (`03_extract_quick_metadata_with_gemini.py`)
     - Uses Gemini Pro to extract metadata from first 3 pages
     - Works across all publication formats
     - Fast and reliable for deduplication
   - **Duplicate Detection** (`04_deduplicate_pdfs_and_move_to_dlq.py`)
     - Identifies duplicates using intelligent similarity matching
     - Moves duplicates to DLQ before expensive processing
     - **Saves massive processing costs** downstream

4. **Content Processing** (Only Unique PDFs)
   - **Full Text Extraction** (`05_extract_full_text_content_from_pdfs.py`)
     - Extracts complete text from unique PDFs only
     - Implements multiple extraction strategies
   - **Metadata Extraction** (`06_extract_metadata_from_extracted_text.py`)
     - Extracts metadata from extracted text files
     - Rule-based extraction for titles, authors, abstracts, DOIs

5. **Metadata Enrichment** (`08_enrich_metadata_with_crossref_api.py`)
   - Enhances metadata using CrossRef API
   - Fetches publication details and citations

6. **Semantic Processing**
   - **Chunking** (`07_create_semantic_chunks_from_text.py`)
     - Creates semantically meaningful text chunks
     - Maintains document structure and context
   - **Vector Embeddings** (`09_generate_vector_embeddings_for_chunks.py`)
     - Generates FAISS vector embeddings
     - Enables semantic search and retrieval

7. **Search & Retrieval** (`10_search_and_retrieve_chunks_from_vector_db.py`)
   - Provides semantic search functionality
   - Returns ranked results with similarity scores

### **Key Benefits of Early Deduplication:**

- **Massive Cost Savings**: No expensive processing of duplicate papers
- **Reliable Detection**: Gemini AI works across all publication formats  
- **Fast Processing**: Only analyzes first 3 pages for deduplication
- **Production Ready**: Prevents resource waste in large-scale processing

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