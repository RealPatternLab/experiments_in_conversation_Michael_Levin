# Scientific Publications Pipeline

This pipeline processes research papers, PDFs, and academic publications to create RAG-ready semantic chunks and Q&A pairs for fine-tuning.

## Purpose
Transform academic publications into structured, searchable knowledge that can be used for:
- Retrieval-Augmented Generation (RAG) applications
- Fine-tuning language models with Q&A pairs
- Academic research and citation analysis

## Quick Start

### **Run Complete Pipeline**
```bash
# From the scientific_publications directory
python3 run_scientific_publications_pipeline.py
```

### **Run from Specific Step**
```bash
# Start from step 5 (text extraction)
python3 run_scientific_publications_pipeline.py --start-from-step 5

# Start from step 7 (semantic chunking)
python3 run_scientific_publications_pipeline.py --start-from-step 7
```

### **Dry Run (Preview)**
```bash
# See what would be executed without running
python3 run_scientific_publications_pipeline.py --dry-run

# Dry run from specific step
python3 run_scientific_publications_pipeline.py --start-from-step 3 --dry-run
```

### **Pipeline Script Features**
- **Default Arguments**: Each step has pre-configured arguments - just run the script!
- **Automatic Directory Creation**: Creates all required directories automatically
- **UV Integration**: Uses `uv run` for proper dependency management
- **Error Handling**: Stops on failure with clear error messages
- **Progress Logging**: Detailed logging for each step
- **Flexible Execution**: Run all steps or start from any specific step

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
     - **Enhanced with automatic metadata enrichment** from quick_metadata
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

### **Enhanced Metadata Enrichment:**

- **Automatic Cross-Reference**: Semantic chunks automatically enriched with metadata from quick_metadata files
- **Rich Citations**: Professional-looking citations with real author names, journals, and publication details
- **DOI Integration**: Automatic DOI linking for papers with available DOIs
- **Page Numbers**: Extracted page numbers for precise citation tracking

## Directory Structure

```
scientific_publications/
├── run_scientific_publications_pipeline.py       # Main pipeline orchestrator
├── tools/                                        # Processing tools
│   ├── step01_sort_and_archive_incoming_files.py
│   ├── step02_detect_corruption_and_sanitize_pdfs.py
│   ├── step03_extract_quick_metadata_with_gemini.py
│   ├── step04_deduplicate_pdfs_and_move_to_dlq.py
│   ├── step05_extract_full_text_content_from_pdfs.py
│   ├── step06_extract_metadata_from_extracted_text.py
│   ├── step07_create_semantic_chunks_from_text.py
│   ├── step08_enrich_metadata_with_crossref_api.py
│   ├── step09_generate_vector_embeddings_for_chunks.py
│   └── chunk_models.py                          # Pydantic models for chunks
├── data/                                         # Data directories (auto-created)
│   ├── source_data/                             # Input and intermediate data
│   │   ├── raw/                                 # Raw incoming files
│   │   ├── raw_pdf/                             # PDFs moved from raw/
│   │   ├── archive/                             # Archived raw files
│   │   ├── DLQ/                                 # Dead letter queue (non-PDFs)
│   │   └── preprocessed/                        # Processed files
│   │       └── sanitized/                       # Sanitized files
│   │           └── pdfs/                        # Sanitized PDFs
│   └── transformed_data/                         # Final processed data
│       ├── quick_metadata/                      # Quick metadata from Gemini
│       ├── text_extraction/                     # Extracted text files
│       ├── metadata_extraction/                 # Rule-based metadata
│       ├── semantic_chunks/                     # Semantic chunk files
│       ├── metadata_enrichment/                 # Crossref/Unpaywall enriched
│       └── vector_embeddings/                   # FAISS index & embeddings
└── logs/                                         # Pipeline execution logs
```

## Data Flow
```
Raw Files → Archive + PDF Routing → Sanitization → Quick Metadata → Deduplication → 
Text Extraction → Metadata Extraction → Semantic Chunking → Metadata Enrichment → 
Vector Embeddings → FAISS Index → RAG Ready!
```

## Input Requirements
- **File Types**: PDF files containing research papers
- **Content**: Academic publications, research articles, conference papers
- **Placement**: Place PDFs in `data/source_data/raw/` directory

## Output Structure
- **Semantic Chunks**: Contextually coherent text segments with enriched metadata
- **Metadata**: Rich metadata including authors, journals, DOIs, publication dates
- **Vector Embeddings**: 3072-dimensional vectors for similarity search
- **RAG Index**: FAISS index for fast retrieval with professional citations 