# Scientific Publications Pipeline - Orchestration Guide

## Overview

This directory contains the core processing tools for the Scientific Publications Pipeline, which transforms raw PDF scientific papers into structured, searchable knowledge bases for RAG (Retrieval-Augmented Generation) applications and fine-tuning datasets.

**Target Audience**: Pipeline Orchestration Agents who need to understand and execute the scientific publications processing workflow.

## Pipeline Architecture

The Scientific Publications Pipeline follows a structured data flow from raw ingestion to vector embeddings, with **early deduplication** to prevent processing duplicate papers:

```
Raw PDFs → File Sorting → Sanitization → Quick Metadata (Gemini) → Deduplication → Full Text Extraction → Metadata → Semantic Chunking → Vector Embeddings → RAG System
```

## Tool Descriptions

### 1. File Management & Ingestion

#### `01_sort_and_archive_incoming_files.py` (formerly `01_sort_files_by_type.py`)
**Purpose**: Initial file routing and organization
**What it does**: 
- Scans the `raw/` directory for incoming files
- Categorizes files by type (PDF, HTML, etc.)
- Moves files to type-specific directories (`raw_pdf/`, `raw_html/`, etc.)
- Archives all files with timestamps for permanent storage
- Routes non-PDF files to DLQ (Dead Letter Queue) for manual review

**Why it's needed**: Ensures proper file organization and prevents data loss during processing

#### `02_detect_corruption_and_sanitize_pdfs.py` (formerly `02_sanitize_pdf_files.py`)
**Purpose**: File corruption detection and name sanitization
**What it does**:
- Detects corrupted or unreadable PDF files
- Sanitizes filenames by removing problematic characters
- Generates safe, timestamped filenames
- Moves sanitized files to `preprocessed/sanitized/pdfs/`
- Removes original files from raw directories

**Why it's needed**: Ensures file integrity and prevents processing errors downstream

### 2. Early Deduplication (Critical Cost-Saving Step)

#### `03_extract_quick_metadata_with_gemini.py` (NEW - Critical Step)
**Purpose**: Extract metadata from first 3 pages using Gemini Pro for deduplication
**What it does**:
- Uses Google Gemini Pro AI to analyze first 3 pages of PDFs
- Extracts title, authors, journal, DOI, publication year, abstract
- Works across all publication formats (reliable where PyPDF2/PyMuPDF failed)
- Creates `*_quick_metadata.json` files for deduplication
- **Fast and reliable** - only processes first 3 pages

**Why it's needed**: **Critical for cost savings** - identifies duplicates before expensive full processing

#### `04_deduplicate_pdfs_and_move_to_dlq.py` (NEW - Critical Step)
**Purpose**: Identify and remove duplicate PDFs before expensive processing
**What it does**:
- Compares quick metadata to identify duplicates
- Uses intelligent similarity matching (title, authors, DOI, journal, year)
- Moves duplicate PDFs to DLQ directory
- Keeps only unique PDFs for full processing
- **Saves massive processing costs** downstream

**Why it's needed**: **Prevents processing the same paper multiple times** - essential for production efficiency

### 3. Content Processing (Only Unique PDFs)

#### `05_extract_full_text_content_from_pdfs.py` (formerly `03_extract_text_from_pdfs.py`)
**Purpose**: Full PDF text extraction from all pages
**What it does**:
- Extracts complete text content from **unique PDFs only**
- Implements multiple extraction strategies (PyPDF2, pdfplumber, PyMuPDF)
- Handles various PDF formats and structures
- Saves extracted text as `{filename}_extracted_text.txt`
- Skips already processed files to enable incremental processing
- Outputs to `transformed_data/text_extraction/`

**Why it's needed**: Converts PDF content into machine-readable text for further analysis

### 4. Metadata Processing (Two Approaches)

#### `extract_metadata_directly_from_pdf_files.py` (Alternative Approach)
**Purpose**: Extract metadata directly from PDF files
**What it does**:
- Extracts metadata directly from PDF files using PyPDF2, pdfplumber, and PyMuPDF
- Combines results from multiple extraction methods for best coverage
- Creates initial metadata JSON files
- Outputs to `transformed_data/metadata_extraction/`
- **Alternative to Gemini-based extraction**

**Why it's needed**: Provides foundational metadata from PDF structure and properties

#### `06_extract_metadata_from_extracted_text.py` (Alternative Approach)
**Purpose**: Extract metadata from already-extracted text files
**What it does**:
- Analyzes extracted text files for basic metadata
- Uses rule-based extraction for titles, authors, abstracts, DOIs
- Creates initial metadata JSON files
- Outputs to `transformed_data/metadata_extraction/`
- **Alternative to direct PDF metadata extraction**

**Why it's needed**: Useful when you want to analyze text content rather than PDF structure

#### `enhance_metadata_using_gemini_ai.py` (Pipeline Step 6)
**Purpose**: Enhance metadata using Google Gemini Pro
**What it does**:
- Analyzes existing metadata using Gemini Pro AI
- Improves titles, author names, abstracts, and keywords
- Generates missing metadata fields
- Enhances confidence scores
- Outputs to `transformed_data/metadata_extraction/`

**Why it's needed**: Uses AI to improve and complete metadata where extraction was incomplete

#### `08_enrich_metadata_with_crossref_api.py` (Pipeline Step 8)
**Purpose**: Metadata enrichment using external APIs
**What it does**:
- Enhances basic metadata using CrossRef API
- Fetches publication details, citations, and DOIs
- Enriches chunk metadata with publication context
- Creates enhanced metadata files
- Outputs to `transformed_data/metadata_enrichment/`
- Implements "already processed" logic for efficiency

**Why it's needed**: Provides rich, authoritative metadata for better search and retrieval

### 5. Semantic Processing & Enhancement

#### `07_create_semantic_chunks_from_text.py` (formerly `05_chunk_extracted_text.py`)
**Purpose**: Intelligent text chunking for semantic processing
**What it does**:
- Splits extracted text into semantically meaningful chunks
- Uses LLM-based chunking strategies for context preservation
- Maintains document structure and relationships
- Creates chunked data with metadata and context
- Outputs to `transformed_data/semantic_chunking/`
- Skips already processed files for efficiency

**Why it's needed**: Breaks down large documents into manageable, contextually coherent pieces for AI processing

### 6. Vector Processing & Storage

#### `09_generate_vector_embeddings_for_chunks.py` (formerly `07_embed_semantic_chunks_faiss.py`)
**Purpose**: Vector embedding generation and FAISS storage
**What it does**:
- Generates vector embeddings for semantic chunks
- Uses OpenAI embeddings for high-quality vector representation
- Stores vectors in FAISS index for fast similarity search
- Creates searchable vector database
- Outputs to `transformed_data/vector_embeddings/`

**Why it's needed**: Enables semantic search and retrieval in the RAG system

#### `10_search_and_retrieve_chunks_from_vector_db.py` (formerly `08_retrieve_relevant_chunks_faiss.py`)
**Purpose**: Semantic search and retrieval from vector database
**What it does**:
- Performs similarity search on vector embeddings
- Retrieves most relevant chunks for given queries
- Supports various search strategies and parameters
- Returns ranked results with similarity scores

**Why it's needed**: Provides the core retrieval functionality for RAG applications

### 7. Legacy & Testing Tools

#### `chunk_models.py` & `chunk_pdfs.py`
**Purpose**: Legacy database-driven chunking tools
**Status**: These are from the old database-based approach and may not be needed in the new file-based pipeline

#### `generate_chunking_prompt.py`
**Purpose**: Generate prompts for semantic chunking
**Status**: Currently commented out in the pipeline as the prompt file already exists

#### `test_*.py` files
**Purpose**: Tool validation and performance testing
**What they do**:
- Test various extraction strategies and parameters
- Validate processing quality and performance
- Ensure pipeline reliability and consistency

## Data Flow

### **NEW Pipeline Flow with Early Deduplication:**

1. **Raw PDFs** (`source_data/raw/`) 
   → **Sorted & Archived** (`source_data/archive/`)
   → **Sanitized** (`source_data/preprocessed/sanitized/pdfs/`)

2. **Sanitized PDFs** 
   → **Quick Metadata** (`transformed_data/quick_metadata/`) via `03_extract_quick_metadata_with_gemini.py`
   → **Deduplication** via `04_deduplicate_pdfs_and_move_to_dlq.py`
   → **Duplicates moved to DLQ** (`source_data/DLQ/`)

3. **Unique PDFs Only** 
   → **Full Text Extraction** (`transformed_data/text_extraction/`) via `05_extract_full_text_content_from_pdfs.py`
   → **Metadata Processing** via various approaches
   → **Semantic Chunks** (`transformed_data/semantic_chunking/`)
   → **Vector Embeddings** (`transformed_data/vector_embeddings/`)

### **Key Benefits of Early Deduplication:**

- **Massive Cost Savings**: No expensive processing of duplicate papers
- **Reliable Detection**: Gemini AI works across all publication formats
- **Fast Processing**: Only analyzes first 3 pages for deduplication
- **Production Ready**: Prevents resource waste in large-scale processing

## Usage for Pipeline Orchestration Agents

### **Processing a New PDF (NEW Pipeline with Deduplication):**

1. **Place PDF in raw directory**: `source_data/raw/`
2. **Run file sorting**: `python 01_sort_and_archive_incoming_files.py`
3. **Sanitize files**: `python 02_detect_corruption_and_sanitize_pdfs.py`
4. **Extract quick metadata**: `python 03_extract_quick_metadata_with_gemini.py`
5. **Deduplicate PDFs**: `python 04_deduplicate_pdfs_and_move_to_dlq.py`
6. **Extract full text**: `python 05_extract_full_text_content_from_pdfs.py`
7. **Process metadata**: Choose approach (PDF-based, text-based, or Gemini-enhanced)
8. **Create chunks**: `python 07_create_semantic_chunks_from_text.py`
9. **Generate embeddings**: `python 09_generate_vector_embeddings_for_chunks.py`

### **Alternative Approaches (Metadata Processing):**

**PDF-Based Metadata:**
- Use `extract_metadata_directly_from_pdf_files.py` after step 6

**Text-Based Metadata:**
- Use `06_extract_metadata_from_extracted_text.py` after step 6

**Gemini Enhancement:**
- Use `enhance_metadata_using_gemini_ai.py` after metadata extraction

### Key Principles

- **Early Deduplication**: Identify duplicates before expensive processing
- **Incremental Processing**: Tools skip already processed files
- **Data Integrity**: Each step validates input and output
- **Error Handling**: Corrupted files are detected and quarantined
- **Scalability**: Tools can process single files or entire directories
- **Reproducibility**: All processing steps are deterministic and logged

### Configuration

Each tool supports:
- `--max-files N`: Process only N files (useful for testing)
- `--dry-run`: Preview changes without making them
- `--verbose`: Detailed logging and progress information

## Dependencies

- **PDF Processing**: PyPDF2, pdfplumber, PyMuPDF
- **AI Integration**: Google Gemini Pro (for reliable metadata extraction)
- **Vector Operations**: FAISS, numpy
- **API Integration**: OpenAI, CrossRef, Google Gemini
- **Data Handling**: pandas, json, pathlib

## Output Structure

```
transformed_data/
├── quick_metadata/          # Quick metadata for deduplication (NEW)
├── text_extraction/         # Full extracted text files (unique PDFs only)
├── metadata_extraction/     # Basic metadata JSON files
├── semantic_chunking/       # Chunked text with context
├── metadata_enrichment/     # Enhanced metadata with API data
└── vector_embeddings/       # FAISS index and vector files

source_data/
├── DLQ/                     # Duplicate PDFs moved here (NEW)
└── preprocessed/sanitized/pdfs/  # Unique PDFs for processing
```

## Tool Naming Convention

- **Numbered tools** (`01_`, `02_`, etc.): Core pipeline tools in execution order with descriptive names
- **Named tools**: Alternative or specialized tools with clear, action-oriented names
- **Legacy tools**: Old database-driven tools that may not be needed

## Current Status

✅ **All Required Tools Available**: The pipeline now has all the tools it expects
✅ **Early Deduplication Implemented**: Critical cost-saving step added
✅ **Self-Documenting Names**: All tools have descriptive names that explain their purpose
✅ **No Duplication**: All tools serve distinct purposes
✅ **Production Ready**: Pipeline can be run end-to-end with duplicate prevention

## For Pipeline Orchestration Agents

This guide provides everything you need to:
- **Understand** what each tool does and why it's needed
- **Execute** the pipeline step-by-step with early deduplication
- **Troubleshoot** issues by understanding the data flow
- **Customize** processing by understanding tool capabilities
- **Monitor** progress through clear logging and status indicators

### **Critical Success Factors:**

1. **Early Deduplication**: Run steps 3-4 before expensive processing
2. **Gemini Integration**: Ensure GOOGLE_API_KEY is set for reliable metadata extraction
3. **DLQ Management**: Monitor DLQ directory for moved duplicates
4. **Resource Efficiency**: Only process unique PDFs for full extraction

The tools are designed to be **self-documenting** through their names, making them accessible to both human operators and automated pipeline orchestration systems.

This pipeline transforms unstructured PDF scientific papers into a structured, searchable knowledge base that can power advanced AI applications, research tools, and knowledge discovery systems, while **preventing expensive duplicate processing**. 