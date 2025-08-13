# Scientific Publications Pipeline

A comprehensive pipeline for processing scientific publications (PDFs) through multiple stages including metadata extraction, semantic chunking, and vector embedding for RAG applications.

## 🏗️ **Architecture Overview**

The pipeline is designed with a **separation of concerns** approach:

### **Main Pipeline** (`run_scientific_publications_pipeline.py`)
- **Purpose**: Process PDFs through steps 1-8 (data ingestion and transformation)
- **Frequency**: Runs continuously as new files arrive
- **Output**: Processed chunks ready for embedding

### **Embedding Service** (`embed_all_processed_chunks.py`)
- **Purpose**: Create FAISS search index from all processed chunks
- **Frequency**: Runs on schedule (e.g., once per day)
- **Output**: Updated search index for RAG applications

This separation allows for:
- **Real-time processing** of new PDFs
- **Efficient bulk embedding** during off-peak hours
- **Independent scaling** of processing vs. embedding
- **Better operational control** and monitoring

## 🚀 **Quick Start**

### **1. Run the Main Pipeline (Steps 1-8)**
```bash
# Process all PDFs through the pipeline
uv run python3 run_scientific_publications_pipeline.py

# Start from a specific step
uv run python3 run_scientific_publications_pipeline.py --start-from-step 5
```

### **2. Create Search Index (Step 9)**
```bash
# Embed all processed chunks into FAISS index
uv run python3 tools/embed_all_processed_chunks.py

# Use custom directories
uv run python3 tools/embed_all_processed_chunks.py \
    --input-dir data/transformed_data/semantic_chunks \
    --output-dir data/transformed_data/vector_embeddings
```

## 📋 **Pipeline Steps**

### **Data Ingestion & Processing (Steps 1-5)**
1. **Sort and Archive**: Route incoming files, archive all, move PDFs to processing
2. **Sanitize PDFs**: Detect corruption and clean PDF files
3. **Extract Metadata**: Use Gemini to extract quick metadata
4. **Deduplicate**: Remove duplicate PDFs based on metadata
5. **Extract Text**: Convert PDFs to searchable text

### **Data Transformation (Steps 6-8)**
6. **Extract Metadata**: Rule-based metadata extraction from text
7. **Semantic Chunking**: Create meaningful text chunks using Gemini
8. **Enrich Metadata**: Enhance metadata using Crossref and Unpaywall APIs

### **Search Index Creation (Separate Tool)**
9. **Vector Embeddings**: Generate embeddings and create FAISS index

## 📁 **Directory Structure**

```
data/
├── source_data/
│   ├── raw/                    # New PDFs uploaded here
│   ├── raw_pdf/               # PDFs ready for processing
│   ├── archive/               # Archived original files
│   ├── DLQ/                   # Dead Letter Queue (duplicates, errors)
│   └── preprocessed/
│       └── sanitized/
│           └── pdfs/          # Cleaned PDFs
└── transformed_data/
    ├── quick_metadata/        # Gemini-extracted metadata
    ├── extracted_text/        # PDF text content
    ├── metadata_extraction/   # Rule-based metadata
    ├── semantic_chunks/       # Processed text chunks
    ├── metadata_enrichment/   # Enhanced metadata
    └── vector_embeddings/     # FAISS index and embeddings
```

## 🔄 **Data Flow**

1. **PDF Upload** → `raw/` directory
2. **Pipeline Processing** → Steps 1-8 transform PDFs into chunks
3. **Chunk Accumulation** → Chunks ready for embedding
4. **Scheduled Embedding** → Daily update of search index
5. **RAG Application** → Streamlit app searches updated index

## 🎯 **Output Structure**

### **Semantic Chunks**
- JSON files with text chunks and metadata
- Enhanced with information from quick_metadata
- Includes page numbers and semantic context

### **FAISS Index**
- `chunks.index`: Vector search index
- `chunks_embeddings.npy`: Embedding vectors
- `chunks_metadata.pkl`: Comprehensive metadata for retrieval

## 🛠️ **Dependencies**

- **Python 3.8+**
- **uv** for package management
- **OpenAI API** for embeddings
- **Google Gemini API** for metadata extraction and chunking
- **FAISS** for vector similarity search
- **PyPDF2** for PDF processing

## 📊 **Production Workflow**

### **Continuous Processing**
```bash
# Monitor for new PDFs and process continuously
while true; do
    uv run python3 run_scientific_publications_pipeline.py
    sleep 300  # Check every 5 minutes
done
```

### **Scheduled Embedding**
```bash
# Add to crontab for daily updates at 2 AM
0 2 * * * cd /path/to/pipeline && uv run python3 tools/embed_all_processed_chunks.py
```

## 🔍 **Monitoring & Logs**

- **Pipeline logs**: `logs/pipeline.log`
- **Embedding logs**: `logs/embed_semantic_chunks_faiss.log`
- **Individual tool logs**: Available in each tool's output

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Missing API Keys**: Ensure `OPENAI_API_KEY` and `GOOGLE_API_KEY` are set
2. **Path Issues**: Run from the pipeline root directory
3. **Dependencies**: Use `uv run` to ensure correct Python environment

### **Debug Mode**
```bash
# Check individual steps
uv run python3 tools/step01_sort_and_archive_incoming_files.py --help
```

## 📈 **Scaling Considerations**

- **Pipeline**: Can be containerized and auto-scaled
- **Embedding**: Resource-intensive, run during off-peak hours
- **Storage**: Monitor disk usage for accumulated chunks
- **API Limits**: Respect OpenAI and Gemini rate limits

## 🔗 **Integration**

The pipeline integrates with:
- **Streamlit RAG Application**: Search processed publications
- **GitHub**: Source control and deployment
- **Streamlit Cloud**: Hosting and deployment
- **External APIs**: Crossref, Unpaywall for metadata enrichment 