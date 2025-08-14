# Michael Levin Scientific Publications RAG System

An AI-powered system that creates a conversational interface with Michael Levin using Retrieval-Augmented Generation (RAG) based on his scientific publications. The system processes PDFs through a comprehensive pipeline and provides intelligent responses with source citations.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd michael-levin-qa-engine-neu1
   ```

2. **Create and activate virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -e .
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Run the Streamlit app**
   ```bash
   uv run streamlit run streamlit_app.py
   ```

## 🔬 What This System Does

This system processes scientific publications (PDFs) through a 7-step pipeline to create a comprehensive knowledge base, then provides a conversational interface where users can ask questions about Michael Levin's research and receive answers based on his actual publications.

### Key Features
- **PDF Processing Pipeline**: Automatically processes scientific PDFs with duplicate detection
- **Semantic Chunking**: Intelligently splits documents into meaningful sections
- **Vector Embeddings**: Creates FAISS indices for fast semantic search
- **RAG Interface**: Chat with an AI that cites specific papers and sections
- **Source Citations**: Every response includes links to source PDFs and DOIs

## 📊 Pipeline Overview

The system processes PDFs through these steps:

1. **Hash Validation** (`step_01`) - Detects duplicate files
2. **Metadata Extraction** (`step_02`) - Extracts title, authors, DOI, etc.
3. **Text Extraction** (`step_03`) - Converts PDFs to searchable text
4. **Metadata Enrichment** (`step_04`) - Enhances with Crossref data
5. **Semantic Chunking** (`step_05`) - Splits text into meaningful chunks
6. **Consolidated Embedding** (`step_06`) - Creates FAISS vector index
7. **Archive** (`step_07`) - Moves processed files to archive

## 🏗️ Project Structure

```
michael-levin-qa-engine-neu1/
├── SCIENTIFIC_PUBLICATION_PIPELINE/     # Core processing pipeline
│   ├── step_01_raw/                     # Input PDF directory
│   ├── step_02_metadata/                # Extracted metadata
│   ├── step_03_extracted_text/          # Raw text from PDFs
│   ├── step_04_enriched_metadata/       # Enhanced metadata
│   ├── step_05_semantic_chunks/         # Semantic text chunks
│   ├── step_06_faiss_embeddings/        # FAISS vector indices
│   └── step_07_archive/                 # Processed PDFs
├── streamlit_app.py                      # Main web interface
├── pyproject.toml                       # Project configuration
└── .env                                 # Environment variables
```

## 🎯 Usage

### Processing New PDFs

1. **Add PDFs to the pipeline**
   ```bash
   # Copy PDFs to the input directory
   cp your_paper.pdf SCIENTIFIC_PUBLICATION_PIPELINE/step_01_raw/
   ```

2. **Run the pipeline**
   ```bash
   cd SCIENTIFIC_PUBLICATION_PIPELINE
   
   # Run each step in sequence
   uv run python step_01_unique_hashcode_validator.py
   uv run python step_02_metadata_extractor.py
   uv run python step_03_text_extractor.py
   uv run python step_04_optional_metadata_enrichment.py
   uv run python step_05_semantic_chunker_split.py
   uv run python step_06_consolidated_embedding.py
   uv run python step_07_move_to_archive.py
   ```

### Using the Chat Interface

1. **Start the Streamlit app**
   ```bash
   uv run streamlit run streamlit_app.py
   ```

2. **Ask questions** about Michael Levin's research
3. **Get responses** with citations to specific papers and sections
4. **Access source materials** through provided links

## 🔧 Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

### Type Checking
```bash
uv run mypy .
```

## 📚 How It Works

### 1. Document Processing
- PDFs are validated for uniqueness using SHA-256 hashes
- Metadata is extracted (title, authors, DOI, publication year)
- Full text is extracted and cleaned
- Text is semantically chunked into research-relevant sections

### 2. Vector Embedding
- Each text chunk is converted to a vector embedding using OpenAI's text-embedding-3-large
- Embeddings are stored in a FAISS index for fast similarity search
- All chunks from all documents are consolidated into a single index

### 3. RAG System
- User queries are converted to embeddings
- FAISS finds the most similar text chunks
- Relevant context is retrieved and sent to OpenAI's GPT model
- Responses are generated with source citations

## 🚨 Troubleshooting

### Common Issues

**"No embeddings available"**
- Run the embedding step: `uv run python step_06_consolidated_embedding.py`

**"Pipeline progress queue issues"**
- Use the restore script: `uv run python restore_progress_queue.py`

**"OpenAI API key not found"**
- Check your `.env` file has `OPENAI_API_KEY=your_key_here`

### Logs
Check the log files in `SCIENTIFIC_PUBLICATION_PIPELINE/` for detailed error information:
- `consolidated_embedding.log`
- `semantic_chunking_split.log`
- `text_extraction.log`
- `metadata_enrichment.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `uv run pytest`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Michael Levin for his groundbreaking research
- OpenAI for the embedding and language models
- FAISS for efficient vector similarity search
- Streamlit for the web interface framework

---

*This system transforms scientific publications into an interactive knowledge base, making research accessible through natural conversation.*
