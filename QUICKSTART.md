# Quick Start Guide - Michael Levin RAG System

## 🚀 Get Up and Running in 5 Minutes

This guide will get you from zero to a working RAG system in just a few steps.

## Prerequisites

- **Python 3.9+** installed
- **uv** package manager ([install here](https://docs.astral.sh/uv/))
- **OpenAI API key** ([get one here](https://platform.openai.com/api-keys))

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd michael-levin-qa-engine-neu1

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

## Step 2: Configure API Keys

```bash
# Create environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

## Step 3: Add Your First PDF

```bash
# Copy a scientific PDF to the input directory
cp your_research_paper.pdf SCIENTIFIC_PUBLICATION_PIPELINE/step_01_raw/
```

## Step 4: Run the Pipeline

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

## Step 5: Start the Chat Interface

```bash
# Go back to root directory
cd ..

# Start Streamlit
uv run streamlit run streamlit_app.py
```

## Step 6: Start Chatting!

1. Open your browser to `http://localhost:8501`
2. Ask questions about the research in your PDF
3. Get responses with citations to specific sections
4. Click on PDF links to view source materials

## 🎯 What Just Happened?

### Pipeline Processing:
1. **Hash Validation**: Your PDF was checked for duplicates
2. **Metadata Extraction**: Title, authors, DOI were extracted
3. **Text Extraction**: PDF content was converted to searchable text
4. **Metadata Enrichment**: Additional data was fetched from Crossref
5. **Semantic Chunking**: Text was split into research-relevant sections
6. **Embedding Generation**: Each chunk was converted to a vector
7. **Archive**: Processed files were organized by year

### RAG System:
- Your PDF chunks are now searchable via semantic similarity
- The AI can find relevant sections and cite them
- Responses include links to source materials

## 🔧 Common Issues & Solutions

### "No embeddings available"
```bash
# Run the embedding step
cd SCIENTIFIC_PUBLICATION_PIPELINE
uv run python step_06_consolidated_embedding.py
```

### "Pipeline progress queue issues"
```bash
# Restore the pipeline state
uv run python restore_progress_queue.py
```

### "OpenAI API key not found"
```bash
# Check your .env file
cat .env
# Should contain: OPENAI_API_KEY=your_key_here
```

## 📚 Next Steps

### Add More PDFs:
```bash
# Copy multiple PDFs at once
cp *.pdf SCIENTIFIC_PUBLICATION_PIPELINE/step_01_raw/
# Then run the pipeline again
```

### Customize the System:
- Modify chunking strategies in `step_05_semantic_chunker_split.py`
- Adjust similarity thresholds in `streamlit_app.py`
- Change embedding models in `step_06_consolidated_embedding.py`

### Monitor Performance:
```bash
# Check processing logs
tail -f SCIENTIFIC_PUBLICATION_PIPELINE/*.log

# View pipeline status
cat SCIENTIFIC_PUBLICATION_PIPELINE/pipeline_progress_queue.json
```

## 🎉 You're All Set!

You now have a working RAG system that can:
- ✅ Process scientific PDFs automatically
- ✅ Create semantic embeddings for search
- ✅ Provide intelligent responses with citations
- ✅ Link to source materials and DOIs
- ✅ Handle multiple documents efficiently

## 📖 Learn More

- **README.md**: Comprehensive project overview
- **DESIGN_DOCUMENT.md**: System design and architecture
- **SCIENTIFIC_PUBLICATIONS_PIPELINE.md**: Detailed pipeline documentation
- **ARCHITECTURE.md**: Technical architecture and decisions

## 🤝 Need Help?

- Check the logs in `SCIENTIFIC_PUBLICATION_PIPELINE/*.log`
- Review the troubleshooting section above
- Examine the pipeline status with `pipeline_progress_queue.json`
- Use `restore_progress_queue.py` for state recovery

---

*Happy researching! 🧠✨*
