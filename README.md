# Michael Levin Scientific Knowledge Base - Multi-Pipeline RAG System

An AI-powered system that creates a conversational interface with Michael Levin using Retrieval-Augmented Generation (RAG) based on his scientific publications **and videos**. The system processes both PDFs and YouTube content through separate, specialized pipelines and provides intelligent responses with source citations across multiple content types.

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

This system processes **scientific publications (PDFs)**, **scientific videos (YouTube)**, and **conversations/working meetings** through specialized pipelines to create a comprehensive, multimodal knowledge base. Users can ask questions about Michael Levin's research and receive answers based on his actual publications, presentations, and conversations.

### Key Features
- **Triple Pipeline Architecture**: Separate processing for publications, videos, and conversations
- **Data Standardization**: Shared models and base classes ensure consistency across pipelines
- **Multimodal Content**: Text, audio, and visual content processing
- **Semantic Chunking**: Intelligently splits documents and transcripts
- **Vector Embeddings**: Creates FAISS indices for fast semantic search
- **RAG Interface**: Chat with an AI that cites specific sources
- **Source Citations**: Every response includes links to source PDFs, videos with timestamps, and conversation clips

## 📊 Pipeline Overview

### Publications Pipeline (7 Steps)
1. **Hash Validation** (`step_01`) - Detects duplicate files
2. **Metadata Extraction** (`step_02`) - Extracts title, authors, DOI, etc.
3. **Text Extraction** (`step_03`) - Converts PDFs to searchable text
4. **Metadata Enrichment** (`step_04`) - Enhances with Crossref data
5. **Semantic Chunking** (`step_05`) - Splits text into meaningful chunks
6. **Consolidated Embedding** (`step_06`) - Creates FAISS vector index
7. **Archive** (`step_07`) - Moves processed files to archive

### Videos Pipeline (8 Steps)
1. **Playlist Processing** (`step_01`) - Processes YouTube playlist URLs
2. **Video Download** (`step_02`) - Downloads videos and extracts metadata
3. **Enhanced Transcription** (`step_03`) - Creates high-quality transcripts
4. **Semantic Chunking** (`step_04`) - Splits transcripts into meaningful chunks
5. **Frame Extraction** (`step_05`) - Extracts key video frames
6. **Frame-Chunk Alignment** (`step_06`) - Aligns frames with transcript chunks
7. **Consolidated Embedding** (`step_07`) - Creates multimodal FAISS indices
8. **Archive** (`step_08`) - Organizes processed content

### Conversations Pipeline (8 Steps)
1. **Playlist Processing** (`step_01`) - Processes YouTube playlist URLs
2. **Video Download** (`step_02`) - Downloads videos and extracts metadata
3. **Enhanced Transcription** (`step_03`) - Creates transcripts with speaker diarization
4. **Semantic Chunking** (`step_04`) - Extracts Q&A pairs and Levin's insights
5. **Frame Extraction** (`step_05`) - Extracts key video frames
6. **Frame-Chunk Alignment** (`step_06`) - Aligns frames with semantic chunks
7. **Consolidated Embedding** (`step_07`) - Creates FAISS indices for conversations
8. **Archive** (`step_08`) - Organizes processed content

## 🏗️ Project Structure

```
michael-levin-qa-engine-neu1/
├── SCIENTIFIC_PUBLICATION_PIPELINE/     # Publications processing pipeline
│   ├── step_01_raw/                     # Input PDF directory
│   ├── step_02_metadata/                # Extracted metadata
│   ├── step_03_extracted_text/          # Raw text from PDFs
│   ├── step_04_enriched_metadata/       # Enhanced metadata
│   ├── step_05_semantic_chunks/         # Semantic text chunks
│   ├── step_06_faiss_embeddings/        # FAISS vector indices
│   └── step_07_archive/                 # Processed PDFs
├── SCIENTIFIC_VIDEO_PIPELINE/           # Videos processing pipeline
│   ├── formal_presentations_1_on_0/     # Formal presentation content
│   │   ├── step_01_raw/                 # YouTube playlist URLs
│   │   ├── step_02_extracted_playlist_content/ # Downloaded videos
│   │   ├── step_03_transcription/       # Generated transcripts
│   │   ├── step_04_extract_chunks/      # Semantic chunks
│   │   ├── step_05_frames/              # Extracted video frames
│   │   ├── step_06_frame_chunk_alignment/ # Frame-chunk alignment
│   │   ├── step_07_faiss_embeddings/    # Multimodal FAISS indices
│   │   └── step_08_archive/             # Processed content
│   └── Conversations_and_working_meetings_1_on_1/ # Conversations content
│       ├── step_01_raw/                 # YouTube playlist URLs
│       ├── step_02_extracted_playlist_content/ # Downloaded videos
│       ├── step_03_transcription/       # Transcripts with speaker diarization
│       ├── step_04_extract_chunks/      # Q&A pairs and semantic chunks
│       ├── step_05_frames/              # Extracted video frames
│       ├── step_06_frame_chunk_alignment/ # Frame-chunk alignment
│       ├── step_07_faiss_embeddings/    # FAISS indices for conversations
│       └── step_08_archive/             # Processed content
├── streamlit_app.py                      # Main web interface
├── pyproject.toml                       # Project configuration
├── ARCHITECTURE.md                      # System architecture details
├── PIPELINE_EVOLUTION_PLAN.md           # Evolution strategy
└── .env                                 # Environment variables
├── utils/                               # Shared data models and base classes
│   ├── pipeline_data_models.py          # Common data structures
│   ├── base_frame_chunk_aligner.py      # Base classes for pipelines
│   └── README.md                        # Standardization documentation
```

## 🔧 **Data Standardization System**

**NEW**: The system now implements shared data models and base classes to ensure consistency across all pipelines:

### What It Provides
- **Shared Data Models**: Common structures for all pipeline outputs
- **Base Classes**: Reusable functionality for video pipelines
- **Automatic Validation**: Catches data structure issues early
- **Type Safety**: Pydantic models ensure data integrity

### Benefits
- **Reduced Troubleshooting**: Consistent data structures across pipelines
- **Easier Maintenance**: Common code centralized
- **Future-Proof**: Easy to add new pipeline types
- **Quality Assurance**: Automatic validation catches issues early

### Documentation
- **Full Details**: See `utils/README.md` for comprehensive documentation
- **Examples**: Check `utils/conversations_aligner_example.py` for usage patterns
- **Testing**: Run `utils/test_data_validation.py` to see validation in action

## 🎯 Usage

### Processing New Publications

1. **Add PDFs to the publications pipeline**
   ```bash
   # Copy PDFs to the input directory
   cp your_paper.pdf SCIENTIFIC_PUBLICATION_PIPELINE/step_01_raw/
   ```

2. **Run the publications pipeline**
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

### Processing New Videos

1. **Add YouTube URLs to the videos pipeline**
   ```bash
   # Edit the playlist file
   echo "https://www.youtube.com/watch?v=VIDEO_ID" >> SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/step_01_raw/youtube_playlist.txt
   ```

2. **Run the videos pipeline**
   ```bash
   cd SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0
   
   # Run each step in sequence
   uv run python step_01_playlist_processor.py
   uv run python step_02_video_downloader.py
   uv run python step_03_transcription_webhook.py
   uv run python step_04_extract_chunks.py
   uv run python step_05_frame_extractor.py
   uv run python step_06_frame_chunk_alignment.py
   uv run python step_07_consolidated_embedding.py
   uv run python step_08_archive.py
   ```

## 🔄 Evolution Strategy

This system is designed with **separation of concerns** and **evolutionary architecture** in mind:

### Current State: Separate Pipelines
- **Independent development** - each pipeline can evolve separately
- **Version control** - clear separation of concerns
- **Experimental freedom** - try new approaches without risk
- **Fault isolation** - issues in one don't break the other

### Future Evolution: Unified Search
- **Phase 1**: Unified search layer (query-time fusion)
- **Phase 2**: Periodic merge pipeline (unified FAISS index)
- **Phase 3**: Real-time sync (always up-to-date)

See [PIPELINE_EVOLUTION_PLAN.md](PIPELINE_EVOLUTION_PLAN.md) for detailed implementation strategy.

## 🎨 Architecture Highlights

### Embedding Strategy
- **Unified Model**: Both pipelines use OpenAI's `text-embedding-3-large` (3072 dimensions)
- **Consistent Semantic Space**: Compatible FAISS indices for future merging
- **High Quality**: State-of-the-art embeddings for scientific content

### Multimodal Capabilities
- **Text**: Publications and video transcripts
- **Visual**: Video frames with semantic alignment
- **Temporal**: Precise timing information for video content
- **Metadata**: Rich context across all content types

### Search Capabilities
- **Semantic Search**: Find content by meaning, not just keywords
- **Cross-Modal Search**: Find related publications when searching videos
- **Source Attribution**: Every result includes detailed source information
- **Quality Metrics**: Confidence scores and alignment quality indicators

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed system architecture and technical decisions
- **[PIPELINE_EVOLUTION_PLAN.md](PIPELINE_EVOLUTION_PLAN.md)**: Strategic evolution from separate to unified pipelines
- **[QUICKSTART.md](QUICKSTART.md)**: Quick setup and usage guide
- **[SCIENTIFIC_PUBLICATIONS_PIPELINE.md](SCIENTIFIC_PUBLICATIONS_PIPELINE.md)**: Publications pipeline details

## 🚧 Development Status

### ✅ Completed
- **Publications Pipeline**: Full 7-step pipeline operational
- **Videos Pipeline**: Full 8-step pipeline operational
- **Embedding Consistency**: Both pipelines use OpenAI text-embedding-3-large
- **FAISS Indices**: Separate indices for publications and videos

### 🔄 In Progress
- **Phase 1**: Unified search layer development
- **Cross-pipeline testing**: Ensuring compatibility

### 📋 Planned
- **Phase 2**: Periodic merge pipeline (next month)
- **Phase 3**: Real-time sync (long-term)
- **Advanced ranking**: Cross-modal result fusion
- **Knowledge graph**: Cross-references between content types

## 🤝 Contributing

This system is designed for experimentation and evolution. Contributions are welcome in the following areas:

- **Pipeline improvements** for either publications or videos
- **Search algorithms** and ranking improvements
- **Frontend enhancements** for the unified interface
- **Performance optimization** and scalability improvements

## 📄 License

[Add your license information here]

---

*This system represents an evolutionary approach to building a comprehensive scientific knowledge base, starting with separate, specialized pipelines and moving toward unified search capabilities.*
