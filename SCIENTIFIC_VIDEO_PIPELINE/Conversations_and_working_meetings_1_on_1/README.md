# Conversations and Working Meetings Pipeline

This pipeline processes 1-on-1 conversations and working meetings featuring Michael Levin, extracting semantic content and creating RAG-ready outputs.

## 🏗️ **Pipeline Architecture**

This pipeline follows the standard 7-step video processing workflow:

1. **Step 1**: Playlist processing and metadata extraction
2. **Step 2**: Video content download and extraction
3. **Step 3**: Transcription with speaker diarization (AssemblyAI)
4. **Step 4**: Semantic chunking and Q&A generation
5. **Step 5**: Frame extraction (every 60 seconds)
6. **Step 6**: Frame-chunk alignment
7. **Step 7**: FAISS embedding generation

## 🔧 **Data Standardization**

**IMPORTANT**: This pipeline now uses shared data models and base classes for consistency across all video pipelines.

### Shared Data Models
- **Location**: `utils/pipeline_data_models.py`
- **Purpose**: Ensures consistent data structures across all pipelines
- **Benefits**: Automatic validation, type safety, reduced troubleshooting

### Base Classes
- **Location**: `utils/base_frame_chunk_aligner.py`
- **Purpose**: Provides common functionality while allowing pipeline-specific customization
- **Usage**: Inherit from `BaseFrameChunkAligner` for consistent behavior

### Current Status
- **Step 6** (Frame-chunk alignment) is ready for refactoring to use the base class
- **Data validation** can be implemented using the shared models
- **Future steps** should follow the standardized data structure

## 📁 **Directory Structure**

```
Conversations_and_working_meetings_1_on_1/
├── step_01_raw/                    # Playlist metadata and URLs
├── step_02_extracted_playlist_content/  # Downloaded video content
├── step_03_transcription/          # AssemblyAI transcripts with speakers
├── step_04_extract_chunks/         # Semantic chunking and Q&A generation
├── step_05_frames/                 # Extracted video frames
├── step_06_frame_chunk_alignment/  # Frame-chunk alignment (ready for refactoring)
├── step_07_faiss_embeddings/       # Vector embeddings for RAG
├── step_08_cleanup/                # Cleanup and maintenance
├── docs/                           # Pipeline documentation
├── logs/                           # Pipeline execution logs
└── README.md                       # This file
```

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- AssemblyAI API key
- OpenAI API key (for GPT-4 enhancement)
- FFmpeg (for video processing)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Pipeline
```bash
# Run individual steps
uv run python step_01_playlist_processor.py
uv run python step_02_video_downloader.py
uv run python step_03_transcription_webhook.py
uv run python step_04_extract_chunks.py
uv run python step_05_frame_extractor.py
uv run python step_06_frame_chunk_alignment.py
uv run python step_07_consolidated_embedding.py

# Or run the complete pipeline
uv run python run_conversations_pipeline.py
```

## 🔍 **Key Features**

### Speaker Identification System
- Interactive prompts for speaker identification (A, B, C, etc.)
- Stores speaker mappings for consistent labeling
- Identifies Michael Levin vs. other speakers

### Enhanced Semantic Chunking
- Uses GPT-4 to extract Q&A pairs from conversations
- Generates synthetic questions for responses without explicit questions
- Preserves Levin's authentic voice and tone
- Includes full context for standalone understanding

### Timestamp Integration
- Extracts start/stop times from labeled transcripts
- Generates YouTube hyperlinks with precise timestamps
- Enables direct linking to specific conversation segments

### Multi-Modal RAG Preparation
- Aligns video frames with semantic chunks
- Creates visual context for text-based queries
- Prepares content for unified search across all pipelines

## 📊 **Data Outputs**

### Step 4: Enhanced Chunks
- **Location**: `step_04_extract_chunks/enhanced_chunks/`
- **Format**: JSON with Q&A pairs, speaker attribution, timestamps
- **Structure**: Follows shared data model standards

### Step 6: Frame-Chunk Alignment
- **Location**: `step_06_frame_chunk_alignment/`
- **Format**: JSON with aligned frames and semantic content
- **Structure**: Ready for standardization using shared models

### Step 7: FAISS Embeddings
- **Location**: `step_07_faiss_embeddings/`
- **Format**: FAISS index + metadata pickle files
- **Integration**: Unified with other pipelines in Streamlit frontend

## 🔄 **Migration to Standardized Models**

### Phase 1: Data Structure Analysis ✅
- Identified inconsistencies between pipeline outputs
- Created shared data models for standardization

### Phase 2: Base Class Implementation ✅
- Implemented `BaseFrameChunkAligner` abstract class
- Created pipeline-specific extensions

### Phase 3: Pipeline Refactoring (Next)
- Refactor `step_06_frame_chunk_alignment.py` to inherit from base class
- Implement data validation using shared models
- Ensure consistent output structure

### Phase 4: Integration Testing
- Validate outputs against shared data models
- Test Streamlit frontend compatibility
- Verify cross-pipeline consistency

## 🧪 **Testing and Validation**

### Data Validation
```python
from utils.pipeline_data_models import validate_pipeline_data, PipelineType

# Validate pipeline output
is_valid = validate_pipeline_data(data, PipelineType.CONVERSATIONS)
```

### Base Class Testing
```python
from utils.base_frame_chunk_aligner import BaseFrameChunkAligner

# Test base class functionality
aligner = BaseFrameChunkAligner(PipelineType.CONVERSATIONS)
stats = aligner.get_common_processing_stats()
```

## 📝 **Development Guidelines**

### Adding New Features
1. **Follow shared data models** for consistency
2. **Extend base classes** rather than duplicating code
3. **Implement validation** using the shared validation functions
4. **Document pipeline-specific requirements** clearly

### Data Structure Changes
1. **Update shared models** in `utils/pipeline_data_models.py`
2. **Maintain backward compatibility** where possible
3. **Test validation** with existing data
4. **Update documentation** to reflect changes

### Pipeline Extensions
1. **Inherit from appropriate base classes**
2. **Implement required abstract methods**
3. **Add pipeline-specific fields** through inheritance
4. **Use factory functions** for data creation

## 🔗 **Related Documentation**

- **Shared Models**: `utils/README.md`
- **Base Classes**: `utils/base_frame_chunk_aligner.py`
- **Data Validation**: `utils/test_data_validation.py`
- **Main Architecture**: `ARCHITECTURE.md`
- **Quick Start Guide**: `QUICKSTART.md`

## 🎯 **Next Steps for Developers**

1. **Familiarize yourself** with the shared data models
2. **Understand the base class architecture** and inheritance patterns
3. **Refactor existing pipeline code** to use the standardized approach
4. **Implement data validation** at key pipeline steps
5. **Test consistency** across different pipeline types

## 🆘 **Troubleshooting**

### Common Issues
- **Data validation failures**: Check against shared model requirements
- **Inheritance errors**: Ensure all abstract methods are implemented
- **Type mismatches**: Use Pydantic validation to catch issues early

### Getting Help
- **Check shared model documentation** in `utils/README.md`
- **Review base class implementations** for examples
- **Run validation tests** to identify structural issues
- **Consult pipeline architecture** documentation

This pipeline is part of a larger effort to standardize data structures across all video processing pipelines, ensuring consistency, maintainability, and reduced troubleshooting in the unified RAG system.
