# Conversations and Working Meetings Pipeline (1-on-2)

This pipeline processes 1-on-2 conversations and working meetings featuring Michael Levin and 2 other researchers, extracting semantic content and creating RAG-ready outputs.

## 🏗️ **Pipeline Architecture**

This pipeline extends the standard 7-step video processing workflow to handle **three-speaker conversations**:

1. **Step 1**: Playlist processing and metadata extraction
2. **Step 2**: Video content download and extraction
3. **Step 3**: Transcription with speaker diarization (AssemblyAI)
4. **Step 4**: Semantic chunking and Q&A generation for 3-speaker dynamics
5. **Step 5**: Frame extraction (every 60 seconds)
6. **Step 6**: Frame-chunk alignment
7. **Step 7**: FAISS embedding generation

## 🔧 **Data Standardization**

**IMPORTANT**: This pipeline uses shared data models and base classes for consistency across all video pipelines.

### Shared Data Models
- **Location**: `utils/pipeline_data_models.py`
- **Purpose**: Ensures consistent data structures across all pipelines
- **Benefits**: Automatic validation, type safety, reduced troubleshooting

### Base Classes
- **Location**: `utils/base_frame_chunk_aligner.py`
- **Purpose**: Provides common functionality while allowing pipeline-specific customization
- **Usage**: Inherit from `BaseFrameChunkAligner` for consistent behavior

## 📁 **Directory Structure**

```
Conversations_and_working_meetings_1_on_2/
├── step_01_raw/                    # Playlist metadata and URLs
├── step_02_extracted_playlist_content/  # Downloaded video content
├── step_03_transcription/          # AssemblyAI transcripts with speakers
├── step_04_extract_chunks/         # Semantic chunking and Q&A generation
├── step_05_frames/                 # Extracted video frames
├── step_06_frame_chunk_alignment/  # Frame-chunk alignment
├── step_07_faiss_embeddings/       # Vector embeddings for RAG
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

## 🔍 **Key Features for 1-on-2 Conversations**

### Enhanced Speaker Identification System
- **Three-speaker support**: Handles Levin + 2 other researchers
- **Interactive prompts** for speaker identification (A, B, C)
- **Speaker role mapping**: Identifies Michael Levin vs. other speakers
- **Collaboration context**: Tracks interaction patterns between speakers

### Multi-Speaker Semantic Chunking
- **Conversation dynamics**: Captures 3-way discussions and debates
- **Speaker interaction patterns**: Identifies who asks, who answers, who elaborates
- **Collaborative insights**: Extracts shared understanding and disagreements
- **Q&A extraction**: Generates multi-speaker Q&A pairs

### Enhanced Context Preservation
- **Multi-speaker context**: Maintains conversation flow across speakers
- **Role-based analysis**: Tracks expertise areas and contributions
- **Collaboration networks**: Maps researcher interaction patterns
- **Cross-speaker references**: Links related concepts across different speakers

## 📊 **Data Outputs**

### Step 4: Enhanced Chunks
- **Location**: `step_04_extract_chunks/enhanced_chunks/`
- **Format**: JSON with Q&A pairs, speaker attribution, timestamps
- **Structure**: Follows shared data model standards with 3-speaker support

### Step 6: Frame-Chunk Alignment
- **Location**: `step_06_frame_chunk_alignment/`
- **Format**: JSON with aligned frames and semantic content
- **Structure**: Ready for standardization using shared models

### Step 7: FAISS Embeddings
- **Location**: `step_07_faiss_embeddings/`
- **Format**: FAISS index + metadata pickle files
- **Integration**: Unified with other pipelines in Streamlit frontend

## 🎭 **Speaker Identification for 1-on-2**

### Speaker Roles
1. **Michael Levin**: Primary researcher and expert
2. **Speaker 2**: Second researcher (academic, industry, etc.)
3. **Speaker 3**: Third researcher or moderator

### Identification Process
```
🎯 MICHAEL LEVIN IDENTIFICATION
Video: [VIDEO_ID]
Available speakers: A, B, C
==================================================
Which speaker is Michael Levin? (A, B, C): B
✅ Michael Levin identified as Speaker B
==================================================

🎭 SPEAKER 2 IDENTIFICATION
Video: [VIDEO_ID]
Speaker ID: A
==================================================
Enter the name for Speaker A: [NAME]
Enter the role/organization for Speaker A: [ROLE]
Any additional context about Speaker A? (optional): [CONTEXT]
✅ Speaker A identified as: [NAME] ([ROLE])

🎭 SPEAKER 3 IDENTIFICATION
Video: [VIDEO_ID]
Speaker ID: C
==================================================
Enter the name for Speaker C: [NAME]
Enter the role/organization for Speaker C: [ROLE]
Any additional context about Speaker C? (optional): [CONTEXT]
✅ Speaker C identified as: [NAME] ([ROLE])
```

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
is_valid = validate_pipeline_data(data, PipelineType.CONVERSATIONS_1_ON_2)
```

### Base Class Testing
```python
from utils.base_frame_chunk_aligner import BaseFrameChunkAligner

# Test base class functionality
aligner = BaseFrameChunkAligner(PipelineType.CONVERSATIONS_1_ON_2)
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
- **1-on-1 Pipeline**: `../Conversations_and_working_meetings_1_on_1/README.md`

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

## 🎉 **Conclusion**

This pipeline extends the 1-on-1 conversations pipeline to handle **three-speaker dynamics**, enabling:

- **Multi-speaker analysis** of research conversations
- **Collaboration pattern identification** between researchers
- **Enhanced context preservation** across multiple speakers
- **Rich semantic content** for AI training and research
- **Standardized data structures** for consistency across pipelines

The 1-on-2 pipeline ensures that every multi-speaker conversation is properly attributed, contextualized, and prepared for advanced research analysis and AI training.
