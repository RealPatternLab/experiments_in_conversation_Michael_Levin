# Media Pipeline Architecture

This directory contains separate, specialized pipelines for different types of media content. Each pipeline is completely independent with its own tools, data directories, and processing logic.

## Pipeline Structure

### Scientific Publications
- **Location**: `scientific_publications/`
- **Purpose**: Process research papers, PDFs, and academic publications
- **Data Flow**: Raw PDFs → Text Extraction → Semantic Chunking → Vector Embeddings
- **Output**: RAG-ready semantic chunks and Q&A pairs for fine-tuning

### YouTube Videos - Formal Solo Presentations
- **Location**: `youtube_videos_formal_solo_presentations/`
- **Purpose**: Process formal academic presentations, lectures, and solo talks
- **Data Flow**: Video Download → Frame Extraction → Transcription → Chunking → Enhancement
- **Output**: Structured content chunks with citation frames

### YouTube Videos - Podcasts (1-on-0)
- **Location**: `youtube_videos_podcasts_1_on_0`
- **Purpose**: Process solo podcast episodes or monologues
- **Data Flow**: Video Download → Frame Extraction → Transcription → Speaker Analysis → Chunking
- **Output**: Monologue chunks with enhanced context

### YouTube Videos - Podcasts (1-on-1)
- **Location**: `youtube_videos_podcasts_1_on_1`
- **Purpose**: Process two-person conversations and interviews
- **Data Flow**: Video Download → Frame Extraction → Transcription → Speaker Diarization → Conversation Chunking
- **Output**: Dialog chunks with speaker identification

### YouTube Videos - Podcasts (1-on-2)
- **Location**: `youtube_videos_podcasts_1_on_2`
- **Purpose**: Process three-person conversations and panel discussions
- **Data Flow**: Video Download → Frame Extraction → Transcription → Multi-Speaker Diarization → Conversation Chunking
- **Output**: Multi-speaker dialog chunks

### YouTube Videos - Podcasts (1-on-3)
- **Location**: `youtube_videos_podcasts_1_on_3`
- **Purpose**: Process four-person conversations and group discussions
- **Data Flow**: Video Download → Frame Extraction → Transcription → Multi-Speaker Diarization → Conversation Chunking
- **Output**: Multi-speaker dialog chunks

### YouTube Videos - Podcasts (1-on-4)
- **Location**: `youtube_videos_podcasts_1_on_4`
- **Purpose**: Process five-person conversations and large panel discussions
- **Data Flow**: Video Download → Frame Extraction → Transcription → Multi-Speaker Diarization → Conversation Chunking
- **Output**: Multi-speaker dialog chunks

### Raw Ingestion
- **Location**: `raw_ingestion/`
- **Purpose**: Common ingestion layer for all media types
- **Data Flow**: File Upload → Type Detection → Routing to Appropriate Pipeline
- **Output**: Organized files ready for pipeline processing

## Standard Pipeline Structure

Each media pipeline follows this standard structure:

```
pipeline_name/
├── pipeline/                    # Pipeline orchestration scripts
│   ├── main_pipeline.py        # Main pipeline script
│   └── logs/                   # Pipeline execution logs
├── tools/                       # Media-specific processing tools
└── data/                        # Data directories
    ├── source_data/             # Input and intermediate data
    │   ├── raw/                 # Raw input files
    │   ├── archive/             # Archived raw files
    │   ├── DLQ/                 # Dead letter queue
    │   └── preprocessed/        # Processed files
    └── transformed_data/        # Final processed data
        ├── [media_specific_outputs]
        └── [vector_embeddings]  # If applicable
```

## Benefits of This Architecture

1. **Separation of Concerns**: Each media type has its own specialized processing
2. **Independent Development**: Teams can work on different pipelines without conflicts
3. **Specialized Tools**: Each pipeline can use media-specific processing tools
4. **Clear Data Flow**: Source data and transformed data are clearly separated
5. **Easy Testing**: Each pipeline can be tested independently
6. **Scalability**: New media types can be added as new pipelines

## Adding New Media Types

To add a new media type:
1. Create a new directory under `media_pipelines/`
2. Follow the standard structure (pipeline/, tools/, data/)
3. Implement pipeline-specific processing logic
4. Add routing logic to `raw_ingestion/`
5. Document the new pipeline in this README 