# YouTube Videos - Formal Solo Presentations Pipeline

This pipeline processes formal academic presentations, lectures, and solo talks from YouTube to create structured, searchable content with citation frames.

## Purpose
Transform formal solo presentations into structured content that can be used for:
- Academic lecture search and retrieval
- Citation-based content verification
- Educational content analysis
- RAG applications for formal presentations

## Data Flow
```
Video Download → Frame Extraction → Transcription → Content Chunking → Enhancement → Vector Embeddings
```

## Input Requirements
- **Content Type**: Formal academic presentations, lectures, solo talks
- **Format**: YouTube videos with single speaker
- **Content**: Structured presentations with clear sections and topics
- **Duration**: Typically 15-90 minutes

## Processing Steps
1. **Video Download**: Download video using yt-dlp
2. **Frame Extraction**: Extract frames every 5 seconds for citation purposes
3. **Transcription**: Generate accurate transcript using AssemblyAI
4. **Content Chunking**: Break content into logical sections and topics
5. **Enhancement**: Use GPT-4 to improve context and clarity
6. **Vector Embeddings**: Generate embeddings for similarity search

## Output Structure
- **Content Chunks**: Logical sections and topic-based chunks
- **Citation Frames**: Visual frames for content verification
- **Enhanced Transcripts**: Improved and contextualized text
- **Vector Embeddings**: Searchable vector representations

## Tools Required
- yt-dlp for video download
- FFmpeg for frame extraction
- AssemblyAI for transcription
- GPT-4 for content enhancement
- Vector embedding generation

## Data Directories
- `source_data/`: Downloaded videos and raw frames
- `transformed_data/`: Processed chunks, enhanced transcripts, and embeddings

## Configuration
- Frame extraction frequency (5 seconds)
- Transcription quality settings
- Chunking parameters
- Enhancement prompts
- Embedding model configuration

## Special Considerations
- **Speaker Consistency**: Single speaker throughout presentation
- **Content Structure**: Formal academic format with clear sections
- **Citation Frames**: Visual verification of content claims
- **Quality Control**: High accuracy requirements for academic content 