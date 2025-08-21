# Conversations Video Pipeline Documentation

This folder contains all documentation for the Conversations Video Pipeline, which is specialized for processing conversations between Michael Levin and other researchers.

## 🎯 **Pipeline Purpose**

The Conversations Video Pipeline is designed to process videos featuring conversations between Michael Levin and other researchers. Unlike the formal presentations pipeline (which handles single-speaker content), this pipeline:

- **Identifies multiple speakers** using AssemblyAI's speaker diarization
- **Focuses on Levin's views and knowledge** while capturing guest researcher perspectives
- **Extracts Q&A pairs** for fine-tuning purposes
- **Creates semantic chunks** that reflect conversation dynamics
- **Preserves context** between speakers for better understanding

## 📁 **Contents**

- **PIPELINE_ARCHITECTURE.md** - Comprehensive architecture overview and technical details
- **QUICK_REFERENCE.md** - Quick commands and troubleshooting guide
- **CONVERSATION_PROCESSING.md** - Details about speaker identification and Q&A extraction
- **STEP_04_CONVERSATION_CHUNKING.md** - Guide for semantic chunking of conversations

## 🗂️ **Folder Organization**

The pipeline has been organized for better maintainability:

```
SCIENTIFIC_VIDEO_PIPELINE/Conversations_and_working_meetings_1_on_1/
├── docs/                    # 📚 All documentation files
│   ├── README.md           # This file
│   ├── PIPELINE_ARCHITECTURE.md
│   ├── QUICK_REFERENCE.md
│   ├── CONVERSATION_PROCESSING.md
│   └── STEP_04_CONVERSATION_CHUNKING.md
├── logs/                    # 📝 All log files and reports
│   ├── *.log               # Step-specific log files
│   ├── conversations_pipeline_report_*.txt
│   └── cleanup_report_*.json
├── step_01_raw/            # Raw playlist data
├── step_02_extracted_playlist_content/  # Downloaded videos
├── step_03_transcription/  # Transcripts with speaker diarization
├── step_04_extract_chunks/ # Semantic chunks for conversations
├── step_05_frames/         # Extracted video frames
├── step_06_frame_chunk_alignment/  # Frame-chunk alignments
├── step_07_faiss_embeddings/  # FAISS indices and embeddings
├── step_08_cleanup.py      # Cleanup operations
├── run_conversations_pipeline.py  # Main pipeline runner
├── pipeline_progress_queue.py     # Progress tracking
└── logging_config.py       # Centralized logging
```

## 🔄 **Pipeline Steps**

### **Step 1: Playlist Processing**
- Process YouTube playlist URLs
- Extract metadata for all videos
- Initialize progress tracking

### **Step 2: Video Download**
- Download videos using yt-dlp
- Extract comprehensive metadata
- Prepare for transcription

### **Step 3: Transcription with Speaker Diarization**
- Use AssemblyAI for high-quality transcription
- Identify different speakers (Speaker A, Speaker B)
- Generate speaker-separated transcripts

### **Step 4: Semantic Chunking for Conversations**
- Create chunks based on speaker turns
- Focus on Levin's views and knowledge
- Extract Q&A patterns
- Label speakers (Michael Levin vs. Guest Researcher)

### **Step 5: Frame Extraction**
- Extract video frames at regular intervals
- Provide visual context for citations

### **Step 6: Frame-Chunk Alignment**
- Align extracted frames with transcript chunks
- Enable visual citations in search results

### **Step 7: Consolidated Embedding**
- Generate embeddings for conversation chunks
- Build FAISS indices for search
- Support both RAG and fine-tuning use cases

### **Step 8: Cleanup**
- Remove unnecessary files
- Preserve essential conversation data
- Manage disk space efficiently

## 🎭 **Speaker Identification**

The pipeline automatically identifies different speakers using AssemblyAI's speaker diarization:

- **Speaker A**: Usually Michael Levin (to be confirmed manually)
- **Speaker B**: Guest researcher or interviewer
- **Automatic labeling**: Based on conversation patterns and content analysis
- **Manual verification**: Required to confirm speaker identities

## 💬 **Q&A Extraction**

The pipeline extracts Q&A pairs for fine-tuning purposes:

- **Question detection**: Identifies questions asked by either speaker
- **Answer extraction**: Captures corresponding answers
- **Context preservation**: Maintains conversation flow
- **Fine-tuning preparation**: Formats data for training

## 🔍 **Semantic Chunking Strategy**

Unlike single-speaker content, conversation chunking:

- **Preserves speaker context**: Maintains who said what
- **Focuses on Levin's knowledge**: Prioritizes his views and explanations
- **Captures guest perspectives**: Stores guest researcher knowledge for reference
- **Maintains conversation flow**: Preserves natural dialogue structure

## 🚀 **Usage**

### **Run Complete Pipeline**
```bash
uv run python run_conversations_pipeline.py
```

### **Run Individual Steps**
```bash
# Step 1: Playlist Processing
uv run python step_01_playlist_processor.py

# Step 2: Video Download
uv run python step_02_video_downloader.py

# Step 3: Transcription with Speaker Diarization
uv run python step_03_transcription_webhook.py

# Step 4: Semantic Chunking for Conversations
uv run python step_04_extract_chunks.py

# Continue with other steps...
```

## 📊 **Expected Outputs**

- **Speaker-identified transcripts** with clear speaker labels
- **Semantic chunks** focused on Levin's views and knowledge
- **Q&A pairs** extracted for fine-tuning
- **FAISS indices** for conversation search and retrieval
- **Visual citations** through frame-chunk alignment

## 🔧 **Key Differences from Formal Presentations Pipeline**

1. **Speaker diarization** instead of single-speaker processing
2. **Conversation-focused chunking** instead of presentation chunking
3. **Q&A extraction** for fine-tuning purposes
4. **Speaker labeling** for Michael Levin vs. guest researchers
5. **Context preservation** between speakers

## 📚 **Documentation Structure**

- **Architecture**: Technical implementation details
- **Quick Reference**: Common commands and troubleshooting
- **Conversation Processing**: Speaker identification and Q&A extraction
- **Chunking Guide**: Semantic chunking strategies for conversations

## 🎯 **Next Steps**

1. **Review speaker identification** in step 4
2. **Verify Q&A extraction quality**
3. **Test conversation search functionality**
4. **Validate fine-tuning data preparation**

The Conversations Pipeline is designed to handle the complexity of multi-speaker content while maintaining focus on Michael Levin's knowledge and views.
