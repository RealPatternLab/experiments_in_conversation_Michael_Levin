# Scientific Video Pipeline Architecture & Operation Guide

## Overview

The Scientific Video Pipeline is a sophisticated 7-step video processing system designed for academic content analysis. It processes YouTube videos through transcription, semantic chunking, frame extraction, and embedding generation, with intelligent skip logic and centralized progress tracking.

## 🏗️ Pipeline Architecture

### **Core Components**

1. **Pipeline Runner** (`run_video_pipeline_1_on_0.py`)
   - Orchestrates all 7 steps sequentially
   - Uses `uv run` for Python execution
   - Provides comprehensive logging and timing

2. **Progress Queue System** (`pipeline_progress_queue.py`)
   - Centralized state management for all pipeline steps
   - Thread-safe operations with file-based persistence
   - Tracks completion status for each video at each step

3. **Webhook Integration** (`assemblyai_webhooks.json`)
   - Manages AssemblyAI transcription requests
   - Tracks pending and completed transcriptions
   - Prevents infinite loops in monitoring

### **Pipeline Steps**

| Step | Script | Purpose | Output |
|------|--------|---------|---------|
| 1 | `step_01_playlist_processor.py` | Process YouTube playlists | Playlist metadata |
| 2 | `step_02_video_downloader.py` | Download videos with yt-dlp | Video files + metadata |
| 3 | `step_03_transcription_webhook.py` | Generate transcripts via AssemblyAI | JSON + text transcripts |
| 4 | `step_04_extract_chunks.py` | Create semantic chunks with LLM | Structured chunk data |
| 5 | `step_05_frame_extractor.py` | Extract video frames | JPG frames + metadata |
| 6 | `step_06_frame_chunk_alignment.py` | Align frames with chunks | Frame-chunk mappings |
| 7 | `step_07_consolidated_embedding.py` | Generate embeddings + FAISS indices | Vector indices |
| 8 | `step_08_cleanup.py` | Remove unnecessary files to free disk space | Cleanup report + space savings |

## 🔄 Smart Processing System

### **Intelligent Skip Logic**

The pipeline implements a sophisticated skip system that prevents unnecessary reprocessing:

- **Progress Queue Check**: Each step first checks `pipeline_progress_queue.json` for completion status
- **File Existence Check**: Fallback to checking actual output files
- **Content Hash Validation**: For steps 5-7, validates content hasn't changed
- **Atomic Updates**: Progress queue updates are atomic and consistent

### **Skip Behavior by Step**

| Step | Skip Condition | Skip Speed |
|------|----------------|------------|
| 1 | No new videos in playlist | ~2 seconds |
| 2 | Video files already exist | ~0.1 seconds |
| 3 | Transcript files exist + webhook synced | ~0.1 seconds |
| 4 | Chunk files exist | ~0.1 seconds |
| 5 | Frame summary + files exist | ~0.1 seconds |
| 6 | Alignment files exist | ~0.1 seconds |
| 7 | Embedding files exist for timestamp | ~0.1 seconds |
| 8 | Always runs (cleanup operation) | ~1-5 seconds |

## 📊 Progress Queue System

### **Data Structure**

```json
{
  "pipeline_progress": {
    "VIDEO_ID": {
      "step_01_playlist_processing": "completed|pending|failed",
      "step_02_video_download": "completed|pending|failed",
      "step_03_transcription": "completed|pending|failed",
      "step_04_semantic_chunking": "completed|pending|failed",
      "step_05_frame_extraction": "completed|pending|failed",
      "step_06_frame_chunk_alignment": "completed|pending|failed",
      "step_07_consolidated_embedding": "completed|pending|failed",
      "step_08_cleanup": "completed|pending|failed",
      "processing_metadata": { ... }
    }
  },
  "playlist_progress": { ... },
  "pipeline_info": { ... }
}
```

### **Key Methods**

- `add_video(video_id, video_title, playlist_id)`: Register new video
- `update_video_step_status(video_id, step, status, metadata)`: Update step completion
- `get_video_status(video_id)`: Query current video status
- `get_pending_videos_for_step(step)`: Get videos needing processing

## 🧹 Cleanup System

### **Step 8: Intelligent File Cleanup**

The cleanup step removes unnecessary files while preserving essential data for Streamlit and pipeline tracking:

#### **Files Removed:**
- **Large Video Files** (`.mp4`) - Not needed for analysis or Streamlit
- **Unreferenced Frames** - Frame files not referenced in chunk alignments
- **yt-dlp Artifacts** - `.info.json`, `.webp`, `.vtt`, `.description` files
- **Temporary Scripts** - One-time utility scripts for debugging
- **Old Logs** - Truncated to last 1000 lines
- **Old Reports** - Keeps only the most recent pipeline report
- **Python Cache** - `__pycache__` directories
- **Audio Files** - Temporary `.wav` files from transcription
- **Submission Metadata** - AssemblyAI submission files

#### **Files Preserved:**
- **Progress Queue** - `pipeline_progress_queue.json`
- **Webhook State** - `assemblyai_webhooks.json`
- **All Step Scripts** - Core pipeline functionality
- **Transcripts** - `*_transcript.json` files
- **Semantic Chunks** - `*_chunks.json` files
- **Referenced Frames** - Frames used in chunk alignments
- **Alignments** - `*_alignments.json` files
- **Embeddings** - FAISS indices and vector data
- **Documentation** - All `.md` files
- **Latest Report** - Most recent pipeline execution report

#### **Space Savings:**
- **Typical Savings**: 0.1-1.0 GB per cleanup run
- **File Count Reduction**: 300-1000+ files removed
- **Smart Detection**: Only removes files that are truly unnecessary

#### **Usage:**
```bash
# Dry run to see what would be removed
uv run python step_08_cleanup.py --dry-run

# Actual cleanup
uv run python step_08_cleanup.py

# With verbose logging
uv run python step_08_cleanup.py --verbose
```

## 🔗 Webhook Integration

### **AssemblyAI Webhook Management**

The transcription system uses webhooks to handle asynchronous processing:

```json
{
  "pending_transcriptions": {
    "TRANSCRIPT_ID": {
      "video_id": "VIDEO_ID",
      "video_title": "Video Title",
      "submitted_at": "TIMESTAMP",
      "status": "pending"
    }
  },
  "completed_transcriptions": {
    "TRANSCRIPT_ID": {
      "video_id": "VIDEO_ID",
      "video_title": "Video Title",
      "completed_at": "TIMESTAMP",
      "status": "completed"
    }
  }
}
```

### **Anti-Hanging Mechanisms**

1. **Duplicate Entry Cleanup**: Automatically removes videos appearing in both pending and completed
2. **Timeout Protection**: Maximum monitoring time (default: 1 hour)
3. **Immediate Webhook Updates**: Updates webhook file as soon as transcripts are downloaded
4. **Safety Checks**: Multiple exit conditions in monitoring loops

## 🚀 Execution Workflow

### **Standard Pipeline Run**

```bash
# Run complete pipeline
uv run python run_video_pipeline_1_on_0.py

# Run individual steps
uv run python step_01_playlist_processor.py
uv run python step_02_video_downloader.py
uv run python step_03_transcription_webhook.py --monitor
uv run python step_04_extract_chunks.py
uv run python step_05_frame_extractor.py
uv run python step_06_frame_chunk_alignment.py
uv run python step_07_consolidated_embedding.py
uv run python step_08_cleanup.py
```

### **Execution Patterns**

1. **First Run**: Processes all videos through all steps
2. **Incremental Run**: Skips existing content, processes only new videos
3. **Re-run**: Skips everything if no changes detected
4. **New Video Addition**: Automatically detects and processes new content

## 📁 File Organization

### **Directory Structure**

```
SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/
├── step_01_raw/                          # Playlist metadata
├── step_02_extracted_playlist_content/   # Downloaded videos
├── step_03_transcription/                 # AssemblyAI transcripts
├── step_04_extract_chunks/               # Semantic chunks
├── step_05_frames/                       # Extracted video frames
├── step_06_frame_chunk_alignment/        # Frame-chunk mappings
├── step_07_faiss_embeddings/             # FAISS indices
├── pipeline_progress_queue.json           # Centralized state
├── assemblyai_webhooks.json              # Webhook management
└── run_video_pipeline_1_on_0.py         # Main pipeline runner
```

### **File Naming Conventions**

- **Videos**: `video_{VIDEO_ID}/video.mp4`
- **Transcripts**: `{VIDEO_ID}_transcript.json` + `.txt`
- **Chunks**: `{VIDEO_ID}_chunks.json`
- **Frames**: `{VIDEO_ID}_frames_summary.json` + `{VIDEO_ID}/frame_*.jpg`
- **Embeddings**: `{TIMESTAMP}_*` prefixed files

## 🔧 Troubleshooting & Maintenance

### **Common Issues & Solutions**

#### **Step 3 Hanging**
- **Symptom**: Pipeline stuck in transcription monitoring
- **Cause**: Webhook file not updated when transcripts exist
- **Solution**: Fixed in current version - automatic webhook syncing

#### **Skip Logic Not Working**
- **Symptom**: Steps reprocessing existing content
- **Cause**: Progress queue not updated or corrupted
- **Solution**: Check `pipeline_progress_queue.json` integrity

#### **File Path Issues**
- **Symptom**: Steps can't find video files
- **Cause**: Incorrect relative paths in metadata
- **Solution**: Fixed in current version - full path storage

### **Debugging Commands**

```bash
# Check progress queue status
cat pipeline_progress_queue.json | jq '.'

# Check webhook status
cat assemblyai_webhooks.json | jq '.'

# View step logs
tail -f transcription_webhook.log
tail -f chunking.log
tail -f frame_extraction.log

# Force reprocess specific step
rm -rf step_04_extract_chunks/*_chunks.json
uv run python step_04_extract_chunks.py
```

### **Maintenance Tasks**

1. **Log Rotation**: Archive old log files periodically
2. **Progress Queue Backup**: Backup `pipeline_progress_queue.json` before major changes
3. **Webhook Cleanup**: Monitor `assemblyai_webhooks.json` for orphaned entries
4. **Storage Management**: Monitor disk usage in frame and embedding directories

## 📈 Performance Characteristics

### **Execution Times**

| Scenario | Duration | Notes |
|----------|----------|-------|
| **First Run (2 videos)** | ~5-10 minutes | Full processing |
| **Incremental (1 new video)** | ~5-10 minutes | Skip existing + process new |
| **Re-run (no changes)** | ~3-5 seconds | Skip everything |
| **Large Video Processing** | ~10-20 minutes | Depends on video length |

### **Resource Usage**

- **CPU**: Intensive during LLM processing (step 4) and frame extraction (step 5)
- **Memory**: Peak usage during embedding generation (step 7)
- **Storage**: Frame extraction can generate significant disk usage
- **Network**: AssemblyAI API calls and video downloads

## 🔮 Future Enhancements

### **Planned Features**

1. **Parallel Processing**: Independent steps could run concurrently
2. **Cloud Integration**: Support for cloud storage and compute
3. **Web Dashboard**: Real-time monitoring and control interface
4. **Configuration Management**: YAML-based pipeline configuration
5. **Plugin System**: Modular step implementation

### **Scalability Improvements**

1. **Batch Processing**: Handle multiple videos simultaneously
2. **Distributed Processing**: Multi-machine pipeline execution
3. **Caching Layer**: Redis-based progress tracking
4. **API Rate Limiting**: Intelligent throttling for external services

## 📚 Best Practices

### **Development Workflow**

1. **Test with Small Videos**: Use short videos for development
2. **Monitor Logs**: Check logs after each step execution
3. **Validate Outputs**: Verify file generation and metadata
4. **Incremental Testing**: Test one step at a time

### **Production Deployment**

1. **Environment Setup**: Ensure all dependencies are installed
2. **API Key Management**: Secure storage of AssemblyAI keys
3. **Monitoring**: Set up log monitoring and alerting
4. **Backup Strategy**: Regular backups of progress queue and metadata

### **Content Management**

1. **Playlist Organization**: Use descriptive playlist names
2. **Video Metadata**: Ensure videos have meaningful titles
3. **Regular Updates**: Run pipeline periodically to catch new content
4. **Quality Control**: Review generated chunks and alignments

## 🎯 Conclusion

The Scientific Video Pipeline represents a mature, production-ready system for academic video content processing. With its intelligent skip logic, centralized progress tracking, and robust error handling, it provides a reliable foundation for large-scale video analysis workflows.

The system's architecture balances efficiency with reliability, ensuring that:
- **Existing work is preserved** through intelligent skip logic
- **New content is processed** automatically and efficiently
- **System state is maintained** consistently across all operations
- **Errors are handled gracefully** with comprehensive logging and recovery

This makes it ideal for both research environments where content is continuously added and production systems requiring reliable, repeatable processing workflows.
