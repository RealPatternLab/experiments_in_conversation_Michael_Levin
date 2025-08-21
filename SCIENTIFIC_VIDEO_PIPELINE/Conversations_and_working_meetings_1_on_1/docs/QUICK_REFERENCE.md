# Conversations Video Pipeline - Quick Reference Guide

## 🚀 **Current Status: DEVELOPMENT IN PROGRESS**

The conversations pipeline is being developed based on the proven formal presentations pipeline:
- ✅ **Pipeline Structure**: 8-step architecture established
- ✅ **Progress Queue**: Specialized for conversations and speaker identification
- ✅ **Logging System**: Centralized logging with automatic cleanup
- ✅ **Documentation**: Comprehensive architecture and usage guides
- 🔄 **Implementation**: Core components being developed

## 📋 **Pipeline Steps Overview**

| Step | Script | Purpose | Status |
|------|--------|---------|---------|
| 1 | `step_01_playlist_processor.py` | Process YouTube playlists | ✅ Ready |
| 2 | `step_02_video_downloader.py` | Download videos with yt-dlp | 🔄 In Development |
| 3 | `step_03_transcription_webhook.py` | Generate transcripts with speaker diarization | 🔄 In Development |
| 4 | `step_04_extract_chunks.py` | Create semantic chunks for conversations | 🔄 In Development |
| 5 | `step_05_frame_extractor.py` | Extract video frames | 🔄 In Development |
| 6 | `step_06_frame_chunk_alignment.py` | Align frames with chunks | 🔄 In Development |
| 7 | `step_07_consolidated_embedding.py` | Generate embeddings + FAISS indices | 🔄 In Development |
| 8 | `step_08_cleanup.py` | Remove unnecessary files to free disk space | 🔄 In Development |

## 🎯 **Key Commands**

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
uv run python step_03_transcription_webhook.py --monitor

# Step 4: Semantic Chunking for Conversations
uv run python step_04_extract_chunks.py

# Step 5: Frame Extraction
uv run python step_05_frame_extractor.py

# Step 6: Frame-Chunk Alignment
uv run python step_06_frame_chunk_alignment.py

# Step 7: Consolidated Embedding
uv run python step_07_consolidated_embedding.py

# Step 8: Cleanup (remove unnecessary files)
uv run python step_08_cleanup.py --dry-run  # Preview what will be removed
uv run python step_08_cleanup.py            # Actual cleanup
```

## 📊 **Performance Expectations**

| Scenario | Expected Duration | Notes |
|----------|------------------|-------|
| **First Run (2 videos)** | 10-20 minutes | Full processing with speaker diarization |
| **Incremental (1 new video)** | 10-20 minutes | Skip existing + process new |
| **Re-run (no changes)** | 3-5 seconds | Skip everything |
| **Speaker Re-identification** | 2-5 minutes | Re-process speaker mapping |
| **Q&A Re-extraction** | 1-3 minutes | Re-process Q&A patterns |
| **Cleanup (Step 8)** | 1-5 seconds | Removes unnecessary files |

## 🔍 **Monitoring & Debugging**

### **Check Pipeline Status**
```bash
# Progress queue status
cat logs/pipeline_progress_queue.json | jq '.'

# Webhook status
cat assemblyai_webhooks.json | jq '.'

# Recent logs
tail -f logs/pipeline_execution_*.log
```

### **View Step-Specific Logs**
```bash
# Playlist processing logs
tail -f logs/playlist_processing_*.log

# Transcription logs
tail -f logs/transcription_webhook_*.log

# Chunking logs
tail -f logs/chunking_*.log

# Speaker identification logs
tail -f logs/speaker_identification_*.log

# Q&A extraction logs
tail -f logs/qa_extraction_*.log
```

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Speaker Identification Issues**
- **Symptom**: Speakers not properly identified
- **Cause**: AssemblyAI speaker diarization failed or unclear
- **Solution**: Review transcript quality, re-run speaker identification

#### **Q&A Extraction Problems**
- **Symptom**: Missing or incorrect Q&A pairs
- **Cause**: Question/answer detection logic failed
- **Solution**: Review conversation patterns, adjust detection thresholds

#### **Skip Logic Not Working**
- **Symptom**: Steps reprocessing existing content
- **Cause**: Progress queue not updated or corrupted
- **Solution**: Check `logs/pipeline_progress_queue.json` integrity

#### **File Path Issues**
- **Symptom**: Steps can't find video files
- **Cause**: Incorrect relative paths in metadata
- **Solution**: Check `step_02_extracted_playlist_content/download_summary.json`

### **Force Reprocessing**
```bash
# Remove specific step outputs to force reprocessing
rm -rf step_04_extract_chunks/*_chunks.json
uv run python step_04_extract_chunks.py

# Remove all outputs for a video
rm -rf step_0*/*VIDEO_ID*

# Reset progress queue for a video
python -c "
from pipeline_progress_queue import get_progress_queue
pq = get_progress_queue()
pq.reset_video('VIDEO_ID')
"
```

## 📁 **Key Files & Directories**

### **State Management**
- `logs/pipeline_progress_queue.json` - Centralized pipeline state
- `assemblyai_webhooks.json` - AssemblyAI transcription tracking

### **Output Directories**
- `step_01_raw/` - Playlist metadata
- `step_02_extracted_playlist_content/` - Downloaded videos
- `step_03_transcription/` - AssemblyAI transcripts with speaker labels
- `step_04_extract_chunks/` - Semantic chunks for conversations
- `step_05_frames/` - Extracted video frames
- `step_06_frame_chunk_alignment/` - Frame-chunk mappings
- `step_07_faiss_embeddings/` - FAISS indices

### **Special Files for Conversations**
- `*_speakers.json` - Speaker identification mappings
- `*_qa_pairs.json` - Extracted Q&A pairs for fine-tuning
- `*_conversation_chunks.json` - Conversation-focused semantic chunks

## 🔧 **Maintenance**

### **Regular Tasks**
1. **Monitor speaker identification quality** - Review speaker mappings
2. **Validate Q&A extraction** - Check Q&A pair quality
3. **Monitor disk usage** in frame and embedding directories
4. **Backup progress queue** before major changes

### **Backup Commands**
```bash
# Backup progress queue
cp logs/pipeline_progress_queue.json logs/pipeline_progress_queue.json.backup

# Backup webhook file
cp assemblyai_webhooks.json assemblyai_webhooks.json.backup

# Backup speaker mappings
cp step_04_extract_chunks/*_speakers.json step_04_extract_chunks/backup/
```

## 🎭 **Speaker Identification Workflow**

### **Automatic Detection**
1. **AssemblyAI diarization** identifies Speaker A and Speaker B
2. **Content analysis** scores speakers based on Levin indicators
3. **Automatic mapping** assigns speaker identities

### **Manual Verification**
1. **Review speaker mappings** in `*_speakers.json` files
2. **Confirm identities** through transcript review
3. **Update mappings** if needed
4. **Re-run processing** if corrections made

### **Levin Indicators**
The system uses these keywords to identify Michael Levin:
- bioelectricity, regeneration, xenobots, morphogenesis
- developmental biology, collective intelligence
- goal-directed behavior, unconventional intelligence

## 💬 **Q&A Extraction Workflow**

### **Detection Process**
1. **Question identification** using linguistic patterns
2. **Answer detection** following questions
3. **Context preservation** maintaining conversation flow
4. **Classification** by question type and topic

### **Q&A Types**
- **Knowledge Questions**: About scientific concepts
- **Clarification Questions**: Requests for explanation
- **Hypothesis Questions**: About potential mechanisms
- **Application Questions**: About practical implications

### **Output Format**
```json
{
  "question": {
    "text": "What is bioelectricity?",
    "questioner": "speaker_b",
    "start_time": 120.5,
    "end_time": 125.2
  },
  "answer": {
    "text": "Bioelectricity refers to the electrical signals...",
    "respondent": "speaker_a",
    "start_time": 125.3,
    "end_time": 180.1
  },
  "qa_type": "knowledge_question",
  "topic": "bioelectricity"
}
```

## 🔍 **Semantic Chunking Strategy**

### **Conversation Chunking**
- **Speaker-based chunks**: New chunk when speaker changes
- **Topic-based chunks**: New chunk when topic shifts
- **Context preservation**: Maintains dialogue flow
- **Levin focus**: Prioritizes Levin's explanations

### **Chunk Types**
1. **Levin Explanation**: Michael Levin explaining concepts
2. **Guest Presentation**: Guest researcher presenting work
3. **Q&A Exchange**: Question and answer interactions
4. **Discussion**: General conversation and discussion
5. **Clarification**: Requests for clarification

## 🚀 **Next Steps**

### **Immediate Development**
1. **Complete step implementations** for steps 2-8
2. **Test speaker identification** with sample conversations
3. **Validate Q&A extraction** quality
4. **Test conversation search** functionality

### **Testing Strategy**
1. **Start with short conversations** (5-10 minutes)
2. **Verify speaker diarization** accuracy
3. **Test Q&A detection** with known patterns
4. **Validate chunk quality** and context preservation

## 📞 **Support**

For issues or questions:
1. Check the logs for detailed error information
2. Verify the progress queue and webhook file integrity
3. Review the comprehensive documentation in `PIPELINE_ARCHITECTURE.md`
4. Check the troubleshooting section in `CONVERSATION_PROCESSING.md`

---

**Last Updated**: 2025-01-XX  
**Pipeline Version**: 1.0 (Development)  
**Status**: Core structure established, implementation in progress

The Conversations Pipeline builds on the proven formal presentations pipeline while adding specialized capabilities for multi-speaker content analysis.
