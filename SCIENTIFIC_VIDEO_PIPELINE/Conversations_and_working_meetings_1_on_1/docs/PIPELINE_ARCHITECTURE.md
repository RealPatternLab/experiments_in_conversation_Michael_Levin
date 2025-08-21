# Conversations Video Pipeline Architecture & Operation Guide

## Overview

The Conversations Video Pipeline is a sophisticated 8-step video processing system designed for analyzing conversations between Michael Levin and other researchers. It processes YouTube videos through transcription with speaker diarization, semantic chunking focused on conversation dynamics, Q&A extraction, and embedding generation, with intelligent skip logic and centralized progress tracking.

## 🏗️ **Pipeline Architecture**

### **Core Components**

1. **Pipeline Runner** (`run_conversations_pipeline.py`)
   - Orchestrates all 8 steps sequentially
   - Uses `uv run` for Python execution
   - Provides comprehensive logging and timing
   - Specialized for conversation processing

2. **Progress Queue System** (`pipeline_progress_queue.py`)
   - Centralized state management for all pipeline steps
   - Thread-safe operations with file-based persistence
   - Tracks completion status for each video at each step
   - Specialized tracking for speaker identification and Q&A extraction

3. **Webhook Integration** (`assemblyai_webhooks.json`)
   - Manages AssemblyAI transcription requests
   - Tracks pending and completed transcriptions
   - Prevents infinite loops in monitoring
   - Supports speaker diarization

### **Pipeline Steps**

| Step | Script | Purpose | Output | Specialization |
|------|--------|---------|---------|----------------|
| 1 | `step_01_playlist_processor.py` | Process YouTube playlists | Playlist metadata | Conversation-focused metadata |
| 2 | `step_02_video_downloader.py` | Download videos with yt-dlp | Video files + metadata | Preserve conversation context |
| 3 | `step_03_transcription_webhook.py` | Generate transcripts with speaker diarization | JSON + text transcripts with speakers | **Speaker A/B identification** |
| 4 | `step_04_extract_chunks.py` | Create semantic chunks for conversations | Structured chunk data | **Levin-focused + Q&A extraction** |
| 5 | `step_05_frame_extractor.py` | Extract video frames | JPG frames + metadata | Visual context for conversations |
| 6 | `step_06_frame_chunk_alignment.py` | Align frames with chunks | Frame-chunk mappings | Visual citations for conversations |
| 7 | `step_07_consolidated_embedding.py` | Generate embeddings + FAISS indices | Vector indices | Conversation search optimization |
| 8 | `step_08_cleanup.py` | Remove unnecessary files to free disk space | Cleanup report + space savings | Preserve conversation data |

## 🔄 **Smart Processing System**

### **Intelligent Skip Logic**

The pipeline implements a sophisticated skip system that prevents unnecessary reprocessing:

- **Progress Queue Check**: Each step first checks `pipeline_progress_queue.json` for completion status
- **File Existence Check**: Fallback to checking actual output files
- **Content Hash Validation**: For steps 5-7, validates content hasn't changed
- **Atomic Updates**: Progress queue updates are atomic and consistent
- **Speaker Identification Tracking**: Special tracking for speaker labeling progress
- **Q&A Extraction Tracking**: Special tracking for Q&A pair extraction

### **Skip Behavior by Step**

| Step | Skip Condition | Skip Speed | Special Features |
|------|----------------|------------|------------------|
| 1 | No new videos in playlist | ~2 seconds | Conversation metadata detection |
| 2 | Video files already exist | ~0.1 seconds | Conversation context preservation |
| 3 | Transcript files exist + webhook synced | ~0.1 seconds | **Speaker diarization validation** |
| 4 | Chunk files exist | ~0.1 seconds | **Speaker identification + Q&A extraction** |
| 5 | Frame summary + files exist | ~0.1 seconds | Conversation visual context |
| 6 | Alignment files exist | ~0.1 seconds | Conversation frame alignment |
| 7 | Embedding files exist for timestamp | ~0.1 seconds | Conversation search optimization |
| 8 | Always runs (cleanup operation) | ~1-5 seconds | Preserve conversation data |

## 📊 **Progress Queue System**

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
      "speaker_identification": "completed|pending|failed",
      "qa_extraction": "completed|pending|failed",
      "speaker_mapping": {
        "speaker_a": "Michael Levin",
        "speaker_b": "Guest Researcher"
      },
      "qa_metadata": {
        "total_qa_pairs": 15,
        "extraction_timestamp": "2025-01-XX..."
      },
      "processing_metadata": { ... }
    }
  },
  "playlist_progress": { ... },
  "pipeline_info": {
    "pipeline_type": "conversations_1_on_1",
    "description": "Pipeline for conversations between Michael Levin and other researchers"
  }
}
```

### **Key Methods**

- `add_video(video_id, video_title, playlist_id)`: Register new conversation video
- `update_video_step_status(video_id, step, status, metadata)`: Update step completion
- `update_speaker_identification(video_id, speakers, metadata)`: Update speaker mapping
- `update_qa_extraction(video_id, qa_pairs, metadata)`: Update Q&A extraction status
- `get_video_status(video_id)`: Query current video status
- `get_pending_videos_for_step(step)`: Get videos needing processing

## 🎭 **Speaker Identification System**

### **AssemblyAI Speaker Diarization**

The transcription system uses AssemblyAI's speaker diarization capabilities:

```python
# Configuration for speaker diarization
config = aai.TranscriptionConfig(
    speaker_labels=True,      # Enable speaker diarization
    speakers_expected=2,      # Expect 2 speakers (Levin + 1 other)
    summarization=True,       # Generate summary
    sentiment_analysis=True,  # Analyze sentiment
    auto_highlights=True,     # Extract key highlights
    entity_detection=True,    # Detect named entities
    iab_categories=True,      # Content categorization
    punctuate=True,           # Proper punctuation
    format_text=True          # Clean text formatting
)
```

### **Speaker Mapping Strategy**

1. **Automatic Detection**: AssemblyAI identifies Speaker A and Speaker B
2. **Content Analysis**: Analyze speech patterns to identify Michael Levin
3. **Manual Verification**: Confirm speaker identities through review
4. **Mapping Storage**: Store confirmed speaker mappings in progress queue

### **Speaker Identification Process**

```python
def identify_speakers(transcript_data):
    """Identify which speaker is Michael Levin"""
    
    # Analyze speech patterns
    speaker_patterns = {
        'speaker_a': analyze_speaker_patterns(transcript_data, 'A'),
        'speaker_b': analyze_speaker_patterns(transcript_data, 'B')
    }
    
    # Use heuristics to identify Levin
    levin_indicators = [
        'bioelectricity', 'regeneration', 'xenobots', 'morphogenesis',
        'developmental biology', 'collective intelligence', 'goal-directed behavior'
    ]
    
    # Score each speaker based on Levin indicators
    scores = {}
    for speaker, patterns in speaker_patterns.items():
        score = sum(1 for indicator in levin_indicators 
                   if indicator in patterns['text'].lower())
        scores[speaker] = score
    
    # Map speakers
    if scores['speaker_a'] > scores['speaker_b']:
        return {'speaker_a': 'Michael Levin', 'speaker_b': 'Guest Researcher'}
    else:
        return {'speaker_a': 'Guest Researcher', 'speaker_b': 'Michael Levin'}
```

## 💬 **Q&A Extraction System**

### **Q&A Detection Strategy**

The pipeline implements sophisticated Q&A detection:

```python
def detect_qa_patterns(transcript_data):
    """Detect Q&A patterns in conversations"""
    
    qa_pairs = []
    current_question = None
    
    for utterance in transcript_data['utterances']:
        speaker = utterance['speaker']
        text = utterance['text']
        
        # Detect questions
        if is_question(text):
            current_question = {
                'question': text,
                'questioner': speaker,
                'start_time': utterance['start'],
                'end_time': utterance['end']
            }
        
        # Detect answers (following questions)
        elif current_question and is_answer(text):
            qa_pair = {
                'question': current_question,
                'answer': {
                    'text': text,
                    'respondent': speaker,
                    'start_time': utterance['start'],
                    'end_time': utterance['end']
                },
                'qa_type': classify_qa_type(current_question['question'], text)
            }
            qa_pairs.append(qa_pair)
            current_question = None
    
    return qa_pairs
```

### **Q&A Classification**

The system classifies Q&A pairs for different use cases:

- **Knowledge Questions**: Questions about scientific concepts
- **Clarification Questions**: Requests for explanation or detail
- **Hypothesis Questions**: Questions about potential mechanisms
- **Application Questions**: Questions about practical implications

## 🔍 **Semantic Chunking Strategy**

### **Conversation Chunking Approach**

Unlike single-speaker content, conversation chunking preserves dialogue structure:

```python
def create_conversation_chunks(transcript_data):
    """Create semantic chunks for conversations"""
    
    chunks = []
    current_chunk = None
    chunk_id = 0
    
    for utterance in transcript_data['utterances']:
        speaker = utterance['speaker']
        text = utterance['text']
        
        # Start new chunk if speaker changes or topic shifts
        if (current_chunk is None or 
            current_chunk['speaker'] != speaker or 
            topic_shift_detected(current_chunk, text)):
            
            # Save previous chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # Start new chunk
            chunk_id += 1
            current_chunk = {
                'chunk_id': f"{transcript_data['video_id']}_chunk_{chunk_id:03d}",
                'video_id': transcript_data['video_id'],
                'speaker': speaker,
                'speaker_identity': get_speaker_identity(speaker),
                'text': text,
                'start_time': utterance['start'],
                'end_time': utterance['end'],
                'chunk_type': classify_chunk_type(text),
                'topic': extract_topic(text),
                'levin_knowledge_focus': is_levin_knowledge(text, speaker),
                'qa_context': extract_qa_context(text)
            }
        else:
            # Extend current chunk
            current_chunk['text'] += ' ' + text
            current_chunk['end_time'] = utterance['end']
    
    return chunks
```

### **Chunk Types for Conversations**

1. **Levin Explanation**: Michael Levin explaining concepts
2. **Guest Presentation**: Guest researcher presenting their work
3. **Q&A Exchange**: Question and answer interactions
4. **Discussion**: General conversation and discussion
5. **Clarification**: Requests for clarification or detail

## 🧹 **Cleanup System**

### **Step 8: Intelligent File Cleanup**

The cleanup step removes unnecessary files while preserving essential conversation data:

#### **Files Removed:**
- **Large Video Files** (`.mp4`) - Not needed for analysis
- **Unreferenced Frames** - Frame files not referenced in chunk alignments
- **yt-dlp Artifacts** - Temporary download files
- **Old Logs** - Truncated to last 1000 lines
- **Old Reports** - Keeps only the most recent pipeline report

#### **Files Preserved:**
- **Progress Queue** - `pipeline_progress_queue.json`
- **Webhook State** - `assemblyai_webhooks.json`
- **Speaker Mappings** - Speaker identification data
- **Q&A Extractions** - Q&A pair data for fine-tuning
- **All Transcripts** - Speaker-separated transcripts
- **Semantic Chunks** - Conversation-focused chunks
- **Frame Alignments** - Visual citation data
- **Embeddings** - FAISS indices for search

## 🔗 **Webhook Integration**

### **AssemblyAI Webhook Management**

The transcription system uses webhooks to handle asynchronous processing:

```json
{
  "pending_transcriptions": {
    "TRANSCRIPT_ID": {
      "video_id": "VIDEO_ID",
      "video_title": "Video Title",
      "submitted_at": "TIMESTAMP",
      "status": "pending",
      "speaker_diarization": true
    }
  },
  "completed_transcriptions": {
    "TRANSCRIPT_ID": {
      "video_id": "VIDEO_ID",
      "video_title": "Video Title",
      "completed_at": "TIMESTAMP",
      "status": "completed",
      "speakers_detected": 2
    }
  }
}
```

### **Anti-Hanging Mechanisms**

1. **Duplicate Entry Cleanup**: Automatically removes videos appearing in both pending and completed
2. **Timeout Protection**: Maximum monitoring time (default: 1 hour)
3. **Speaker Validation**: Ensures speaker diarization completed successfully
4. **Safety Checks**: Multiple exit conditions in monitoring loops

## 🚀 **Execution Workflow**

### **Standard Pipeline Run**

```bash
# Run complete pipeline
uv run python run_conversations_pipeline.py

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
5. **Speaker Re-identification**: Re-processes speaker identification if needed

## 📁 **File Organization**

### **Directory Structure**

```
SCIENTIFIC_VIDEO_PIPELINE/Conversations_and_working_meetings_1_on_1/
├── step_01_raw/                          # Playlist metadata
├── step_02_extracted_playlist_content/   # Downloaded videos
├── step_03_transcription/                 # AssemblyAI transcripts with speakers
├── step_04_extract_chunks/               # Semantic chunks for conversations
├── step_05_frames/                       # Extracted video frames
├── step_06_frame_chunk_alignment/        # Frame-chunk mappings
├── step_07_faiss_embeddings/             # FAISS indices
├── pipeline_progress_queue.json           # Centralized state
├── assemblyai_webhooks.json              # Webhook management
└── run_conversations_pipeline.py         # Main pipeline runner
```

### **File Naming Conventions**

- **Videos**: `video_{VIDEO_ID}/video.mp4`
- **Transcripts**: `{VIDEO_ID}_transcript.json` + `.txt` (with speaker labels)
- **Chunks**: `{VIDEO_ID}_chunks.json` (conversation-focused)
- **Speaker Mappings**: `{VIDEO_ID}_speakers.json`
- **Q&A Pairs**: `{VIDEO_ID}_qa_pairs.json`
- **Frames**: `{VIDEO_ID}_frames_summary.json` + `{VIDEO_ID}/frame_*.jpg`
- **Embeddings**: `{TIMESTAMP}_*` prefixed files

## 🔧 **Troubleshooting & Maintenance**

### **Common Issues & Solutions**

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
- **Solution**: Check `pipeline_progress_queue.json` integrity

### **Debugging Commands**

```bash
# Check progress queue status
cat pipeline_progress_queue.json | jq '.'

# Check webhook status
cat assemblyai_webhooks.json | jq '.'

# View step logs
tail -f logs/transcription_webhook_*.log
tail -f logs/chunking_*.log
tail -f logs/speaker_identification_*.log

# Force reprocess specific step
rm -rf step_04_extract_chunks/*_chunks.json
uv run python step_04_extract_chunks.py
```

## 📈 **Performance Characteristics**

### **Execution Times**

| Scenario | Duration | Notes |
|----------|----------|-------|
| **First Run (2 videos)** | ~10-20 minutes | Full processing with speaker diarization |
| **Incremental (1 new video)** | ~10-20 minutes | Skip existing + process new |
| **Re-run (no changes)** | ~3-5 seconds | Skip everything |
| **Speaker Re-identification** | ~2-5 minutes | Re-process speaker mapping |
| **Q&A Re-extraction** | ~1-3 minutes | Re-process Q&A patterns |

### **Resource Usage**

- **CPU**: Intensive during LLM processing (step 4) and frame extraction (step 5)
- **Memory**: Peak usage during embedding generation (step 7)
- **Storage**: Frame extraction can generate significant disk usage
- **Network**: AssemblyAI API calls and video downloads

## 🔮 **Future Enhancements**

### **Planned Features**

1. **Advanced Speaker Recognition**: Use voice fingerprinting for more accurate identification
2. **Conversation Flow Analysis**: Analyze conversation dynamics and patterns
3. **Emotion Detection**: Identify emotional context in conversations
4. **Topic Segmentation**: Automatic topic boundary detection
5. **Multi-language Support**: Handle conversations in different languages

### **Scalability Improvements**

1. **Parallel Processing**: Independent steps could run concurrently
2. **Cloud Integration**: Support for cloud storage and compute
3. **Real-time Processing**: Stream processing for live conversations
4. **Advanced Caching**: Redis-based progress tracking and caching

## 📚 **Best Practices**

### **Development Workflow**

1. **Test with Short Conversations**: Use brief conversation videos for development
2. **Validate Speaker Identification**: Manually verify speaker mappings
3. **Review Q&A Extraction**: Ensure Q&A pairs are correctly identified
4. **Monitor Conversation Context**: Preserve dialogue flow in chunks

### **Production Deployment**

1. **Speaker Verification**: Implement manual review process for speaker identification
2. **Q&A Quality Control**: Review and validate extracted Q&A pairs
3. **Conversation Search Testing**: Test search functionality with conversation queries
4. **Fine-tuning Validation**: Verify Q&A data quality for training

## 🎯 **Conclusion**

The Conversations Video Pipeline represents a sophisticated, production-ready system for processing multi-speaker scientific conversations. With its specialized speaker identification, Q&A extraction, and conversation-focused chunking, it provides a robust foundation for analyzing conversations between Michael Levin and other researchers.

The system's architecture balances efficiency with accuracy, ensuring that:
- **Speaker identities are properly identified** through diarization and analysis
- **Conversation context is preserved** in semantic chunks
- **Q&A patterns are extracted** for fine-tuning purposes
- **System state is maintained** consistently across all operations
- **Errors are handled gracefully** with comprehensive logging and recovery

This makes it ideal for research environments where conversation analysis is critical and production systems requiring reliable, repeatable processing of multi-speaker content.
