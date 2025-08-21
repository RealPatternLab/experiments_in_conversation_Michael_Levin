# File Organization Guide

This document describes the organization of files and directories in the Michael Levin RAG System project.

## 📁 Directory Structure

### Root Directory
The root directory contains only essential project files and configuration:

- **Core Application Files**
  - `streamlit_app.py` - Main Streamlit web application
  - `webhook_server.py` - AssemblyAI webhook server
  - `pyproject.toml` - Project configuration and dependencies
  - `uv.lock` - Locked dependency versions
  - `requirements.txt` - Python dependencies
  - `webhook_requirements.txt` - Webhook-specific dependencies

- **Main Documentation**
  - `README.md` - Project overview and quick start
  - `ARCHITECTURE.md` - System architecture documentation
  - `QUICKSTART.md` - Quick start guide

- **Configuration Files**
  - `.env` - Environment variables (not in repo)
  - `.gitignore` - Git ignore patterns
  - `.streamlit/` - Streamlit configuration

### 📚 Documentation (`docs/`)
All project documentation is organized in the `docs/` directory:

- `README.md` - Internal documentation index
- `PIPELINE_OVERVIEW.md` - Pipeline system overview
- `TECHNICAL_ARCHITECTURE.md` - Technical implementation details
- `IMPLEMENTATION_PLAN.md` - Development roadmap
- `DOCUMENTATION_COMPLETE.md` - Documentation status
- `PIPELINE_SUMMARY.md` - Pipeline summary
- `WEBHOOK_SETUP_GUIDE.md` - Webhook configuration guide
- `CONVERSATION_MEMORY_FEATURES.md` - Conversation features
- `PIPELINE_EVOLUTION_PLAN.md` - Pipeline evolution strategy
- `SCIENTIFIC_PUBLICATIONS_PIPELINE.md` - Publications pipeline docs
- `DESIGN_DOCUMENT.md` - System design documentation

### 📝 Logs (`logs/`)
All application logs are centralized in the `logs/` directory:

- `interactions_*.jsonl` - User interaction logs
- `playlist_processing.log` - YouTube playlist processing logs
- `transcription_webhook.log` - AssemblyAI webhook logs

### 🎬 Media (`media/`)
Media files used by the application:

- `thinking_box.gif` - Loading animation
- `thinking_construction.mp4` - Video content

### 🛠️ Utilities (`utils/`)
Utility scripts and tools:

- `youtube_howto_compiled.py` - YouTube video processing utility

### 🔬 Pipeline Directories
Each pipeline has its own directory with consistent structure:

#### Publications Pipeline (`SCIENTIFIC_PUBLICATION_PIPELINE/`)
- `step_01_raw/` - Input PDFs
- `step_02_metadata/` - Extracted metadata
- `step_03_extracted_text/` - Raw text from PDFs
- `step_04_enriched_metadata/` - Enhanced metadata
- `step_05_semantic_chunks/` - Semantic text chunks
- `step_06_faiss_embeddings/` - FAISS vector indices
- `step_07_archive/` - Processed PDFs
- `logs/` - Pipeline-specific logs

#### Videos Pipeline (`SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/`)
- `step_01_raw/` - YouTube playlist URLs
- `step_02_extracted_playlist_content/` - Downloaded videos
- `step_03_transcription/` - Generated transcripts
- `step_04_extract_chunks/` - Semantic chunks
- `step_05_frames/` - Extracted video frames
- `step_06_frame_chunk_alignment/` - Frame-chunk alignment
- `step_07_faiss_embeddings/` - FAISS indices
- `step_08_archive/` - Processed content
- `logs/` - Pipeline-specific logs
- `docs/` - Pipeline-specific documentation

#### Conversations Pipeline (`SCIENTIFIC_VIDEO_PIPELINE/Conversations_and_working_meetings_1_on_1/`)
- `step_01_raw/` - YouTube playlist URLs
- `step_02_extracted_playlist_content/` - Downloaded videos
- `step_03_transcription/` - Transcripts with speaker diarization
- `step_04_extract_chunks/` - Q&A pairs and semantic chunks
- `step_05_frames/` - Extracted video frames
- `step_06_frame_chunk_alignment/` - Frame-chunk alignment
- `step_07_faiss_embeddings/` - FAISS indices
- `step_08_cleanup/` - Processed content
- `logs/` - Pipeline-specific logs
- `docs/` - Pipeline-specific documentation

## 📋 File Organization Rules

### ✅ What Goes Where

1. **Root Directory**: Only essential project files, main documentation, and configuration
2. **`docs/`**: All project documentation and guides
3. **`logs/`**: All application logs (centralized)
4. **`media/`**: Media files used by the application
5. **`utils/`**: Utility scripts and tools
6. **Pipeline directories**: Pipeline-specific code, data, and logs

### ❌ What NOT to Put in Root

- Temporary files
- Test scripts
- Debug scripts
- One-time migration scripts
- Individual log files
- Pipeline-specific documentation
- Media files not used by the app

### 🔄 File Movement History

The following files were moved during organization:

- **Log files moved to `logs/`:**
  - `playlist_processing.log` → `logs/playlist_processing.log`
  - `transcription_webhook.log` → `logs/transcription_webhook.log`

- **Documentation moved to `docs/`:**
  - `WEBHOOK_SETUP_GUIDE.md` → `docs/WEBHOOK_SETUP_GUIDE.md`
  - `CONVERSATION_MEMORY_FEATURES.md` → `docs/CONVERSATION_MEMORY_FEATURES.md`
  - `PIPELINE_EVOLUTION_PLAN.md` → `docs/PIPELINE_EVOLUTION_PLAN.md`
  - `SCIENTIFIC_PUBLICATIONS_PIPELINE.md` → `docs/SCIENTIFIC_PUBLICATIONS_PIPELINE.md`
  - `DESIGN_DOCUMENT.md` → `docs/DESIGN_DOCUMENT.md`

- **Media files moved to `media/`:**
  - `thinking_construction.mp4` → `media/thinking_construction.mp4`
  - `thinking_box.gif` → `media/thinking_box.gif`

- **Utility scripts moved to `utils/`:**
  - `youtube_howto_compiled.py` → `utils/youtube_howto_compiled.py`

- **Pipeline logs moved to pipeline-specific `logs/` directories:**
  - Publications pipeline logs → `SCIENTIFIC_PUBLICATION_PIPELINE/logs/`

### 🧹 Cleanup Actions Performed

1. **Removed temporary files:**
   - `test_timestamp_extraction.py`
   - `test_authentic_voice.py`
   - `test_improved_enhancement.py`
   - `generate_gemini_prompts.py`
   - `gemini_generated_prompts.json`
   - `fix_video_faiss_structure.py`
   - `test_unified_search.py`
   - `test_faiss_retriever.py`
   - `test_webhook_system.py`
   - `test_status_update.py`
   - `test_archive_status.py`
   - `debug_timestamps.py`
   - `migrate_embedding_timestamps.py`

2. **Updated file references:**
   - Updated `streamlit_app.py` to reference `media/thinking_box.gif`
   - Updated cleanup scripts to look for logs in correct locations

## 🚀 Maintaining Organization

### Adding New Files

1. **New documentation**: Place in `docs/`
2. **New utilities**: Place in `utils/`
3. **New media**: Place in `media/`
4. **New logs**: Place in appropriate `logs/` directory
5. **Pipeline-specific code**: Place in appropriate pipeline directory

### Running Cleanup

The cleanup scripts automatically handle:
- Log file truncation (keeping last 1000 lines)
- Cache directory cleanup
- Temporary file removal
- Old report cleanup

### Regular Maintenance

- Run cleanup scripts after major pipeline runs
- Check for new temporary files monthly
- Review file organization quarterly
- Update this document when adding new file types
