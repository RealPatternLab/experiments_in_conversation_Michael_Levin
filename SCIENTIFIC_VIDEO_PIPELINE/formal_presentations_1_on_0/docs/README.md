# Video Pipeline Documentation

This folder contains all documentation for the Scientific Video Pipeline.

## Contents

- **PIPELINE_ARCHITECTURE.md** - Comprehensive architecture overview and technical details
- **QUICK_REFERENCE.md** - Quick commands and troubleshooting guide
- **SMART_PROCESSING_SYSTEM.md** - Details about the intelligent skip logic and progress tracking
- **STEP_08_CLEANUP_GUIDE.md** - Guide for the cleanup step and disk space management

## Folder Organization

The pipeline has been reorganized for better maintainability:

```
SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/
├── docs/                    # 📚 All documentation files
│   ├── README.md           # This file
│   ├── PIPELINE_ARCHITECTURE.md
│   ├── QUICK_REFERENCE.md
│   ├── SMART_PROCESSING_SYSTEM.md
│   └── STEP_08_CLEANUP_GUIDE.md
├── logs/                    # 📝 All log files and reports
│   ├── *.log               # Step-specific log files
│   ├── pipeline_report_*.txt
│   └── cleanup_report_*.json
├── step_01_raw/            # Raw playlist data
├── step_02_extracted_playlist_content/  # Downloaded videos
├── step_03_transcription/  # Transcript files
├── step_04_extract_chunks/ # Semantic chunks
├── step_05_frames/         # Extracted video frames
├── step_06_frame_chunk_alignment/  # Frame-chunk alignments
├── step_07_faiss_embeddings/  # FAISS indices and embeddings
└── *.py                    # Pipeline step scripts
```

## Benefits of New Organization

1. **Cleaner Root Directory** - Main pipeline directory is now focused on scripts and data
2. **Centralized Logging** - All logs go to `logs/` folder with automatic cleanup
3. **Documentation Hub** - All docs are in one place for easy reference
4. **Better Maintainability** - Easier to find and manage different types of files
5. **Automatic Log Rotation** - Old logs are automatically cleaned up to prevent disk space issues

## Logging System

The pipeline now uses a centralized logging system (`logging_config.py`) that:

- Automatically creates timestamped log files in the `logs/` folder
- Provides consistent logging format across all steps
- Automatically cleans up old log files (keeps last 5 per step)
- Ensures all pipeline outputs are properly organized

## Usage

- **Documentation**: Check the `docs/` folder for comprehensive guides
- **Logs**: Check the `logs/` folder for execution logs and reports
- **Scripts**: Run pipeline steps from the root directory as usual

The reorganization is transparent to users - all existing commands continue to work as before.
