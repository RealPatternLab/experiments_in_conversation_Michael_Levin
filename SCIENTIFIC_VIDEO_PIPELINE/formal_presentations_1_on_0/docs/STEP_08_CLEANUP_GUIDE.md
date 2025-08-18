# Step 8: Pipeline Cleanup - Comprehensive Guide

## Overview

Step 8 is a comprehensive cleanup system that removes unnecessary files from the pipeline while preserving all essential data needed for Streamlit applications and pipeline tracking. This step helps manage disk space and keeps the repository clean and organized.

## 🎯 **Purpose**

The cleanup step addresses the common problem of pipeline repositories growing large over time due to:
- **Large video files** (`.mp4`) that are no longer needed after processing
- **Unreferenced frame files** that were extracted but not used in analysis
- **Temporary utility scripts** created during debugging
- **Old log files** and reports that accumulate over time
- **Python cache directories** that are automatically generated

## 🗂️ **What Gets Cleaned Up**

### **1. Video Files (Largest Space Savings)**
- **Target**: `.mp4` files in `step_02_extracted_playlist_content/video_*/`
- **Rationale**: Videos are only needed for initial processing (transcription, frame extraction)
- **Space Impact**: 20-100+ MB per video
- **Safety**: Only removes videos after all processing steps are complete

### **2. Unreferenced Frame Files**
- **Target**: Frame files not referenced in chunk alignments
- **Rationale**: Many frames are extracted but only a subset are used in analysis
- **Space Impact**: 10-50 MB per video
- **Safety**: Cross-references with `step_06_frame_chunk_alignment/*_alignments.json`

### **3. yt-dlp Artifacts**
- **Target**: `.info.json`, `.webp`, `.vtt`, `.description` files
- **Rationale**: These are temporary files from video download process
- **Space Impact**: 0.5-2 MB per video
- **Safety**: Only removes after video processing is complete

### **4. Temporary Utility Scripts**
- **Target**: One-time debugging and utility scripts
- **Examples**: `fix_webhook_status.py`, `update_step*_progress.py`, `remove_*.py`
- **Rationale**: These are not part of the core pipeline
- **Safety**: Whitelist approach - only removes known temporary scripts

### **5. Old Log Files**
- **Target**: Log files with more than 1000 lines
- **Action**: Truncates to last 1000 lines (preserves recent history)
- **Rationale**: Keeps logs manageable while preserving debugging information
- **Safety**: Never completely removes log files

### **6. Old Pipeline Reports**
- **Target**: Old `pipeline_report_*.txt` files
- **Action**: Keeps only the most recent report
- **Rationale**: Historical reports are not needed for current operations
- **Safety**: Always preserves at least one report

### **7. Python Cache**
- **Target**: `__pycache__` directories
- **Rationale**: These are automatically regenerated and not needed
- **Space Impact**: 1-10 MB
- **Safety**: Completely safe to remove

### **8. Audio Files**
- **Target**: Temporary `.wav`, `.mp3`, `.m4a` files
- **Rationale**: Audio is only needed for AssemblyAI transcription
- **Safety**: Only removes after transcription is complete

### **9. Submission Metadata**
- **Target**: `*_submission.json` files
- **Rationale**: These are internal AssemblyAI tracking files
- **Safety**: Not needed for analysis or Streamlit

## 🛡️ **What Gets Preserved**

### **Essential Pipeline Files**
- `pipeline_progress_queue.json` - Centralized state management
- `assemblyai_webhooks.json` - Webhook tracking
- All step scripts (`step_*.py`) - Core functionality

### **Analysis Results**
- All transcript files (`*_transcript.json`)
- All semantic chunk files (`*_chunks.json`)
- All frame-chunk alignment files (`*_alignments.json`)
- All embedding files and FAISS indices

### **Referenced Frames**
- Frame files that are actually used in chunk alignments
- Frame summary files (`*_frames_summary.json`)

### **Documentation & Reports**
- All documentation files (`.md`)
- Latest pipeline execution report
- Current cleanup reports

## 🚀 **Usage**

### **Basic Cleanup**
```bash
# Run cleanup step
uv run python step_08_cleanup.py
```

### **Dry Run (Recommended First)**
```bash
# See what would be removed without actually removing
uv run python step_08_cleanup.py --dry-run
```

### **Verbose Logging**
```bash
# Detailed logging for debugging
uv run python step_08_cleanup.py --verbose
```

### **As Part of Full Pipeline**
```bash
# Run complete pipeline including cleanup
uv run python run_video_pipeline_1_on_0.py
```

## 📊 **Expected Results**

### **Space Savings**
- **Small Pipeline** (2-3 videos): 0.1-0.3 GB
- **Medium Pipeline** (5-10 videos): 0.5-1.5 GB
- **Large Pipeline** (20+ videos): 2.0+ GB

### **File Count Reduction**
- **Typical**: 300-1000+ files removed
- **Videos**: 1 file per video processed
- **Frames**: 50-200 files per video (unreferenced)
- **Scripts**: 5-15 temporary files
- **Cache**: 1-10 directories

### **Execution Time**
- **Dry Run**: 1-5 seconds
- **Actual Cleanup**: 2-10 seconds
- **Large Repositories**: Up to 30 seconds

## 🔍 **Safety Features**

### **1. Dry Run Mode**
- Always run with `--dry-run` first
- Shows exactly what will be removed
- No files are actually deleted

### **2. Essential File Protection**
- Whitelist approach for critical files
- Never removes core pipeline components
- Preserves all analysis results

### **3. Cross-Reference Validation**
- Frames are only removed if not referenced in alignments
- Videos are only removed after all processing is complete
- Logs are truncated, not deleted

### **4. Comprehensive Logging**
- Detailed log of all operations
- Cleanup report saved with timestamp
- Easy to audit what was removed

## 📋 **Cleanup Report**

After each cleanup run, a detailed report is generated:

```json
{
  "cleanup_timestamp": "2025-08-17T14:42:49.742967",
  "total_space_freed_gb": 0.1,
  "total_space_freed_mb": 97.91,
  "files_removed": {
    "videos": 3,
    "unreferenced_frames": 301,
    "temp_scripts": 9,
    "yt_dlp_files": 9,
    "old_reports": 3,
    "cache_directories": 1
  },
  "preserved_essentials": [...],
  "execution_time_seconds": 0.043,
  "dry_run": false
}
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **"No files were removed"**
- **Cause**: Cleanup already run recently
- **Solution**: Check if cleanup was already performed
- **Verify**: Look for recent cleanup reports

#### **"Permission denied"**
- **Cause**: File permissions or file in use
- **Solution**: Check file permissions and close any open applications
- **Verify**: Ensure you have write access to the directory

#### **"Unexpected files removed"**
- **Cause**: Dry run not used or script modified
- **Solution**: Always use `--dry-run` first
- **Recovery**: Check cleanup report for details

### **Recovery Options**

#### **Restore from Backup**
```bash
# If you have a backup of the progress queue
cp pipeline_progress_queue.json.backup pipeline_progress_queue.json
```

#### **Re-run Pipeline Steps**
```bash
# Re-run specific steps to regenerate removed files
uv run python step_02_video_downloader.py  # Re-download videos
uv run python step_05_frame_extractor.py   # Re-extract frames
```

## 🔄 **Integration with Pipeline**

### **Progress Queue Integration**
- Step 8 is tracked in the progress queue
- Status: `completed|pending|failed`
- Can be skipped like other steps if already completed

### **Pipeline Flow**
1. **Steps 1-7**: Process videos and generate analysis
2. **Step 8**: Clean up unnecessary files
3. **Result**: Clean, organized repository ready for Streamlit

### **Automated Execution**
- Runs automatically as part of full pipeline
- Can be run independently when needed
- Safe to run multiple times

## 📈 **Best Practices**

### **When to Run Cleanup**
- **After pipeline completion**: Clean up processing artifacts
- **Before committing**: Remove temporary files from repository
- **When disk space is low**: Free up space for new videos
- **Periodically**: Keep repository organized

### **Before Running Cleanup**
1. **Ensure pipeline is complete**: All steps 1-7 finished
2. **Backup if needed**: Copy critical files to safe location
3. **Use dry run**: Always preview what will be removed
4. **Check disk space**: Verify you have enough space for cleanup

### **After Running Cleanup**
1. **Review cleanup report**: Verify expected results
2. **Test Streamlit**: Ensure application still works
3. **Check pipeline**: Verify pipeline can still run
4. **Archive report**: Keep cleanup report for reference

## 🎯 **Conclusion**

Step 8 cleanup is a powerful tool for maintaining a clean, efficient pipeline repository. It automatically identifies and removes unnecessary files while preserving all essential data for analysis and Streamlit applications.

**Key Benefits:**
- ✅ **Significant space savings** (0.1-2.0+ GB per run)
- ✅ **Automatic organization** of repository
- ✅ **Safe operation** with comprehensive protection
- ✅ **Easy recovery** if needed
- ✅ **Full integration** with pipeline workflow

**Remember**: Always use `--dry-run` first to preview what will be removed!
