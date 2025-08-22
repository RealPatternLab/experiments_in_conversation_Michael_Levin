# 🚀 Quick Start Guide: 1-on-2 Conversations Pipeline

## Overview

This guide will help you get started with the **1-on-2 Conversations Pipeline** in under 10 minutes. This pipeline processes YouTube videos featuring conversations between Michael Levin and 2 other researchers, providing rich multi-speaker analysis and semantic content extraction.

## 🎯 **What You'll Accomplish**

By following this guide, you'll:
- Set up the 1-on-2 pipeline environment
- Process your first 3-speaker conversation video
- Generate semantic chunks with multi-speaker context
- Create Q&A pairs with collaboration patterns
- Build searchable embeddings for research analysis

## ⚡ **Prerequisites (2 minutes)**

### 1. Environment Setup
```bash
# Navigate to the pipeline directory
cd SCIENTIFIC_VIDEO_PIPELINE/Conversations_and_working_meetings_1_on_2

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. API Keys
Create a `.env` file in the pipeline directory:
```bash
# Required API keys
OPENAI_API_KEY=your_openai_api_key_here
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Optional: YouTube API key for enhanced metadata
YOUTUBE_API_KEY=your_youtube_api_key_here
```

## 🎬 **Step 1: Prepare Your YouTube Playlist (1 minute)**

### Edit the playlist file:
```bash
# Open the playlist file
nano step_01_raw/youtube_playlist.txt
```

### Add your video URLs:
```txt
# Replace the example URLs with your actual 3-speaker conversation videos
https://www.youtube.com/watch?v=YOUR_VIDEO_ID_1
https://www.youtube.com/watch?v=YOUR_VIDEO_ID_2
https://www.youtube.com/watch?v=YOUR_VIDEO_ID_3
```

**Important**: Ensure your videos feature exactly 3 speakers (Michael Levin + 2 others)

## 🔄 **Step 2: Run the Complete Pipeline (5 minutes)**

### Option A: Run All Steps at Once
```bash
# Run the complete pipeline
python run_conversations_pipeline.py
```

### Option B: Run Steps Individually
```bash
# Step 1: Process playlist
python step_01_playlist_processor.py

# Step 2: Download videos
python step_02_video_downloader.py

# Step 3: Transcribe with speaker diarization
python step_03_transcription_webhook.py

# Step 4: Multi-speaker semantic chunking (INTERACTIVE)
python step_04_extract_chunks.py

# Step 5: Extract video frames
python step_05_frame_extractor.py

# Step 6: Align frames with chunks
python step_06_frame_chunk_alignment.py

# Step 7: Generate embeddings
python step_07_consolidated_embedding.py
```

## 🎭 **Step 3: Speaker Identification (2 minutes)**

When you run **Step 4**, the pipeline will interactively identify speakers:

### Interactive Process:
```
🎭 SPEAKER IDENTIFICATION FOR 1-ON-2 CONVERSATION
Video: YOUR_VIDEO_ID
Available speakers: A, B, C
==================================================

🎯 STEP 1: IDENTIFY MICHAEL LEVIN
==================================================
Which speaker is Michael Levin? (A, B, C): B
✅ Michael Levin identified as Speaker B

🎭 STEP 2: IDENTIFY SPEAKER 2
==================================================
Speaker ID: A
Enter the name for Speaker A: Dr. Jane Smith
Enter the role/organization for Dr. Jane Smith: Biologist, Stanford
Any additional context about Dr. Jane Smith? (optional): Expert in regeneration
✅ Speaker A identified as: Dr. Jane Smith (Biologist, Stanford)

🎭 STEP 3: IDENTIFY SPEAKER 3
==================================================
Speaker ID: C
Enter the name for Speaker C: Dr. John Doe
Enter the role/organization for Dr. John Doe: Physicist, MIT
Any additional context about Dr. John Doe? (optional): 
✅ Speaker C identified as: Dr. John Doe (Physicist, MIT)
```

## 📊 **Step 4: Explore Your Results (2 minutes)**

### Check the outputs:
```bash
# View generated chunks
ls step_04_extract_chunks/

# View Q&A pairs
cat step_04_extract_chunks/YOUR_VIDEO_ID_qa_pairs.json

# View multi-speaker analysis
cat step_04_extract_chunks/YOUR_VIDEO_ID_multi_speaker_analysis.json

# View individual chunks
ls step_04_extract_chunks/YOUR_VIDEO_ID_chunks/
```

### Sample Output Structure:
```json
{
  "pair_id": "YOUR_VIDEO_ID_qa_001",
  "question": {
    "speaker": "A",
    "text": "What is your definition of intelligence?",
    "speaker_name": "Dr. Jane Smith",
    "speaker_role": "Biologist, Stanford"
  },
  "answer": {
    "speaker": "B",
    "text": "I define intelligence as goal-directed problem solving...",
    "speaker_name": "Michael Levin",
    "speaker_role": "Biologist, Tufts University"
  },
  "elaboration": {
    "speaker": "C",
    "text": "That's fascinating because in physics...",
    "speaker_name": "Dr. John Doe",
    "speaker_role": "Physicist, MIT"
  },
  "multi_speaker_dynamics": {
    "interaction_pattern": "question_response_elaboration",
    "collaboration_type": "academic_discussion"
  }
}
```

## 🔍 **Step 5: Advanced Usage (Optional)**

### Customize Speaker Identification:
```bash
# Edit speaker mappings manually
nano step_04_extract_chunks/speaker_mappings.json
```

### Process Multiple Videos:
```bash
# Add more URLs to your playlist
echo "https://www.youtube.com/watch?v=NEW_VIDEO_ID" >> step_01_raw/youtube_playlist.txt

# Re-run the pipeline
python run_conversations_pipeline.py --start-from 1
```

### Analyze Specific Steps:
```bash
# Start from transcription step
python run_conversations_pipeline.py --start-from 3

# Start from chunking step
python run_conversations_pipeline.py --start-from 4
```

## 🚨 **Troubleshooting**

### Common Issues:

#### 1. "Expected 3 speakers, found X"
- **Cause**: Video has different number of speakers than expected
- **Solution**: Ensure your video has exactly 3 speakers, or use the 1-on-1 pipeline instead

#### 2. "API key not found"
- **Cause**: Missing or incorrect API keys
- **Solution**: Check your `.env` file and API key validity

#### 3. "Speaker identification failed"
- **Cause**: AssemblyAI transcription issues
- **Solution**: Check video audio quality and AssemblyAI API status

#### 4. "Pipeline step failed"
- **Cause**: Script execution error
- **Solution**: Check logs in `logs/` directory for detailed error information

### Get Help:
```bash
# Check pipeline logs
tail -f logs/pipeline_conversations_1_on_2_*.log

# Check step-specific logs
tail -f logs/step_*.log

# Run with verbose logging
python run_conversations_pipeline.py --verbose
```

## 🎯 **Next Steps**

### Immediate Actions:
1. **Test with a small video** (5-10 minutes) to verify setup
2. **Review speaker identification** for accuracy
3. **Examine generated chunks** for quality

### Advanced Features:
1. **Custom scientific taxonomies** for your research domain
2. **Enhanced collaboration pattern analysis**
3. **Integration with other pipelines** for unified search
4. **Custom Q&A extraction rules**

### Research Applications:
1. **Collaboration network analysis** across multiple videos
2. **Speaker role evolution** tracking
3. **Cross-video pattern recognition**
4. **AI training data generation** for conversation models

## 🎉 **Success Metrics**

You've successfully set up the 1-on-2 pipeline when:
- ✅ Pipeline runs without errors
- ✅ Speakers are correctly identified
- ✅ Semantic chunks are generated with multi-speaker context
- ✅ Q&A pairs include collaboration patterns
- ✅ Output files are properly structured and searchable

## 📚 **Additional Resources**

- **Full Documentation**: `README.md`
- **Speaker System**: `SPEAKER_IDENTIFICATION_SYSTEM.md`
- **Pipeline Comparison**: `docs/PIPELINE_COMPARISON.md`
- **Architecture**: `../ARCHITECTURE.md`

## 🆘 **Need Help?**

- **Check logs**: `logs/` directory
- **Review documentation**: Start with this guide, then `README.md`
- **Compare with 1-on-1**: See `../Conversations_and_working_meetings_1_on_1/`
- **Validate data**: Use shared validation functions from `utils/`

---

**Congratulations!** You're now ready to analyze multi-speaker conversations with Michael Levin and extract rich semantic content for your research. 🚀
