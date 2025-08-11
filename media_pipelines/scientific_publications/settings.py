#!/usr/bin/env python3
"""
Configuration settings for the Michael Levin RAG system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')

# OpenAI Model
OPENAI_MODEL = "gpt-4o"

# Video Processing Settings
MAX_VIDEO_DURATION = 7200  # 2 hours in seconds
FRAME_EXTRACTION_INTERVAL = 5  # Extract frame every 5 seconds

# Updated YouTube paths to use ingested structure
YOUTUBE_VIDEOS_DIR = Path("data/ingested/youtube/videos")
YOUTUBE_TRANSCRIPTS_DIR = Path("data/ingested/youtube/transcripts")
YOUTUBE_FRAMES_DIR = Path("data/ingested/youtube/frames")
YOUTUBE_METADATA_DIR = Path("data/ingested/youtube/metadata")
YOUTUBE_METADATA_FILE = Path("data/ingested/youtube/metadata/youtube_metadata.json")

# FAISS Index Paths
FAISS_INDEX_DIR = Path("data/faiss")
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "chunks.index"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "chunks_metadata.pkl"
FAISS_EMBEDDINGS_PATH = FAISS_INDEX_DIR / "chunks_embeddings.npy"

# Output Paths (for legacy compatibility)
OUTPUTS_DIR = Path("outputs")
FAISS_INDEX_OUTPUT = OUTPUTS_DIR / "faiss_index.bin"
COMBINED_CHUNKS_OUTPUT = OUTPUTS_DIR / "combined_chunks.json" 

# GitHub configuration
GITHUB_BRANCH = "dev"  # Current branch name for GitHub raw URLs
GITHUB_REPO = "RealPatternLab/Michael-Levin-QA-Engine"  # Repository name 