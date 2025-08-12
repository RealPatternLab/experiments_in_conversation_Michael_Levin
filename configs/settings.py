#!/usr/bin/env python3
"""
Configuration settings for the Michael Levin RAG system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GitHub configuration - used by streamlit_app.py for PDF citations and image URLs
GITHUB_BRANCH = "dev"  # Current branch name for GitHub raw URLs
GITHUB_REPO = "RealPatternLab/experiments_in_conversation_Michael_Levin"  # Repository name 