#!/usr/bin/env python3
"""
Step 06: Frame-Chunk Alignment (Unified)
Aligns video frames with transcript chunks to provide visual context for the RAG system.
Supports both formal presentations and multi-speaker conversations.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class UnifiedFrameChunkAligner:
    """Unified frame-chunk aligner supporting multiple pipeline types"""
    
    def __init__(self, pipeline_config: Dict[str, Any], progress_queue=None):
        """
        Initialize the unified frame-chunk aligner
        
        Args:
            pipeline_config: Pipeline-specific configuration
            progress_queue: Progress tracking queue
        """
        self.config = pipeline_config
        self.progress_queue = progress_queue
        
        # Pipeline metadata
        self.pipeline_type = pipeline_config.get('pipeline_type', 'unknown')
        self.frame_config = pipeline_config.get('frame_extraction', {})
        self.alignment_config = pipeline_config.get('frame_alignment', {})
        self.llm_config = pipeline_config.get('llm_config', {})
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Initialize OpenAI client if LLM enhancement is enabled
        self.openai_client = None
        if pipeline_config.get('llm_enhancement', False):
            self._initialize_openai_client()
        
        # Input/output directories
        self.frames_dir = Path("step_05_frames")
        self.chunks_dir = Path("step_04_extract_chunks")
        self.output_dir = Path("step_06_frame_chunk_alignment")
        self.output_dir.mkdir(exist_ok=True)
        
        # Alignment parameters
        self.timestamp_tolerance = self.alignment_config.get('timestamp_tolerance', 5.0)
        self.min_alignment_confidence = self.alignment_config.get('min_alignment_confidence', 0.7)
        self.frame_strategy = self.alignment_config.get('strategy', 'single_frame_per_chunk')
        
        # Processing statistics
        self.stats = {
            'total_videos': 0,
            'new_alignments': 0,
            'existing_alignments': 0,
            'failed_alignments': 0,
            'total_chunks': 0,
            'chunks_with_frames': 0,
            'total_frames_aligned': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        try:
            # Try to import centralized logging
            from logging_config import setup_logging
            return setup_logging(f'{self.pipeline_type}_frame_alignment')
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(f'{self.pipeline_type}_frame_alignment')
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client for LLM enhancement"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                self.logger.info("✅ OpenAI client initialized for LLM enhancement")
            else:
                self.logger.warning("⚠️ OPENAI_API_KEY not found, LLM enhancement disabled")
        except ImportError:
            self.logger.warning("⚠️ OpenAI package not available, LLM enhancement disabled")
        except Exception as e:
            self.logger.warning(f"⚠️ OpenAI client initialization failed: {e}")
    
    def process_all_videos(self):
        """Process all videos in the input directory"""
        if not self.chunks_dir.exists():
            self.logger.error(f"Chunks directory {self.chunks_dir} does not exist")
            return
        
        if not self.frames_dir.exists():
            self.logger.error(f"Frames directory {self.frames_dir} does not exist")
            return
        
        # Find all chunk files
        chunk_files = list(self.chunks_dir.glob("*_chunks.json"))
        self.logger.info(f"Found {len(chunk_files)} chunk files")
        
        self.stats['total_videos'] = len(chunk_files)
        
        for chunk_file in chunk_files:
            try:
                result = self.process_single_video(chunk_file)
                if result == 'new':
                    self.stats['new_alignments'] += 1
                elif result == 'existing':
                    self.stats['existing_alignments'] += 1
                elif result == 'failed':
                    self.stats['failed_alignments'] += 1
            except Exception as e:
                self.logger.error(f"Failed to process {chunk_file.name}: {e}")
                self.stats['failed_alignments'] += 1
        
        self.log_processing_summary()
    
    def process_single_video(self, chunk_file: Path):
        """Process a single video's chunks and frames"""
        video_id = chunk_file.stem.replace('_chunks', '')
        self.logger.info(f"Processing video: {video_id}")
        
        # Check progress
        if self.progress_queue:
            video_status = self.progress_queue.get_video_status(video_id)
            if video_status and video_status.get('step_06_frame_chunk_alignment') == 'completed':
                self.logger.info(f"Alignments already completed for {video_id}, skipping")
                return 'existing'
        
        # Fallback: Check if alignment file exists
        alignment_file = self.output_dir / f"{video_id}_alignments.json"
        if alignment_file.exists():
            self.logger.info(f"Alignments already exist for {video_id}, skipping")
            return 'existing'
        
        try:
            # Load chunks
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)
            
            # Load frames
            frames = self._load_frames_for_video(video_id)
            if not frames:
                self.logger.warning(f"No frames found for {video_id}")
                return 'failed'
            
            # Align frames with chunks
            alignments = self._align_frames_with_chunks(chunks, frames, video_id)
            
            # Create RAG-ready output
            rag_output = self._create_rag_ready_output(chunks, alignments, video_id)
            
            # Save outputs
            self._save_outputs(video_id, alignments, rag_output)
            
            # Update progress
            if self.progress_queue:
                self.progress_queue.update_video_step(video_id, 'step_06_frame_chunk_alignment', 'completed', {
                    'chunks_processed': len(chunks),
                    'frames_aligned': self.stats['total_frames_aligned'],
                    'chunks_with_frames': self.stats['chunks_with_frames']
                })
            
            self.logger.info(f"✅ Successfully processed {video_id}")
            return 'new'
            
        except Exception as e:
            self.logger.error(f"Error processing {video_id}: {e}")
            return 'failed'
    
    def _load_frames_for_video(self, video_id: str) -> List[Dict]:
        """Load frames for a specific video"""
        try:
            # Look for frame summary file
            frame_summary_file = self.frames_dir / f"{video_id}_frames_summary.json"
            if frame_summary_file.exists():
                with open(frame_summary_file, 'r') as f:
                    summary_data = json.load(f)
                return summary_data.get('frames', [])
            
            # Fallback: look for frame directory
            frame_dir = self.frames_dir / video_id
            if frame_dir.exists() and frame_dir.is_dir():
                frames = []
                for frame_file in frame_dir.glob("*.jpg"):
                    # Extract timestamp from filename or use file modification time
                    timestamp = self._extract_timestamp_from_filename(frame_file.name)
                    frames.append({
                        'filename': frame_file.name,
                        'path': str(frame_file),
                        'timestamp': timestamp
                    })
                return frames
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Failed to load frames for {video_id}: {e}")
            return []
    
    def _extract_timestamp_from_filename(self, filename: str) -> float:
        """Extract timestamp from frame filename"""
        try:
            # Common patterns: frame_001_30.5s.jpg, 001_30.5.jpg, etc.
            import re
            match = re.search(r'(\d+\.?\d*)s?', filename)
            if match:
                return float(match.group(1))
        except:
            pass
        
        # Fallback: use file modification time
        return 0.0
    
    def _align_frames_with_chunks(self, chunks: List[Dict], frames: List[Dict], video_id: str) -> List[Dict]:
        """Align frames with chunks based on pipeline type"""
        alignments = []
        
        for chunk in chunks:
            chunk_start = chunk.get('start_time', 0)
            chunk_end = chunk.get('end_time', 0)
            chunk_center = (chunk_start + chunk_end) / 2
            
            # Find relevant frames
            relevant_frames = []
            for frame in frames:
                frame_time = frame.get('timestamp', 0)
                if abs(frame_time - chunk_center) <= self.timestamp_tolerance:
                    relevant_frames.append(frame)
            
            # Create alignment based on strategy
            if self.frame_strategy == 'single_frame_per_chunk':
                alignment = self._create_single_frame_alignment(chunk, relevant_frames, chunk_center)
            else:
                alignment = self._create_multi_frame_alignment(chunk, relevant_frames, chunk_center)
            
            alignments.append(alignment)
            
            # Update statistics
            if alignment.get('frame'):
                self.stats['chunks_with_frames'] += 1
                self.stats['total_frames_aligned'] += 1
        
        return alignments
    
    def _create_single_frame_alignment(self, chunk: Dict, relevant_frames: List[Dict], chunk_center: float) -> Dict:
        """Create single frame alignment (best practice from conversations pipeline)"""
        if not relevant_frames:
            return {
                'chunk_id': chunk.get('chunk_id'),
                'frame': None,
                'frame_count': 0,
                'alignment_quality': 0.0,
                'timestamp_difference': float('inf')
            }
        
        # Select the frame closest to chunk center
        best_frame = min(relevant_frames, key=lambda f: abs(f.get('timestamp', 0) - chunk_center))
        timestamp_diff = abs(best_frame.get('timestamp', 0) - chunk_center)
        
        # Calculate alignment quality
        quality = self._calculate_alignment_quality(timestamp_diff, self.timestamp_tolerance)
        
        return {
            'chunk_id': chunk.get('chunk_id'),
            'frame': best_frame,
            'frame_count': 1,
            'alignment_quality': quality,
            'timestamp_difference': timestamp_diff
        }
    
    def _create_multi_frame_alignment(self, chunk: Dict, relevant_frames: List[Dict], chunk_center: float) -> Dict:
        """Create multi-frame alignment (legacy support)"""
        if not relevant_frames:
            return {
                'chunk_id': chunk.get('chunk_id'),
                'frames': [],
                'frame_count': 0,
                'alignment_quality': 0.0,
                'timestamp_difference': float('inf')
            }
        
        # Calculate quality for each frame
        frame_qualities = []
        for frame in relevant_frames:
            timestamp_diff = abs(frame.get('timestamp', 0) - chunk_center)
            quality = self._calculate_alignment_quality(timestamp_diff, self.timestamp_tolerance)
            frame_qualities.append((frame, quality, timestamp_diff))
        
        # Sort by quality
        frame_qualities.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'chunk_id': chunk.get('chunk_id'),
            'frames': [f[0] for f in frame_qualities],
            'frame_count': len(relevant_frames),
            'alignment_quality': frame_qualities[0][1] if frame_qualities else 0.0,
            'timestamp_difference': frame_qualities[0][2] if frame_qualities else float('inf')
        }
    
    def _calculate_alignment_quality(self, timestamp_diff: float, tolerance: float) -> float:
        """Calculate alignment quality score"""
        if timestamp_diff <= tolerance:
            # Perfect alignment within tolerance
            return 1.0
        elif timestamp_diff <= tolerance * 2:
            # Good alignment within 2x tolerance
            return 0.8
        elif timestamp_diff <= tolerance * 3:
            # Acceptable alignment within 3x tolerance
            return 0.6
        else:
            # Poor alignment
            return max(0.1, 1.0 - (timestamp_diff / tolerance) * 0.5)
    
    def _create_rag_ready_output(self, chunks: List[Dict], alignments: List[Dict], video_id: str) -> Dict:
        """Create RAG-ready output with visual context"""
        rag_content = []
        
        for chunk, alignment in zip(chunks, alignments):
            # Create visual context summary
            visual_context = self._create_visual_context_summary(chunk, alignment)
            
            # Create RAG-ready chunk
            rag_chunk = {
                'chunk_id': chunk.get('chunk_id'),
                'video_id': video_id,
                'pipeline_type': self.pipeline_type,
                'text_content': {
                    'text': chunk.get('semantic_text', ''),
                    'metadata': {
                        'start_time': chunk.get('start_time', 0),
                        'end_time': chunk.get('end_time', 0),
                        'timestamp': chunk.get('timestamp', ''),
                        'content_summary': chunk.get('content_summary', ''),
                        'primary_topics': chunk.get('primary_topics', []),
                        'secondary_topics': chunk.get('secondary_topics', []),
                        'key_terms': chunk.get('key_terms', [])
                    }
                },
                'visual_content': {
                    'frame': alignment.get('frame') if self.frame_strategy == 'single_frame_per_chunk' else None,
                    'frames': alignment.get('frames') if self.frame_strategy != 'single_frame_per_chunk' else None,
                    'frame_count': alignment.get('frame_count', 0),
                    'alignment_quality': alignment.get('alignment_quality', 0.0)
                },
                'temporal_info': {
                    'chunk_duration': chunk.get('end_time', 0) - chunk.get('start_time', 0),
                    'frame_timing': alignment.get('timestamp_difference', 0),
                    'alignment_confidence': 'high' if alignment.get('alignment_quality', 0) > 0.7 else 'medium'
                },
                'conversation_context': chunk.get('conversation_context', {}),
                'created_at': datetime.now().isoformat()
            }
            
            rag_content.append(rag_chunk)
        
        return {
            'video_id': video_id,
            'pipeline_type': self.pipeline_type,
            'total_chunks': len(chunks),
            'chunks_with_frames': self.stats['chunks_with_frames'],
            'total_frames_aligned': self.stats['total_frames_aligned'],
            'alignment_strategy': self.frame_strategy,
            'content': rag_content,
            'created_at': datetime.now().isoformat()
        }
    
    def _create_visual_context_summary(self, chunk: Dict, alignment: Dict) -> str:
        """Create visual context summary for the chunk"""
        if not alignment.get('frame') and not alignment.get('frames'):
            return "No visual content available for this chunk."
        
        frame_count = alignment.get('frame_count', 0)
        quality = alignment.get('alignment_quality', 0.0)
        
        if self.frame_strategy == 'single_frame_per_chunk':
            frame = alignment.get('frame')
            if frame:
                timestamp = frame.get('timestamp', 'unknown')
                return f"Single frame captured at {timestamp}s with {quality:.1%} alignment quality."
        else:
            frames = alignment.get('frames', [])
            if frames:
                return f"{frame_count} frames available with {quality:.1%} alignment quality."
        
        return "Visual content available but details unavailable."
    
    def _save_outputs(self, video_id: str, alignments: List[Dict], rag_output: Dict):
        """Save alignment and RAG outputs"""
        try:
            # Save alignments
            alignments_file = self.output_dir / f"{video_id}_alignments.json"
            with open(alignments_file, 'w') as f:
                json.dump(alignments, f, indent=2)
            
            # Save RAG-ready output
            rag_file = self.output_dir / f"{video_id}_rag_ready_aligned_content.json"
            with open(rag_file, 'w') as f:
                json.dump(rag_output, f, indent=2)
            
            # Save alignment summary
            summary = {
                'video_id': video_id,
                'pipeline_type': self.pipeline_type,
                'total_chunks': len(alignments),
                'chunks_with_frames': self.stats['chunks_with_frames'],
                'total_frames_aligned': self.stats['total_frames_aligned'],
                'alignment_strategy': self.frame_strategy,
                'timestamp_tolerance': self.timestamp_tolerance,
                'min_alignment_confidence': self.min_alignment_confidence,
                'created_at': datetime.now().isoformat()
            }
            
            summary_file = self.output_dir / f"{video_id}_alignment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"All outputs saved for {video_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving outputs for {video_id}: {e}")
            raise
    
    def log_processing_summary(self):
        """Log a summary of processing statistics"""
        self.logger.info(f"🎉 {self.pipeline_type.upper()} Frame-Chunk Alignment Summary:")
        self.logger.info(f"  Total videos: {self.stats['total_videos']}")
        self.logger.info(f"  New alignments: {self.stats['new_alignments']}")
        self.logger.info(f"  Existing alignments: {self.stats['existing_alignments']}")
        self.logger.info(f"  Failed: {self.stats['failed_alignments']}")
        self.logger.info(f"  Total chunks: {self.stats['total_chunks']}")
        self.logger.info(f"  Chunks with frames: {self.stats['chunks_with_frames']}")
        self.logger.info(f"  Total frames aligned: {self.stats['total_frames_aligned']}")
        
        if self.stats['total_videos'] > 0:
            success_rate = ((self.stats['new_alignments'] + self.stats['existing_alignments']) / self.stats['total_videos']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        if self.stats['total_chunks'] > 0:
            frame_coverage = (self.stats['chunks_with_frames'] / self.stats['total_chunks']) * 100
            self.logger.info(f"  Frame coverage: {frame_coverage:.1f}%")
