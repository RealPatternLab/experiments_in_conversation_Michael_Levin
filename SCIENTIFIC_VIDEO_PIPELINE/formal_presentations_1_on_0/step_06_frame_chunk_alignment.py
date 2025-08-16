#!/usr/bin/env python3
"""
Step 06: Frame-Chunk Alignment
Aligns extracted video frames with transcript chunks based on timestamps.
Creates RAG-ready output with multimodal content alignment.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frame_chunk_alignment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FrameChunkAligner:
    def __init__(self):
        self.chunks_dir = Path("step_04_extract_chunks")
        self.frames_dir = Path("step_05_frames")
        self.output_dir = Path("step_06_frame_chunk_alignment")
        self.output_dir.mkdir(exist_ok=True)
        
        # Alignment parameters
        self.timestamp_tolerance = 5.0  # Seconds tolerance for alignment
        self.min_alignment_confidence = 0.7  # Minimum confidence for alignment
    
    def process_all_videos(self):
        """Process all videos in the input directory"""
        if not self.chunks_dir.exists():
            logger.error(f"Chunks directory {self.chunks_dir} does not exist")
            return
        
        if not self.frames_dir.exists():
            logger.error(f"Frames directory {self.frames_dir} does not exist")
            return
        
        # Find all chunk files
        chunk_files = list(self.chunks_dir.glob("*_chunks.json"))
        logger.info(f"Found {len(chunk_files)} chunk files")
        
        for chunk_file in chunk_files:
            try:
                self.process_single_video(chunk_file)
            except Exception as e:
                logger.error(f"Failed to process {chunk_file.name}: {e}")
    
    def process_single_video(self, chunk_file: Path):
        """Process a single video's chunks and frames"""
        video_id = chunk_file.stem.replace('_chunks', '')
        logger.info(f"Processing video: {video_id}")
        
        # Load chunks
        with open(chunk_file, 'r') as f:
            chunks_data = json.load(f)
        
        # Load frames
        frames_data = self.load_frames_data(video_id)
        
        if not frames_data:
            logger.warning(f"No frames found for {video_id}")
            return
        
        # Create alignments
        alignments = self.create_timestamp_based_alignment(frames_data, chunks_data)
        
        # Create RAG-ready output
        rag_output = self.create_rag_ready_output(alignments)
        
        # Save alignments
        self.save_alignments(alignments, rag_output, video_id)
        
        logger.info(f"Successfully processed video: {video_id}")
    
    def load_frames_data(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load frames data for a video"""
        frames_summary_file = self.frames_dir / f"{video_id}_frames_summary.json"
        
        if not frames_summary_file.exists():
            return None
        
        try:
            with open(frames_summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load frames data for {video_id}: {e}")
            return None
    
    def create_timestamp_based_alignment(self, frames_data: Dict[str, Any], 
                                       chunks_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create alignments based on timestamps"""
        alignments = []
        
        frames = frames_data.get('frames', [])
        chunks = chunks_data.get('chunks', [])
        
        if not frames or not chunks:
            logger.warning("No frames or chunks found for alignment")
            return alignments
        
        # Sort frames by timestamp
        frames.sort(key=lambda x: x.get('timestamp', 0))
        
        # Create frame timestamp lookup
        frame_timestamps = [(f.get('timestamp', 0), f) for f in frames]
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', 'unknown')
            
            # Extract actual chunk timestamps from metadata
            chunk_start_seconds = chunk.get('start_time_seconds', 0)
            chunk_end_seconds = chunk.get('end_time_seconds', 0)
            
            # Convert to milliseconds if needed (frames use seconds)
            chunk_start = chunk_start_seconds
            chunk_end = chunk_end_seconds
            
            logger.debug(f"Aligning chunk {chunk_id}: {chunk_start}s - {chunk_end}s")
            
            # Try to find frames that align with this chunk
            aligned_frames = []
            
            # Find frames that fall within this chunk's actual time range
            for frame_timestamp, frame_info in frame_timestamps:
                if chunk_start <= frame_timestamp <= chunk_end:
                    # Frame is within chunk time range
                    aligned_frames.append({
                        'frame_id': frame_info.get('frame_id'),
                        'timestamp': frame_timestamp,
                        'file_path': frame_info.get('file_path'),
                        'file_size': frame_info.get('file_size'),
                        'alignment_confidence': 0.9  # High confidence for timestamp-based
                    })
            
            # If no frames found within range, find closest frame
            if not aligned_frames and frame_timestamps:
                # Find the frame closest to the chunk's start time
                closest_frame = min(frame_timestamps, key=lambda x: abs(x[0] - chunk_start))
                aligned_frames.append({
                    'frame_id': closest_frame[1].get('frame_id'),
                    'timestamp': closest_frame[0],
                    'file_path': closest_frame[1].get('file_path'),
                    'file_size': closest_frame[1].get('file_size'),
                    'alignment_confidence': 0.6  # Lower confidence for closest match
                })
                logger.debug(f"No frames in range for {chunk_id}, using closest frame at {closest_frame[0]}s")
            
            # Create alignment entry
            alignment_entry = {
                'chunk_id': chunk_id,
                'chunk_text': chunk.get('text', ''),
                'chunk_metadata': {
                    'token_count': chunk.get('token_count', 0),
                    'sentence_count': chunk.get('sentence_count', 0),
                    'primary_topics': chunk.get('primary_topics', []),
                    'secondary_topics': chunk.get('secondary_topics', []),
                    'key_terms': chunk.get('key_terms', []),
                    'content_summary': chunk.get('content_summary', ''),
                    'scientific_domain': chunk.get('scientific_domain', ''),
                    'start_time_seconds': chunk_start,
                    'end_time_seconds': chunk_end
                },
                'aligned_frames': aligned_frames,
                'alignment_quality': {
                    'total_frames': len(aligned_frames),
                    'average_confidence': sum(f.get('alignment_confidence', 0) for f in aligned_frames) / max(len(aligned_frames), 1),
                    'timestamp_coverage': len(aligned_frames) / max(len(frames), 1)
                },
                'metadata': {
                    'alignment_method': 'timestamp_based',
                    'timestamp_tolerance': self.timestamp_tolerance
                }
            }
            
            alignments.append(alignment_entry)
        
        return alignments
    
    def create_rag_ready_output(self, alignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create RAG-ready output format"""
        rag_output = {
            'metadata': {
                'pipeline_step': 'step_06_frame_chunk_alignment',
                'total_alignments': len(alignments),
                'alignment_method': 'timestamp_based',
                'processing_timestamp': None  # Will be set when saved
            },
            'aligned_content': []
        }
        
        for alignment in alignments:
            # Create RAG entry
            rag_entry = {
                'content_id': alignment['chunk_id'],
                'text_content': {
                    'text': alignment['chunk_text'],
                    'metadata': alignment['chunk_metadata']
                },
                'visual_content': {
                    'frames': alignment['aligned_frames'],
                    'frame_count': len(alignment['aligned_frames'])
                },
                'temporal_info': {
                    'frame_timestamps': [f.get('timestamp', 0) for f in alignment['aligned_frames']],
                    'time_range': {
                        'start': min([f.get('timestamp', 0) for f in alignment['aligned_frames']]) if alignment['aligned_frames'] else 0,
                        'end': max([f.get('timestamp', 0) for f in alignment['aligned_frames']]) if alignment['aligned_frames'] else 0
                    }
                },
                'quality_metrics': alignment['alignment_quality'],
                'processing_metadata': alignment['metadata']
            }
            
            rag_output['aligned_content'].append(rag_entry)
        
        return rag_output
    
    def save_alignments(self, alignments: List[Dict[str, Any]], 
                       rag_output: Dict[str, Any], video_id: str):
        """Save alignment results"""
        try:
            # Save detailed alignments
            alignments_file = self.output_dir / f"{video_id}_alignments.json"
            with open(alignments_file, 'w') as f:
                json.dump(alignments, f, indent=2)
            
            logger.info(f"Alignments saved: {alignments_file}")
            
            # Save RAG-ready output
            rag_file = self.output_dir / f"{video_id}_rag_ready_aligned_content.json"
            with open(rag_file, 'w') as f:
                json.dump(rag_output, f, indent=2)
            
            logger.info(f"RAG-ready output saved: {rag_file}")
            
            # Save summary
            summary = {
                'video_id': video_id,
                'total_alignments': len(alignments),
                'total_frames_aligned': sum(len(a.get('aligned_frames', [])) for a in alignments),
                'average_alignment_confidence': sum(
                    a.get('alignment_quality', {}).get('average_confidence', 0) 
                    for a in alignments
                ) / max(len(alignments), 1),
                'output_files': {
                    'alignments': str(alignments_file),
                    'rag_ready': str(rag_file)
                }
            }
            
            summary_file = self.output_dir / f"{video_id}_alignment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Alignment summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save alignments: {e}")

def main():
    """Main execution function"""
    try:
        aligner = FrameChunkAligner()
        aligner.process_all_videos()
        logger.info("Frame-chunk alignment step completed successfully")
    except Exception as e:
        logger.error(f"Frame-chunk alignment step failed: {e}")
        raise

if __name__ == "__main__":
    main()
