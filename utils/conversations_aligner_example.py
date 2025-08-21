#!/usr/bin/env python3
"""
Example of how to refactor the conversations pipeline to use the base class.
This shows the reduced complexity and improved consistency.
"""

from .base_frame_chunk_aligner import BaseFrameChunkAligner
from .pipeline_data_models import PipelineType, FrameInfo
from typing import List, Dict, Any
import logging
import json

class ConversationsFrameChunkAligner(BaseFrameChunkAligner):
    """Conversations-specific frame-chunk aligner inheriting from base class."""
    
    def __init__(self, progress_queue=None):
        super().__init__(PipelineType.CONVERSATIONS, progress_queue)
        self.logger.info("Conversations frame-chunk aligner initialized")
    
    def find_all_frames(self) -> Dict[str, List[Dict]]:
        """Find all frames across all videos - conversations specific implementation."""
        frames_data = {}
        
        if not self.frames_dir.exists():
            self.logger.error("Frames directory not found")
            return {}
        
        # Find all frame summary files
        summary_files = list(self.frames_dir.glob("*_frames_summary.json"))
        self.logger.info(f"Found {len(summary_files)} frame summary files")
        
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                video_id = summary_data.get('video_id')
                frames = summary_data.get('frames', [])
                
                if video_id and frames:
                    frames_data[video_id] = frames
                    self.logger.info(f"Loaded {len(frames)} frames for {video_id}")
                    
            except Exception as e:
                self.logger.error(f"Error loading {summary_file}: {e}")
                continue
        
        return frames_data
    
    def find_all_chunks(self) -> Dict[str, List[Dict]]:
        """Find all semantic chunks across all videos - conversations specific implementation."""
        chunks_data = {}
        
        if not self.chunks_dir.exists():
            self.logger.error("Chunks directory not found")
            return {}
        
        # Look for enhanced chunks (our main semantic content)
        enhanced_chunks_dir = self.chunks_dir / "enhanced_chunks"
        if enhanced_chunks_dir.exists():
            chunk_files = list(enhanced_chunks_dir.glob("*.json"))
            self.logger.info(f"Found {len(chunk_files)} enhanced chunk files")
            
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r') as f:
                        chunks = json.load(f)
                    
                    # Extract video_id from filename
                    video_id = chunk_file.stem.replace('_enhanced_rag_chunks', '')
                    chunks_data[video_id] = chunks
                    self.logger.info(f"Loaded {len(chunks)} enhanced chunks for {video_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading {chunk_file}: {e}")
                    continue
        
        return chunks_data
    
    def process_chunk_frames(self, chunk: Dict[str, Any], frames: List[Dict], 
                           chunk_start: float, chunk_end: float) -> List[FrameInfo]:
        """Process frames for a specific chunk - conversations specific implementation."""
        aligned_frames = []
        
        # Find frames within the chunk's time range
        for frame in frames:
            frame_timestamp = frame.get('timestamp', 0)
            
            # Check if frame is within chunk time range
            if chunk_start <= frame_timestamp <= chunk_end:
                frame_info = FrameInfo(
                    frame_id=frame.get('frame_id', frame.get('filename', '')),
                    timestamp=frame_timestamp,
                    file_path=frame.get('file_path', ''),
                    file_size=frame.get('file_size', 0),
                    alignment_confidence=0.9  # High confidence for conversations
                )
                aligned_frames.append(frame_info)
        
        return aligned_frames
    
    def create_conversations_rag_entry(self, chunk: Dict[str, Any], aligned_frames: List[FrameInfo],
                                     chunk_start: float, chunk_end: float, video_id: str) -> Any:
        """Create a conversations-specific RAG entry with additional context."""
        
        # Use the base method to create standard structure
        base_entry = self.create_standard_rag_entry(
            chunk=chunk,
            aligned_frames=aligned_frames,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            video_id=video_id
        )
        
        # Add conversations-specific context
        if hasattr(base_entry, 'conversation_context'):
            base_entry.conversation_context = {
                'question': chunk.get('chunk_question', ''),
                'answer': chunk.get('chunk_answer', ''),
                'questioner': chunk.get('chunk_questioner', ''),
                'answerer': chunk.get('chunk_answerer', ''),
                'youtube_link': chunk.get('chunk_youtube_link', ''),
                'pipeline_type': 'conversations_1_on_1'
            }
        
        return base_entry
    
    def process_single_video(self, video_id: str, chunks: List[Dict], frames: List[Dict]) -> List[Any]:
        """Process a single video's chunks and frames."""
        alignments = []
        
        for chunk in chunks:
            # Extract timing information
            chunk_start = chunk.get('chunk_start_time', 0)
            chunk_end = chunk.get('chunk_end_time', 0)
            
            # Process frames for this chunk
            aligned_frames = self.process_chunk_frames(chunk, frames, chunk_start, chunk_end)
            
            # Create RAG entry
            rag_entry = self.create_conversations_rag_entry(
                chunk=chunk,
                aligned_frames=aligned_frames,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                video_id=video_id
            )
            
            alignments.append(rag_entry)
        
        return alignments

# Example usage:
def main():
    """Example of how to use the refactored aligner."""
    from pipeline_progress_queue import get_progress_queue
    
    # Initialize progress queue
    progress_queue = get_progress_queue()
    
    # Initialize the conversations aligner
    aligner = ConversationsFrameChunkAligner(progress_queue)
    
    # Find all frames and chunks
    frames_data = aligner.find_all_frames()
    chunks_data = aligner.find_all_chunks()
    
    # Process each video
    for video_id in chunks_data.keys():
        if video_id in frames_data:
            chunks = chunks_data[video_id]
            frames = frames_data[video_id]
            
            # Process the video
            alignments = aligner.process_single_video(video_id, chunks, frames)
            
            # Create RAG output
            rag_output = aligner.create_standard_rag_output(alignments)
            
            # Validate output
            if aligner.validate_output(rag_output):
                # Save results
                aligner.save_alignments(alignments, rag_output, video_id)
            else:
                aligner.logger.error(f"Output validation failed for {video_id}")

if __name__ == "__main__":
    main()
