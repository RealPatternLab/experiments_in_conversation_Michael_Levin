#!/usr/bin/env python3
"""
Base frame-chunk aligner class for video pipelines.
Provides common functionality while allowing pipeline-specific customization.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

from .pipeline_data_models import (
    PipelineType, BaseRAGOutput, BaseRAGEntry, FrameInfo, 
    ChunkMetadata, VisualContent, TemporalInfo, QualityMetrics, 
    ProcessingMetadata, create_rag_entry, create_rag_output
)

class BaseFrameChunkAligner(ABC):
    """Base class for frame-chunk alignment across different pipeline types."""
    
    def __init__(self, pipeline_type: PipelineType, progress_queue=None):
        """Initialize the base aligner."""
        self.pipeline_type = pipeline_type
        self.progress_queue = progress_queue
        
        # Common directories
        self.output_dir = Path("step_06_frame_chunk_alignment")
        self.frames_dir = Path("step_05_frames")
        self.chunks_dir = Path("step_04_extract_chunks")
        
        # Common parameters
        self.timestamp_tolerance = 5.0
        self.min_alignment_confidence = 0.7
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f'frame_chunk_alignment_{pipeline_type.value}')
    
    @abstractmethod
    def find_all_frames(self) -> Dict[str, List[Dict]]:
        """Find all frames across all videos. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def find_all_chunks(self) -> Dict[str, List[Dict]]:
        """Find all chunks across all videos. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def process_chunk_frames(self, chunk: Dict[str, Any], frames: List[Dict], 
                           chunk_start: float, chunk_end: float) -> List[FrameInfo]:
        """Process frames for a specific chunk. Must be implemented by subclasses."""
        pass
    
    def create_standard_rag_entry(self, chunk: Dict[str, Any], aligned_frames: List[FrameInfo],
                                 chunk_start: float, chunk_end: float, **kwargs) -> BaseRAGEntry:
        """Create a standard RAG entry using the shared data models."""
        
        # Create standard metadata
        chunk_metadata = ChunkMetadata(
            token_count=chunk.get('token_count', 0),
            sentence_count=chunk.get('sentence_count', 0),
            primary_topics=chunk.get('primary_topics', []),
            secondary_topics=chunk.get('secondary_topics', []),
            key_terms=chunk.get('key_terms', []),
            content_summary=chunk.get('content_summary', ''),
            scientific_domain=chunk.get('scientific_domain', ''),
            start_time_seconds=chunk_start,
            end_time_seconds=chunk_end
        )
        
        # Create visual content
        visual_content = VisualContent(
            frames=aligned_frames,
            frame_count=len(aligned_frames)
        )
        
        # Create temporal info
        temporal_info = TemporalInfo(
            frame_timestamps=[f.timestamp for f in aligned_frames],
            time_range={'start': chunk_start, 'end': chunk_end}
        )
        
        # Create quality metrics
        avg_confidence = sum(f.alignment_confidence for f in aligned_frames) / max(len(aligned_frames), 1)
        quality_metrics = QualityMetrics(
            total_frames=len(aligned_frames),
            average_confidence=avg_confidence,
            timestamp_coverage=len(aligned_frames) / max(len(aligned_frames), 1)
        )
        
        # Create processing metadata
        processing_metadata = ProcessingMetadata(
            alignment_method='timestamp_based',
            timestamp_tolerance=self.timestamp_tolerance
        )
        
        # Create base metadata
        metadata = {
            'pipeline_type': self.pipeline_type.value,
            'created_at': datetime.now().isoformat(),
            **kwargs
        }
        
        # Create the RAG entry using the factory function
        return create_rag_entry(
            pipeline_type=self.pipeline_type,
            content_id=chunk.get('chunk_id', chunk.get('id', 'unknown')),
            text_content={'text': chunk.get('text', ''), 'metadata': chunk_metadata.dict()},
            visual_content=visual_content,
            temporal_info=temporal_info,
            quality_metrics=quality_metrics,
            processing_metadata=processing_metadata,
            metadata=metadata
        )
    
    def create_standard_rag_output(self, alignments: List[BaseRAGEntry], **kwargs) -> BaseRAGOutput:
        """Create a standard RAG output using the shared data models."""
        
        metadata = {
            'pipeline_step': 'step_06_frame_chunk_alignment',
            'total_alignments': len(alignments),
            'alignment_method': 'timestamp_based',
            'processing_timestamp': datetime.now().isoformat(),
            'pipeline_type': self.pipeline_type.value,
            **kwargs
        }
        
        return create_rag_output(
            pipeline_type=self.pipeline_type,
            metadata=metadata,
            aligned_content=alignments
        )
    
    def save_alignments(self, alignments: List[BaseRAGEntry], 
                       rag_output: BaseRAGOutput, video_id: str):
        """Save alignment results in a standardized format."""
        try:
            # Save detailed alignments
            alignments_file = self.output_dir / f"{video_id}_alignments.json"
            with open(alignments_file, 'w') as f:
                json.dump([alignment.dict() for alignment in alignments], f, indent=2)
            
            # Save RAG-ready output
            rag_file = self.output_dir / f"{video_id}_rag_ready_aligned_content.json"
            with open(rag_file, 'w') as f:
                json.dump(rag_output.dict(), f, indent=2)
            
            # Save summary
            summary_file = self.output_dir / f"{video_id}_alignment_summary.json"
            summary = {
                'video_id': video_id,
                'pipeline_type': self.pipeline_type.value,
                'total_alignments': len(alignments),
                'processing_timestamp': datetime.now().isoformat(),
                'alignment_method': 'timestamp_based',
                'timestamp_tolerance': self.timestamp_tolerance,
                'files_created': [
                    str(alignments_file),
                    str(rag_file),
                    str(summary_file)
                ]
            }
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"✅ Saved alignments for {video_id}: {len(alignments)} entries")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save alignments for {video_id}: {e}")
            raise
    
    def validate_output(self, rag_output: BaseRAGOutput) -> bool:
        """Validate that the output conforms to the expected structure."""
        try:
            # This will raise an exception if validation fails
            rag_output.dict()
            self.logger.info("✅ Output validation passed")
            return True
        except Exception as e:
            self.logger.error(f"❌ Output validation failed: {e}")
            return False
    
    def get_common_processing_stats(self) -> Dict[str, Any]:
        """Get common processing statistics."""
        return {
            'pipeline_type': self.pipeline_type.value,
            'timestamp_tolerance': self.timestamp_tolerance,
            'min_alignment_confidence': self.min_alignment_confidence,
            'output_directory': str(self.output_dir),
            'frames_directory': str(self.frames_dir),
            'chunks_directory': str(self.chunks_dir)
        }
