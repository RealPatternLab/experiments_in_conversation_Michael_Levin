#!/usr/bin/env python3
"""
Shared data models for video pipeline data structures.
Ensures consistency across different pipeline types while allowing for pipeline-specific extensions.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class PipelineType(str, Enum):
    """Pipeline type enumeration."""
    FORMAL_PRESENTATIONS = "formal_presentations"
    CONVERSATIONS = "conversations"
    WORKING_MEETINGS = "working_meetings"

class FrameInfo(BaseModel):
    """Standard frame information structure."""
    frame_id: str = Field(..., description="Unique frame identifier")
    timestamp: float = Field(..., description="Frame timestamp in seconds")
    file_path: str = Field(..., description="Path to frame file")
    file_size: int = Field(0, description="Frame file size in bytes")
    alignment_confidence: float = Field(0.9, description="Confidence score for alignment")

class ChunkMetadata(BaseModel):
    """Standard chunk metadata structure."""
    token_count: int = Field(0, description="Number of tokens in chunk")
    sentence_count: int = Field(0, description="Number of sentences in chunk")
    primary_topics: List[str] = Field(default_factory=list, description="Primary topics")
    secondary_topics: List[str] = Field(default_factory=list, description="Secondary topics")
    key_terms: List[str] = Field(default_factory=list, description="Key terms extracted")
    content_summary: str = Field("", description="Brief content summary")
    scientific_domain: str = Field("", description="Scientific domain classification")
    start_time_seconds: float = Field(0, description="Start time in seconds")
    end_time_seconds: float = Field(0, description="End time in seconds")

class VisualContent(BaseModel):
    """Standard visual content structure."""
    frames: List[FrameInfo] = Field(default_factory=list, description="Aligned frames")
    frame_count: int = Field(0, description="Total number of frames")

class TemporalInfo(BaseModel):
    """Standard temporal information structure."""
    frame_timestamps: List[float] = Field(default_factory=list, description="Frame timestamps")
    time_range: Dict[str, float] = Field(default_factory=dict, description="Start/end times")

class QualityMetrics(BaseModel):
    """Standard quality metrics structure."""
    total_frames: int = Field(0, description="Total frames available")
    average_confidence: float = Field(0.0, description="Average alignment confidence")
    timestamp_coverage: float = Field(0.0, description="Timestamp coverage ratio")

class ProcessingMetadata(BaseModel):
    """Standard processing metadata structure."""
    alignment_method: str = Field("timestamp_based", description="Alignment method used")
    timestamp_tolerance: float = Field(5.0, description="Timestamp tolerance in seconds")

class BaseRAGEntry(BaseModel):
    """Base RAG entry structure shared across all pipelines."""
    content_id: str = Field(..., description="Unique content identifier")
    text_content: Dict[str, Any] = Field(..., description="Text content and metadata")
    visual_content: VisualContent = Field(..., description="Visual content information")
    temporal_info: TemporalInfo = Field(..., description="Temporal information")
    quality_metrics: QualityMetrics = Field(..., description="Quality metrics")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class FormalPresentationsRAGEntry(BaseRAGEntry):
    """RAG entry for formal presentations pipeline."""
    # Inherits all base fields
    # Can add formal presentation specific fields here if needed
    pass

class ConversationsRAGEntry(BaseRAGEntry):
    """RAG entry for conversations pipeline."""
    conversation_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation-specific context (Q&A, speakers, etc.)"
    )

class BaseRAGOutput(BaseModel):
    """Base RAG output structure shared across all pipelines."""
    metadata: Dict[str, Any] = Field(..., description="Pipeline metadata")
    aligned_content: List[BaseRAGEntry] = Field(..., description="Aligned content entries")

class FormalPresentationsRAGOutput(BaseRAGOutput):
    """RAG output for formal presentations pipeline."""
    aligned_content: List[FormalPresentationsRAGEntry] = Field(..., description="Formal presentation entries")

class ConversationsRAGOutput(BaseRAGOutput):
    """RAG output for conversations pipeline."""
    aligned_content: List[ConversationsRAGEntry] = Field(..., description="Conversation entries")

# Factory function to create appropriate RAG entry based on pipeline type
def create_rag_entry(pipeline_type: PipelineType, **kwargs) -> BaseRAGEntry:
    """Create a RAG entry of the appropriate type."""
    if pipeline_type == PipelineType.CONVERSATIONS:
        return ConversationsRAGEntry(**kwargs)
    elif pipeline_type == PipelineType.FORMAL_PRESENTATIONS:
        return FormalPresentationsRAGEntry(**kwargs)
    else:
        return BaseRAGEntry(**kwargs)

# Factory function to create appropriate RAG output based on pipeline type
def create_rag_output(pipeline_type: PipelineType, **kwargs) -> BaseRAGOutput:
    """Create a RAG output of the appropriate type."""
    if pipeline_type == PipelineType.CONVERSATIONS:
        return ConversationsRAGOutput(**kwargs)
    elif pipeline_type == PipelineType.FORMAL_PRESENTATIONS:
        return FormalPresentationsRAGOutput(**kwargs)
    else:
        return BaseRAGOutput(**kwargs)

# Validation function to ensure data consistency
def validate_pipeline_data(data: Dict[str, Any], pipeline_type: PipelineType) -> bool:
    """Validate that data conforms to the expected pipeline structure."""
    try:
        if pipeline_type == PipelineType.CONVERSATIONS:
            ConversationsRAGOutput(**data)
        elif pipeline_type == PipelineType.FORMAL_PRESENTATIONS:
            FormalPresentationsRAGOutput(**data)
        else:
            BaseRAGOutput(**data)
        return True
    except Exception as e:
        print(f"Validation failed for {pipeline_type}: {e}")
        return False

# Utility function to convert existing data to new structure
def migrate_to_standard_structure(existing_data: Dict[str, Any], pipeline_type: PipelineType) -> Dict[str, Any]:
    """Migrate existing data to the new standard structure."""
    # This function would handle conversion from old formats
    # Implementation depends on existing data structure
    pass
