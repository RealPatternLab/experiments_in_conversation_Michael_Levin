#!/usr/bin/env python3
"""
Step 06: Frame-Chunk Alignment for Conversations Pipeline
Aligns video frames with semantic chunks to provide visual context for the RAG system.
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False
    logging.warning("python-dotenv not available. Environment variables may not load properly.")

# Import centralized logging configuration
from logging_config import setup_logging
from pipeline_progress_queue import get_progress_queue

# Configure logging
logger = setup_logging('frame_chunk_alignment')

class FrameChunkAligner:
    """Aligns video frames with semantic chunks for visual context."""
    
    def __init__(self, progress_queue=None):
        """Initialize the frame-chunk aligner."""
        self.output_dir = Path("step_06_frame_chunk_alignment")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = progress_queue
        
        # Input directories
        self.frames_dir = Path("step_05_frames")
        self.chunks_dir = Path("step_04_extract_chunks")
        
        logger.info("Frame-chunk aligner initialized successfully")
    
    def find_all_frames(self) -> Dict[str, List[Dict]]:
        """Find all frames across all videos."""
        frames_data = {}
        
        if not self.frames_dir.exists():
            logger.error("Frames directory not found")
            return {}
        
        # Find all frame summary files
        summary_files = list(self.frames_dir.glob("*_frames_summary.json"))
        logger.info(f"Found {len(summary_files)} frame summary files")
        
        for summary_file in summary_files:
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                video_id = summary_data.get('video_id')
                frames = summary_data.get('frames', [])
                
                if video_id and frames:
                    frames_data[video_id] = frames
                    logger.info(f"Loaded {len(frames)} frames for {video_id}")
                    
            except Exception as e:
                logger.error(f"Error loading {summary_file}: {e}")
                continue
        
        return frames_data
    
    def find_all_chunks(self) -> Dict[str, List[Dict]]:
        """Find all semantic chunks across all videos."""
        chunks_data = {}
        
        if not self.chunks_dir.exists():
            logger.error("Chunks directory not found")
            return {}
        
        # Look for enhanced chunks (our main semantic content)
        enhanced_chunks_dir = self.chunks_dir / "enhanced_chunks"
        if enhanced_chunks_dir.exists():
            chunk_files = list(enhanced_chunks_dir.glob("*.json"))
            logger.info(f"Found {len(chunk_files)} enhanced chunk files")
            
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r') as f:
                        chunks = json.load(f)
                    
                    # Extract video_id from filename
                    video_id = chunk_file.stem.replace('_enhanced_rag_chunks', '')
                    chunks_data[video_id] = chunks
                    logger.info(f"Loaded {len(chunks)} enhanced chunks for {video_id}")
                    
                except Exception as e:
                    logger.error(f"Error loading {chunk_file}: {e}")
                    continue
        
        # Also look for Levin chunks if enhanced chunks not found
        if not chunks_data:
            levin_chunk_files = list(self.chunks_dir.glob("*_levin_chunks.json"))
            logger.info(f"Found {len(levin_chunk_files)} Levin chunk files")
            
            for chunk_file in levin_chunk_files:
                try:
                    with open(chunk_file, 'r') as f:
                        chunks = json.load(f)
                    
                    video_id = chunk_file.stem.replace('_levin_chunks', '')
                    chunks_data[video_id] = chunks
                    logger.info(f"Loaded {len(chunks)} Levin chunks for {video_id}")
                    
                except Exception as e:
                    logger.error(f"Error loading {chunk_file}: {e}")
                    continue
        
        return chunks_data
    
    def align_frames_with_chunks(self, video_id: str, frames: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Align frames with chunks based on timestamps."""
        alignments = []
        
        logger.info(f"Aligning {len(frames)} frames with {len(chunks)} chunks for {video_id}")
        
        for chunk in chunks:
            try:
                # Get chunk timestamps
                start_time_ms = chunk.get('start_time_ms')
                end_time_ms = chunk.get('end_time_ms')
                
                if not start_time_ms or not end_time_ms:
                    logger.warning(f"Chunk {chunk.get('id', 'unknown')} missing timestamps, skipping")
                    continue
                
                # Convert to seconds for easier comparison
                start_time_s = start_time_ms / 1000
                end_time_s = end_time_ms / 1000
                
                # Find frames that fall within this chunk's time range
                relevant_frames = []
                for frame in frames:
                    frame_timestamp = frame.get('timestamp', 0)
                    
                    # Check if frame falls within chunk time range
                    if start_time_s <= frame_timestamp <= end_time_s:
                        relevant_frames.append(frame)
                
                # Create alignment entry
                alignment = {
                    'chunk_id': chunk.get('id', 'unknown'),
                    'chunk_text': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                    'chunk_start_time': start_time_s,
                    'chunk_end_time': end_time_s,
                    'chunk_start_timestamp': chunk.get('start_timestamp', ''),
                    'chunk_end_timestamp': chunk.get('end_timestamp', ''),
                    'chunk_topics': chunk.get('topics', []),
                    'chunk_question': chunk.get('original_question', ''),
                    'chunk_answer': chunk.get('original_answer', ''),
                    'chunk_questioner': chunk.get('questioner', ''),
                    'chunk_answerer': chunk.get('answerer', ''),
                    'chunk_youtube_link': chunk.get('youtube_link', ''),
                    'relevant_frames': relevant_frames,
                    'frame_count': len(relevant_frames),
                    'alignment_quality': self._calculate_alignment_quality(start_time_s, end_time_s, relevant_frames)
                }
                
                alignments.append(alignment)
                
            except Exception as e:
                logger.error(f"Error aligning chunk {chunk.get('id', 'unknown')}: {e}")
                continue
        
        return alignments
    
    def _calculate_alignment_quality(self, start_time: float, end_time: float, frames: List[Dict]) -> str:
        """Calculate the quality of frame-chunk alignment."""
        if not frames:
            return "no_frames"
        
        chunk_duration = end_time - start_time
        frame_count = len(frames)
        
        # Calculate frame density (frames per minute)
        frames_per_minute = (frame_count / chunk_duration) * 60
        
        if frames_per_minute >= 2.0:  # At least 2 frames per minute
            return "excellent"
        elif frames_per_minute >= 1.0:  # At least 1 frame per minute
            return "good"
        elif frames_per_minute >= 0.5:  # At least 1 frame per 2 minutes
            return "fair"
        else:
            return "poor"
    
    def create_visual_context_summary(self, video_id: str, alignments: List[Dict]) -> Dict[str, Any]:
        """Create a summary of visual context for the video."""
        total_chunks = len(alignments)
        chunks_with_frames = sum(1 for a in alignments if a['frame_count'] > 0)
        total_frames = sum(a['frame_count'] for a in alignments)
        
        # Calculate alignment quality distribution
        quality_distribution = {}
        for alignment in alignments:
            quality = alignment['alignment_quality']
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        # Find chunks with best visual coverage
        best_visual_chunks = sorted(alignments, key=lambda x: x['frame_count'], reverse=True)[:5]
        
        return {
            'video_id': video_id,
            'summary': {
                'total_chunks': total_chunks,
                'chunks_with_frames': chunks_with_frames,
                'chunks_without_frames': total_chunks - chunks_with_frames,
                'total_frames': total_frames,
                'average_frames_per_chunk': total_frames / total_chunks if total_chunks > 0 else 0,
                'visual_coverage_percentage': (chunks_with_frames / total_chunks * 100) if total_chunks > 0 else 0
            },
            'quality_distribution': quality_distribution,
            'best_visual_chunks': [
                {
                    'chunk_id': c['chunk_id'],
                    'frame_count': c['frame_count'],
                    'start_time': c['chunk_start_timestamp'],
                    'end_time': c['chunk_end_timestamp'],
                    'topics': c['chunk_topics']
                }
                for c in best_visual_chunks
            ],
            'processing_timestamp': datetime.now().isoformat(),
            'pipeline_type': 'conversations_1_on_1'
        }
    
    def process_all_videos(self) -> Dict[str, Any]:
        """Process all videos to align frames with chunks."""
        logger.info("🚀 Starting frame-chunk alignment for all videos...")
        
        # Find all frames and chunks
        frames_data = self.find_all_frames()
        chunks_data = self.find_all_chunks()
        
        if not frames_data:
            logger.error("No frames found")
            return {'success': False, 'message': 'No frames found'}
        
        if not chunks_data:
            logger.error("No chunks found")
            return {'success': False, 'message': 'No chunks found'}
        
        # Process each video
        all_results = {}
        total_alignments = 0
        
        for video_id in frames_data.keys():
            if video_id not in chunks_data:
                logger.warning(f"No chunks found for video {video_id}, skipping")
                continue
            
            logger.info(f"Processing video: {video_id}")
            
            try:
                # Align frames with chunks
                alignments = self.align_frames_with_chunks(
                    video_id, 
                    frames_data[video_id], 
                    chunks_data[video_id]
                )
                
                # Create visual context summary
                summary = self.create_visual_context_summary(video_id, alignments)
                
                # Save alignments
                alignments_file = self.output_dir / f"{video_id}_alignments.json"
                with open(alignments_file, 'w') as f:
                    json.dump(alignments, f, indent=2)
                
                # Save RAG-ready content (chunks with visual context)
                rag_output = self.create_rag_ready_output(video_id, alignments)
                rag_file = self.output_dir / f"{video_id}_rag_ready_aligned_content.json"
                with open(rag_file, 'w') as f:
                    json.dump(rag_output, f, indent=2)
                
                logger.info(f"RAG-ready content saved: {rag_file}")
                
                # Save summary matching formal presentations format
                summary = {
                    'video_id': video_id,
                    'total_alignments': len(alignments),
                    'total_frames_aligned': sum(len(a.get('relevant_frames', [])) for a in alignments),
                    'average_alignment_confidence': 0.9,  # Default for conversations
                    'output_files': {
                        'alignments': str(alignments_file),
                        'rag_ready': str(rag_file)
                    }
                }
                
                summary_file = self.output_dir / f"{video_id}_alignment_summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                logger.info(f"Alignment summary saved: {summary_file}")
                
                all_results[video_id] = {
                    'alignments_count': len(alignments),
                    'summary': summary,
                    'files': {
                        'alignments': str(alignments_file),
                        'summary': str(summary_file),
                        'rag_content': str(rag_file)
                    }
                }
                
                total_alignments += len(alignments)
                
                logger.info(f"✅ Completed {video_id}: {len(alignments)} alignments")
                
                # Update progress queue
                if self.progress_queue:
                    # Calculate visual coverage percentage
                    total_chunks = len(alignments)
                    chunks_with_frames = sum(1 for a in alignments if a.get('relevant_frames'))
                    visual_coverage = (chunks_with_frames / total_chunks * 100) if total_chunks > 0 else 0
                    
                    self.progress_queue.update_video_step_status(
                        video_id,
                        'step_06_frame_chunk_alignment',
                        'completed',
                        metadata={
                            'alignments_count': len(alignments),
                            'visual_coverage_percentage': visual_coverage,
                            'completed_at': datetime.now().isoformat()
                        }
                    )
                
            except Exception as e:
                logger.error(f"Error processing video {video_id}: {e}")
                all_results[video_id] = {
                    'error': str(e),
                    'success': False
                }
        
        # Create overall summary
        overall_summary = {
            'total_videos_processed': len(all_results),
            'total_alignments': total_alignments,
            'videos_with_errors': sum(1 for r in all_results.values() if r.get('error')),
            'successful_videos': sum(1 for r in all_results.values() if not r.get('error')),
            'results': all_results,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save overall summary
        overall_summary_file = self.output_dir / "overall_alignment_summary.json"
        with open(overall_summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        logger.info(f"📊 Frame-chunk alignment complete:")
        logger.info(f"   Videos processed: {len(all_results)}")
        logger.info(f"   Total alignments: {total_alignments}")
        logger.info(f"   Overall summary: {overall_summary_file}")
        
        return {
            'success': True,
            'message': f'Processed {len(all_results)} videos with {total_alignments} alignments',
            'overall_summary': overall_summary
        }
    
    def create_rag_ready_output(self, video_id: str, alignments: List[Dict]) -> Dict[str, Any]:
        """Create RAG-ready output format matching formal presentations pipeline structure."""
        rag_output = {
            'metadata': {
                'pipeline_step': 'step_06_frame_chunk_alignment',
                'total_alignments': len(alignments),
                'alignment_method': 'timestamp_based',
                'processing_timestamp': datetime.now().isoformat()
            },
            'aligned_content': []
        }
        
        for alignment in alignments:
            # Create RAG entry matching formal presentations structure
            rag_entry = {
                'content_id': alignment['chunk_id'],
                'text_content': {
                    'text': alignment.get('chunk_text', ''),
                    'metadata': {
                        'token_count': len(alignment.get('chunk_text', '').split()) if alignment.get('chunk_text') else 0,
                        'sentence_count': len([s for s in alignment.get('chunk_text', '').split('.') if s.strip()]) if alignment.get('chunk_text') else 0,
                        'primary_topics': alignment.get('chunk_topics', []),
                        'secondary_topics': [],  # Could be enhanced later
                        'key_terms': self._extract_key_terms(alignment.get('chunk_text', '')),
                        'content_summary': self._generate_content_summary(alignment.get('chunk_text', '')),
                        'scientific_domain': 'Developmental Biology, Bioelectricity',
                        'start_time_seconds': alignment.get('chunk_start_time', 0),
                        'end_time_seconds': alignment.get('chunk_end_time', 0)
                    }
                },
                'visual_content': {
                    'frames': [
                        {
                            'frame_id': frame.get('frame_id', frame.get('filename', '')),
                            'timestamp': frame.get('timestamp', 0),
                            'file_path': frame.get('file_path', ''),
                            'file_size': frame.get('file_size', 0),
                            'alignment_confidence': 0.9  # Default confidence for conversations
                        }
                        for frame in alignment.get('relevant_frames', [])
                    ],
                    'frame_count': len(alignment.get('relevant_frames', []))
                },
                'temporal_info': {
                    'frame_timestamps': [f.get('timestamp', 0) for f in alignment.get('relevant_frames', [])],
                    'time_range': {
                        'start': alignment.get('chunk_start_time', 0),
                        'end': alignment.get('chunk_end_time', 0)
                    }
                },
                'quality_metrics': {
                    'total_frames': len(alignment.get('relevant_frames', [])),
                    'average_confidence': 0.9,  # Default for conversations
                    'timestamp_coverage': 0.1  # Default coverage
                },
                'processing_metadata': {
                    'alignment_method': 'timestamp_based',
                    'timestamp_tolerance': 5.0
                },
                # Conversation-specific fields (preserved)
                'conversation_context': {
                    'question': alignment.get('chunk_question', ''),
                    'answer': alignment.get('chunk_answer', ''),
                    'questioner': alignment.get('chunk_questioner', ''),
                    'answerer': alignment.get('chunk_answerer', ''),
                    'youtube_link': alignment.get('chunk_youtube_link', ''),
                    'pipeline_type': 'conversations_1_on_1'
                },
                # Ensure video_id is available at chunk level
                'metadata': {
                    'video_id': video_id,
                    'pipeline_type': 'conversations_1_on_1',
                    'created_at': datetime.now().isoformat()
                }
            }
            
            rag_output['aligned_content'].append(rag_entry)
        
        return rag_output
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for metadata."""
        if not text:
            return []
        
        # Simple key term extraction - could be enhanced with NLP
        words = text.lower().split()
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        key_terms = []
        for word in words:
            word = word.strip('.,!?;:()[]{}"\'').lower()
            if len(word) > 3 and word not in stop_words and word.isalpha():
                key_terms.append(word)
        
        # Return top 10 unique terms
        return list(set(key_terms))[:10]
    
    def _generate_content_summary(self, text: str) -> str:
        """Generate a brief content summary for metadata."""
        if not text:
            return ""
        
        # Simple summary - first sentence or first 100 characters
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            summary = sentences[0]
            if len(summary) > 100:
                summary = summary[:97] + "..."
            return summary
        
        # Fallback to truncated text
        if len(text) > 100:
            return text[:97] + "..."
        return text

def main():
    """Main execution function."""
    try:
        # Initialize progress queue
        progress_queue = get_progress_queue()
        logger.info("✅ Progress queue initialized")
        
        # Initialize aligner
        aligner = FrameChunkAligner(progress_queue)
        
        # Process all videos
        result = aligner.process_all_videos()
        
        print("\n" + "=" * 60)
        print("STEP 6: FRAME-CHUNK ALIGNMENT COMPLETE")
        print("=" * 60)
        print(f"Message: {result['message']}")
        print("=" * 60)
        
        if result['success']:
            print("✅ Frame-chunk alignment completed successfully!")
            print(f"📁 Output saved to: {aligner.output_dir}")
            
            # Show summary
            summary = result['overall_summary']
            print(f"📊 Summary:")
            print(f"   Videos processed: {summary['total_videos_processed']}")
            print(f"   Total alignments: {summary['total_alignments']}")
            print(f"   Successful videos: {summary['successful_videos']}")
            print(f"   Videos with errors: {summary['videos_with_errors']}")
        else:
            print("❌ Frame-chunk alignment failed")
            
    except Exception as e:
        logger.error(f"Frame-chunk alignment failed: {e}")
        print(f"❌ Frame-chunk alignment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
