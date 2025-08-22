#!/usr/bin/env python3
"""
Step 06: Frame-Chunk Alignment for Conversations Pipeline
Aligns video frames with semantic chunks to provide visual context for the RAG system.
"""

import logging
import json
import time
import os
import openai
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
        
        # Initialize OpenAI client for LLM-powered content summaries
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("✅ OpenAI client initialized for LLM content summaries")
                logger.info(f"OpenAI API key found: {api_key[:10]}...")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI client initialization failed: {e}")
                self.openai_client = None
        else:
            logger.warning("No OPENAI_API_KEY found, using fallback summaries")
            self.openai_client = None
        
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
        
        # Look for main chunk files (our semantic content from step 4)
        main_chunk_files = list(self.chunks_dir.glob("*_chunks.json"))
        if main_chunk_files:
            logger.info(f"Found {len(main_chunk_files)} main chunk files")
            
            for chunk_file in main_chunk_files:
                try:
                    with open(chunk_file, 'r') as f:
                        chunks = json.load(f)
                    
                    # Extract video_id from filename
                    video_id = chunk_file.stem.replace('_chunks', '')
                    chunks_data[video_id] = chunks
                    logger.info(f"Loaded {len(chunks)} chunks for {video_id}")
                    
                except Exception as e:
                    logger.error(f"Error loading {chunk_file}: {e}")
                    continue
        
        # Fallback: Look for enhanced chunks if main chunks not found
        if not chunks_data:
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
        
        return chunks_data
    
    def align_frames_with_chunks(self, video_id: str, frames: List[Dict], chunks: List[Dict]) -> List[Dict]:
        """Align frames with chunks based on timestamps."""
        alignments = []
        
        logger.info(f"Aligning {len(frames)} frames with {len(chunks)} chunks for {video_id}")
        
        for chunk in chunks:
            try:
                # Get chunk timestamps (our chunks use start_time and end_time in milliseconds)
                start_time_ms = chunk.get('start_time')
                end_time_ms = chunk.get('end_time')
                
                if not start_time_ms or not end_time_ms:
                    logger.warning(f"Chunk {chunk.get('chunk_id', 'unknown')} missing timestamps, skipping")
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
                
                # Select only the FIRST frame (most efficient approach, matching formal presentations)
                selected_frame = relevant_frames[0] if relevant_frames else None
                
                # Create alignment entry
                alignment = {
                    'chunk_id': chunk.get('chunk_id', 'unknown'),
                    'chunk_text': chunk.get('semantic_text', '')[:200] + '...' if len(chunk.get('semantic_text', '')) > 200 else chunk.get('semantic_text', ''),
                    'chunk_start_time': start_time_s,
                    'chunk_end_time': end_time_s,
                    'chunk_start_timestamp': chunk.get('timestamp', ''),
                    'chunk_end_timestamp': chunk.get('timestamp', ''),
                    'chunk_speaker': chunk.get('speaker', ''),
                    'chunk_speaker_name': chunk.get('speaker_name', ''),
                    'chunk_is_levin': chunk.get('is_levin', False),
                    'frame': selected_frame,  # Single frame reference (not list)
                    'frame_count': 1 if selected_frame else 0,  # Always 1 or 0
                    'alignment_quality': self._calculate_alignment_quality(start_time_s, end_time_s, [selected_frame] if selected_frame else [])
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
        
        # With single-frame approach, quality is based on frame placement within chunk
        chunk_duration = end_time - start_time
        if chunk_duration <= 0:
            return "invalid_timing"
        
        # For single frame, quality depends on how well it represents the chunk
        # Frame should be reasonably centered within the chunk time range
        frame = frames[0]  # We now only have one frame
        frame_timestamp = frame.get('timestamp', 0)
        chunk_center = start_time + (chunk_duration / 2)
        
        # Calculate how close the frame is to the center of the chunk
        time_offset = abs(frame_timestamp - chunk_center)
        offset_percentage = (time_offset / chunk_duration) * 100
        
        if offset_percentage <= 25:  # Frame within 25% of chunk center
            return "excellent"
        elif offset_percentage <= 50:  # Frame within 50% of chunk center
            return "good"
        elif offset_percentage <= 75:  # Frame within 75% of chunk center
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
        
        # Find chunks with best visual coverage (now all chunks with frames have frame_count = 1)
        best_visual_chunks = [a for a in alignments if a['frame_count'] > 0][:5]
        
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
                    'topics': c.get('chunk_speaker', '')
                }
                for c in best_visual_chunks
            ],
            'processing_timestamp': datetime.now().isoformat(),
            'pipeline_type': 'conversations_1_on_2'  # Updated to reflect current pipeline
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
                    'total_frames_aligned': sum(1 for a in alignments if a.get('frame')),  # Count chunks with frames
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
                    chunks_with_frames = sum(1 for a in alignments if a.get('frame'))
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
                        'primary_topics': [alignment.get('chunk_speaker', '')],
                        'secondary_topics': [],  # Could be enhanced later
                        'key_terms': self._extract_key_terms(alignment.get('chunk_text', '')),
                        'content_summary': self._generate_content_summary(
                            alignment.get('chunk_text', ''),
                            speaker_info=f"Speaker: {alignment.get('chunk_speaker_name', 'Unknown')} ({alignment.get('chunk_speaker', '')})",
                            chunk_context=f"Time: {alignment.get('chunk_start_time', 0):.1f}s - {alignment.get('chunk_end_time', 0):.1f}s"
                        ),
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
                        for frame in ([alignment.get('frame')] if alignment.get('frame') else [])
                    ],
                    'frame_count': 1 if alignment.get('frame') else 0
                },
                'temporal_info': {
                    'frame_timestamps': [alignment.get('frame', {}).get('timestamp', 0)] if alignment.get('frame') else [],
                    'time_range': {
                        'start': alignment.get('chunk_start_time', 0),
                        'end': alignment.get('chunk_end_time', 0)
                    }
                },
                'quality_metrics': {
                    'total_frames': 1 if alignment.get('frame') else 0,
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
                    'pipeline_type': 'conversations_1_on_2'
                },
                # Ensure video_id is available at chunk level
                'metadata': {
                    'video_id': video_id,
                    'pipeline_type': 'conversations_1_on_2',
                    'created_at': datetime.now().isoformat()
                }
            }
            
            rag_output['aligned_content'].append(rag_entry)
        
        return rag_output
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key scientific and conceptual terms from text."""
        if not text:
            return []
        
        # Scientific and conceptual terms to prioritize
        priority_terms = {
            'bioelectricity', 'morphogenesis', 'regeneration', 'development', 'healing',
            'xenobots', 'anthrobots', 'synthetic', 'biology', 'intelligence',
            'cognitive', 'neurological', 'psychiatric', 'immune', 'system',
            'aging', 'dementia', 'functional', 'disorder', 'dissociation',
            'placebo', 'consciousness', 'integration', 'disintegration',
            'automaticity', 'attention', 'memory', 'brain', 'inflammation',
            'psychology', 'psychotherapy', 'neuropsychiatry', 'immunopsychiatry',
            'psychosis', 'autoimmunity', 'neurological', 'disorders',
            'functional', 'neurological', 'disorder', 'dissociation',
            'global', 'workspace', 'theory', 'consciousness', 'unity',
            'fragmentation', 'islands', 'integration', 'disintegration',
            'collective', 'intelligence', 'multicellular', 'competency',
            'goal', 'directed', 'behavior', 'emergent', 'properties'
        }
        
        # Extended stop words for conversational content
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'yeah', 'so', 'um', 'uh', 'well', 'okay', 'right', 'like', 'you', 'know', 'i', 'mean',
            'just', 'really', 'very', 'quite', 'actually', 'basically', 'essentially', 'obviously',
            'clearly', 'apparently', 'supposedly', 'allegedly', 'reportedly', 'evidently'
        }
        
        words = text.lower().split()
        key_terms = []
        
        for word in words:
            word = word.strip('.,!?;:()[]{}"\'').lower()
            
            # Skip very short words and stop words
            if len(word) < 3 or word in stop_words:
                continue
            
            # Prioritize scientific terms
            if word in priority_terms:
                key_terms.append(word)
            # Add other meaningful words
            elif word.isalpha() and len(word) > 4:
                key_terms.append(word)
        
        # Return priority terms first, then other terms (max 15 total)
        priority_found = [term for term in key_terms if term in priority_terms]
        other_terms = [term for term in key_terms if term not in priority_terms]
        
        result = priority_found + other_terms[:15 - len(priority_found)]
        return result[:15]
    
    def _generate_content_summary(self, text: str, speaker_info: str = "", chunk_context: str = "") -> str:
        """Generate intelligent content summary using LLM for conversational content."""
        if not text:
            return ""
        
        logger.info(f"Generating content summary for text ({len(text)} chars)")
        
        # Check if OpenAI client is available
        if not self.openai_client:
            logger.warning("OpenAI client not available, using fallback summary")
            return self._generate_fallback_summary(text)
        
        try:
            # Create context-aware prompt for LLM analysis
            prompt = f"""
            Analyze this transcript chunk from a scientific conversation and provide a concise summary.

            Speaker: {speaker_info}
            Time: {chunk_context}
            
            Transcript: {text[:1200]}...

            Write a clear, complete summary in 1-2 sentences that explains:
            - What the speaker was communicating
            - The scientific or clinical context
            - The key insight or point being made

            Keep it concise but complete. Avoid truncation.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a scientific content analyst. Provide complete, concise summaries without truncation. Focus on the substance and scientific content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Clean up the summary
            if summary.startswith("Summary:"):
                summary = summary[8:].strip()
            
            logger.info(f"Generated LLM summary ({len(summary)} chars): {summary[:100]}...")
            
            # Check if the response was truncated (ends with ...)
            if summary.endswith("..."):
                logger.warning(f"LLM response appears truncated: {summary}")
                # Try to get a more complete response with a shorter prompt
                return self._generate_retry_summary(text, speaker_info, chunk_context)
            
            return summary
            
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            # Fallback to basic extraction
            logger.info("Using fallback summary generation")
            return self._generate_fallback_summary(text)
    
    def _generate_retry_summary(self, text: str, speaker_info: str, chunk_context: str) -> str:
        """Generate a retry summary with a shorter, more focused prompt"""
        if not self.openai_client:
            return self._generate_fallback_summary(text)
        
        try:
            # Shorter, more focused prompt
            prompt = f"""
            Summarize this scientific conversation excerpt in 1-2 clear sentences:
            
            Speaker: {speaker_info}
            Text: {text[:800]}...
            
            Focus on the main scientific point or clinical insight.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Provide a complete, concise summary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            # If still truncated, use fallback
            if summary.endswith("..."):
                logger.warning("Retry summary also truncated, using fallback")
                return self._generate_fallback_summary(text)
            
            return summary
            
        except Exception as e:
            logger.warning(f"Retry summary generation failed: {e}")
            return self._generate_fallback_summary(text)
    
    def _generate_fallback_summary(self, text: str) -> str:
        """Fallback summary generation when LLM fails"""
        logger.info("Fallback summary method called")
        if not text:
            return ""
        
        # Clean the text
        text = text.strip()
        
        # Split into sentences and filter out very short or filler sentences
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 15]
        
        if not sentences:
            return text[:100] + "..." if len(text) > 100 else text
        
        # Look for the most informative sentence
        scientific_terms = {
            'bioelectricity', 'morphogenesis', 'regeneration', 'development', 'healing',
            'cognitive', 'neurological', 'psychiatric', 'immune', 'system',
            'aging', 'dementia', 'functional', 'disorder', 'dissociation'
        }
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if len(sentence) < 20:
                continue
                
            score = len(sentence)
            sentence_lower = sentence.lower()
            
            # Bonus for scientific terms
            for term in scientific_terms:
                if term in sentence_lower:
                    score += 20
            
            # Penalty for filler words at start
            if sentence_lower.startswith(('yeah', 'so', 'um', 'uh', 'well')):
                score -= 15
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if best_sentence:
            return best_sentence[:100] + "..." if len(best_sentence) > 100 else best_sentence
        
        return sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]

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
