#!/usr/bin/env python3
"""
Step 04: Extract Semantic Chunks for Conversations
Processes speaker-identified transcripts to create semantic chunks focused on:
1. Michael Levin's knowledge and views (not attributing others' words to him)
2. Q&A patterns for fine-tuning
3. Topic organization for future retrieval
4. Conversation context preservation
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import argparse
from pipeline_progress_queue import get_progress_queue

# Import centralized logging configuration
from logging_config import setup_logging

# Configure logging
logger = setup_logging('extract_chunks')

class ConversationsChunkExtractor:
    def __init__(self, progress_queue=None):
        self.input_dir = Path("step_03_transcription")
        self.output_dir = Path("step_04_extract_chunks")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize progress queue
        self.progress_queue = progress_queue or get_progress_queue()
        
        # Speaker mapping (to be set by user)
        self.speaker_mapping = {}
        
        # Topic keywords for categorization
        self.topic_keywords = {
            'bioelectricity': ['bioelectric', 'electrical', 'voltage', 'current', 'ion', 'membrane', 'gap_junction'],
            'regeneration': ['regeneration', 'regenerate', 'regrow', 'healing', 'repair', 'wound'],
            'development': ['development', 'developmental', 'embryo', 'growth', 'morphogenesis', 'differentiation'],
            'morphology': ['morphology', 'shape', 'form', 'pattern', 'structure', 'anatomy'],
            'cognition': ['cognitive', 'cognition', 'intelligence', 'memory', 'learning', 'behavior'],
            'cancer': ['cancer', 'tumor', 'oncology', 'malignant', 'metastasis', 'carcinogen'],
            'aging': ['aging', 'senescence', 'longevity', 'age', 'degeneration'],
            'xenobots': ['xenobot', 'robot', 'artificial', 'synthetic', 'living_machine'],
            'collective_intelligence': ['collective', 'swarm', 'emergent', 'cooperation', 'coordination'],
            'information_theory': ['information', 'signal', 'communication', 'encoding', 'decoding'],
            'complexity': ['complex', 'complexity', 'emergent_properties', 'self_organization'],
            'evolution': ['evolution', 'evolutionary', 'adaptation', 'selection', 'fitness']
        }
    
    def process_all_videos(self):
        """Process all videos with completed transcripts"""
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist")
            return
        
        # Find all transcript files
        transcript_files = list(self.input_dir.glob("*_transcript.json"))
        if not transcript_files:
            logger.error(f"No transcript files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(transcript_files)} transcript files to process")
        
        # Track processing statistics
        total_videos = 0
        processed_videos = 0
        failed_videos = 0
        
        for transcript_file in transcript_files:
            try:
                video_id = transcript_file.stem.replace('_transcript', '')
                logger.info(f"🎭 Processing video: {video_id}")
                
                # Check if already processed
                if self.progress_queue:
                    video_status = self.progress_queue.get_video_status(video_id)
                    if video_status and video_status.get('step_04_extract_chunks') == 'completed':
                        logger.info(f"Chunks already completed for {video_id}, skipping")
                        continue
                
                # Process the video
                success = self.process_single_video(video_id, transcript_file)
                if success:
                    processed_videos += 1
                else:
                    failed_videos += 1
                
                total_videos += 1
                
            except Exception as e:
                logger.error(f"Failed to process video {transcript_file.name}: {e}")
                failed_videos += 1
                total_videos += 1
        
        # Log summary
        logger.info(f"Chunk Extraction Summary:")
        logger.info(f"  Total videos: {total_videos}")
        logger.info(f"  Successfully processed: {processed_videos}")
        logger.info(f"  Failed: {failed_videos}")
        if total_videos > 0:
            success_rate = (processed_videos / total_videos * 100)
            logger.info(f"  Success rate: {success_rate:.1f}%")
    
    def process_single_video(self, video_id: str, transcript_file: Path) -> bool:
        """Process a single video transcript"""
        try:
            # Load transcript data
            with open(transcript_file, 'r') as f:
                transcript_data = json.load(f)
            
            # Load speaker data
            speakers_file = self.input_dir / f"{video_id}_speakers.json"
            if not speakers_file.exists():
                logger.error(f"Speaker file not found for {video_id}")
                return False
            
            with open(speakers_file, 'r') as f:
                speakers_data = json.load(f)
            
            # Get speaker mapping from user
            if not self.get_speaker_mapping(video_id, speakers_data):
                logger.error(f"Failed to get speaker mapping for {video_id}")
                return False
            
            # Extract semantic chunks
            chunks = self.extract_semantic_chunks(transcript_data, speakers_data, video_id)
            
            # Extract Q&A pairs
            qa_pairs = self.extract_qa_pairs(transcript_data, speakers_data, video_id)
            
            # Organize by topics
            topic_organization = self.organize_by_topics(chunks, qa_pairs, video_id)
            
            # Save all outputs
            self.save_outputs(video_id, chunks, qa_pairs, topic_organization)
            
            # Update progress queue
            if self.progress_queue:
                self.progress_queue.update_video_step_status(
                    video_id,
                    'step_04_extract_chunks',
                    'completed',
                    metadata={
                        'chunks_file': str(self.output_dir / f"{video_id}_chunks.json"),
                        'qa_pairs_file': str(self.output_dir / f"{video_id}_qa_pairs.json"),
                        'topics_file': str(self.output_dir / f"{video_id}_topics.json"),
                        'completed_at': datetime.now().isoformat(),
                        'total_chunks': len(chunks),
                        'total_qa_pairs': len(qa_pairs),
                        'topics_identified': len(topic_organization['topics']),
                        'levin_speaker': self.speaker_mapping.get('levin', {}).get('speaker_id', 'Unknown'),
                        'guest_speaker': self.speaker_mapping.get('guest', {}).get('speaker_id', 'Unknown'),
                        'guest_name': self.speaker_mapping.get('guest', {}).get('name', 'Unknown'),
                        'guest_role': self.speaker_mapping.get('guest', {}).get('role', 'Unknown')
                    }
                )
                logger.info(f"📊 Progress queue updated: step 4 completed for {video_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            return False
    
    def get_speaker_mapping(self, video_id: str, speakers_data: Dict[str, Any]) -> bool:
        """Get speaker mapping from user input"""
        speakers = speakers_data.get('speakers', [])
        if not speakers:
            logger.error(f"No speakers found in {video_id}")
            return False
        
        logger.info(f"🎭 Speakers detected: {speakers}")
        logger.info(f"🎭 Please identify which speaker is Michael Levin:")
        
        # Display sample utterances for each speaker
        for speaker in speakers:
            sample_utterances = [
                turn['text'][:100] + "..." if len(turn['text']) > 100 else turn['text']
                for turn in speakers_data.get('speaker_turns', [])
                if turn['speaker'] == speaker
            ][:3]  # Show first 3 utterances
            
            logger.info(f"\n🎤 Speaker {speaker}:")
            for i, utterance in enumerate(sample_utterances, 1):
                logger.info(f"   {i}. {utterance}")
        
        # Get user input for Michael Levin
        while True:
            try:
                levin_speaker = input(f"\n🎭 Which speaker is Michael Levin? (Enter {speakers[0]} or {speakers[1]}): ").strip().upper()
                if levin_speaker in speakers:
                    break
                else:
                    logger.warning(f"Please enter one of: {speakers}")
            except KeyboardInterrupt:
                logger.info("Speaker mapping cancelled by user")
                return False
        
        # Get guest researcher identity
        guest_speaker = [s for s in speakers if s != levin_speaker][0]
        logger.info(f"\n👤 Now identifying the guest researcher (Speaker {guest_speaker}):")
        
        # Show more context for the guest speaker to help identification
        guest_utterances = [
            turn['text'] for turn in speakers_data.get('speaker_turns', [])
            if turn['speaker'] == guest_speaker
        ][:5]  # Show first 5 utterances for better context
        
        logger.info(f"🎤 Sample content from Speaker {guest_speaker}:")
        for i, utterance in enumerate(guest_utterances, 1):
            logger.info(f"   {i}. {utterance[:150]}{'...' if len(utterance) > 150 else ''}")
        
        # Get guest researcher name
        while True:
            try:
                guest_name = input(f"\n👤 Who is Speaker {guest_speaker}? (Enter full name, e.g., 'Stephen Wolfram'): ").strip()
                if guest_name and len(guest_name) > 1:
                    break
                else:
                    logger.warning("Please enter a valid name")
            except KeyboardInterrupt:
                logger.info("Guest identification cancelled by user")
                return False
        
        # Get guest researcher role/affiliation if available
        guest_role = input(f"👤 What is {guest_name}'s role/affiliation? (e.g., 'Computer Scientist, Wolfram Research' - press Enter to skip): ").strip()
        
        # Set the comprehensive mapping
        self.speaker_mapping = {
            'levin': {
                'speaker_id': levin_speaker,
                'name': 'Michael Levin',
                'role': 'Biologist, Tufts University',
                'type': 'primary_researcher'
            },
            'guest': {
                'speaker_id': guest_speaker,
                'name': guest_name,
                'role': guest_role if guest_role else 'Guest Researcher',
                'type': 'guest_researcher'
            }
        }
        
        logger.info(f"✅ Speaker mapping completed:")
        logger.info(f"   Michael Levin: Speaker {self.speaker_mapping['levin']['speaker_id']} - {self.speaker_mapping['levin']['role']}")
        logger.info(f"   Guest Researcher: Speaker {self.speaker_mapping['guest']['speaker_id']} - {self.speaker_mapping['guest']['name']} ({self.speaker_mapping['guest']['role']})")
        
        return True
    
    def extract_semantic_chunks(self, transcript_data: Dict[str, Any], speakers_data: Dict[str, Any], video_id: str) -> List[Dict[str, Any]]:
        """Extract semantic chunks focused on Michael Levin's knowledge and views"""
        logger.info(f"🧩 Extracting semantic chunks for {video_id}...")
        
        chunks = []
        chunk_id = 0
        
        # Get speaker turns
        speaker_turns = speakers_data.get('speaker_turns', [])
        if not speaker_turns:
            logger.warning(f"No speaker turns found for {video_id}")
            return chunks
        
        # Group consecutive turns by the same speaker
        current_chunk = None
        
        for turn in speaker_turns:
            speaker = turn['speaker']
            text = turn['text']
            start_time = turn['start'] / 1000.0  # Convert to seconds
            end_time = turn['end'] / 1000.0
            
            # Check if this is Michael Levin speaking
            is_levin = (speaker == self.speaker_mapping['levin']['speaker_id'])
            
            # Start new chunk if speaker changes or chunk is too long
            if (current_chunk is None or 
                current_chunk['speaker'] != speaker or 
                (end_time - current_chunk['start_time']) > 300):  # 5 minutes max
                
                # Save previous chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk
                chunk_id += 1
                current_chunk = {
                    'chunk_id': f"{video_id}_chunk_{chunk_id:03d}",
                    'video_id': video_id,
                    'speaker': speaker,
                    'is_levin': is_levin,
                    'text': text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_seconds': end_time - start_time,
                    'word_count': len(text.split()),
                    'chunk_type': self.determine_chunk_type(text, is_levin),
                    'topics': self.extract_topics(text),
                    'levin_knowledge_focus': is_levin,
                    'conversation_context': self.get_conversation_context(speaker_turns, turn),
                    'confidence': turn.get('confidence', 0)
                }
            else:
                # Extend current chunk
                current_chunk['text'] += ' ' + text
                current_chunk['end_time'] = end_time
                current_chunk['duration_seconds'] = end_time - current_chunk['start_time']
                current_chunk['word_count'] = len(current_chunk['text'].split())
                # Update topics with new content
                current_chunk['topics'].extend(self.extract_topics(text))
                # Remove duplicates and keep unique topics
                current_chunk['topics'] = list(set(current_chunk['topics']))
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"✅ Extracted {len(chunks)} semantic chunks")
        levin_chunks = [c for c in chunks if c['is_levin']]
        logger.info(f"🎭 Michael Levin chunks: {len(levin_chunks)}")
        logger.info(f"👤 Guest researcher chunks: {len(chunks) - len(levin_chunks)}")
        
        return chunks
    
    def determine_chunk_type(self, text: str, is_levin: bool) -> str:
        """Determine the type of chunk based on content and speaker"""
        text_lower = text.lower().strip()
        
        if is_levin:
            # Levin's responses and explanations
            if any(indicator in text_lower for indicator in ['i think', 'i believe', 'my view', 'my research']):
                return 'levin_opinion'
            elif any(indicator in text_lower for indicator in ['we have', 'we can', 'we showed', 'we demonstrated']):
                return 'levin_research'
            elif any(indicator in text_lower for indicator in ['the answer is', 'what happens is', 'the way it works']):
                return 'levin_explanation'
            else:
                return 'levin_general'
        else:
            # Guest's content
            if any(indicator in text_lower for indicator in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
                return 'guest_question'
            elif any(indicator in text_lower for indicator in ['i think', 'i believe', 'my view', 'my research']):
                return 'guest_opinion'
            else:
                return 'guest_general'
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using keyword matching"""
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def get_conversation_context(self, speaker_turns: List[Dict], current_turn: Dict) -> Dict[str, Any]:
        """Get conversation context around the current turn"""
        current_index = speaker_turns.index(current_turn)
        
        # Get previous and next turns for context
        prev_turn = speaker_turns[current_index - 1] if current_index > 0 else None
        next_turn = speaker_turns[current_index + 1] if current_index < len(speaker_turns) - 1 else None
        
        context = {
            'previous_speaker': prev_turn['speaker'] if prev_turn else None,
            'previous_text': prev_turn['text'][:200] + "..." if prev_turn and len(prev_turn['text']) > 200 else (prev_turn['text'] if prev_turn else None),
            'next_speaker': next_turn['speaker'] if next_turn else None,
            'next_text': next_turn['text'][:200] + "..." if next_turn and len(next_turn['text']) > 200 else (next_turn['text'] if next_turn else None)
        }
        
        return context
    
    def extract_qa_pairs(self, transcript_data: Dict[str, Any], speakers_data: Dict[str, Any], video_id: str) -> List[Dict[str, Any]]:
        """Extract Q&A pairs from the conversation"""
        logger.info(f"❓ Extracting Q&A pairs for {video_id}...")
        
        qa_pairs = []
        pair_id = 0
        
        speaker_turns = speakers_data.get('speaker_turns', [])
        
        for i, turn in enumerate(speaker_turns):
            speaker = turn['speaker']
            text = turn['text']
            
            # Check if this looks like a question
            if self.is_question(text):
                # Look for an answer in the next few turns
                answer = self.find_answer(speaker_turns, i)
                
                if answer:
                    pair_id += 1
                    qa_pair = {
                        'pair_id': f"{video_id}_qa_{pair_id:03d}",
                        'video_id': video_id,
                        'question': {
                            'speaker': speaker,
                            'text': text,
                            'start_time': turn['start'] / 1000.0,
                            'end_time': turn['end'] / 1000.0,
                            'is_levin': (speaker == self.speaker_mapping['levin']['speaker_id']),
                            'speaker_name': self.get_speaker_name(speaker),
                            'speaker_role': self.get_speaker_role(speaker)
                        },
                        'answer': {
                            'speaker': answer['speaker'],
                            'text': answer['text'],
                            'start_time': answer['start'] / 1000.0,
                            'end_time': answer['end'] / 1000.0,
                            'is_levin': (answer['speaker'] == self.speaker_mapping['levin']['speaker_id']),
                            'speaker_name': self.get_speaker_name(answer['speaker']),
                            'speaker_role': self.get_speaker_role(answer['speaker'])
                        },
                        'qa_type': self.classify_qa_pair(text, answer['text']),
                        'topics': self.extract_topics(text + ' ' + answer['text']),
                        'levin_involved': (speaker == self.speaker_mapping['levin']['speaker_id'] or 
                                         answer['speaker'] == self.speaker_mapping['levin']['speaker_id'])
                    }
                    qa_pairs.append(qa_pair)
        
        logger.info(f"✅ Extracted {len(qa_pairs)} Q&A pairs")
        levin_qa = [qa for qa in qa_pairs if qa['levin_involved']]
        logger.info(f"🎭 Q&A pairs involving Michael Levin: {len(levin_qa)}")
        
        return qa_pairs
    
    def get_speaker_name(self, speaker_id: str) -> str:
        """Get speaker name from speaker ID"""
        if speaker_id == self.speaker_mapping['levin']['speaker_id']:
            return self.speaker_mapping['levin']['name']
        elif speaker_id == self.speaker_mapping['guest']['speaker_id']:
            return self.speaker_mapping['guest']['name']
        else:
            return f"Speaker {speaker_id}"
    
    def get_speaker_role(self, speaker_id: str) -> str:
        """Get speaker role from speaker ID"""
        if speaker_id == self.speaker_mapping['levin']['speaker_id']:
            return self.speaker_mapping['levin']['role']
        elif speaker_id == self.speaker_mapping['guest']['speaker_id']:
            return self.speaker_mapping['guest']['role']
        else:
            return "Unknown"
    
    def is_question(self, text: str) -> bool:
        """Check if text looks like a question"""
        text_lower = text.lower().strip()
        
        # Direct question indicators
        if text_lower.endswith('?'):
            return True
        
        # Question words and phrases
        question_indicators = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'can you', 'could you', 'would you', 'do you', 'are you',
            'is there', 'are there', 'is this', 'is that', 'is it',
            'does', 'did', 'will', 'would', 'should', 'could',
            'tell me', 'explain', 'let me ask', 'quick question'
        ]
        
        return any(indicator in text_lower for indicator in question_indicators)
    
    def find_answer(self, speaker_turns: List[Dict], question_index: int) -> Optional[Dict[str, Any]]:
        """Find an answer to a question in subsequent turns"""
        # Look ahead up to 3 turns for an answer
        for i in range(question_index + 1, min(question_index + 4, len(speaker_turns))):
            turn = speaker_turns[i]
            text = turn['text']
            
            # Check if this looks like an answer
            if self.is_answer(text):
                return turn
        
        return None
    
    def is_answer(self, text: str) -> bool:
        """Check if text looks like an answer"""
        text_lower = text.lower().strip()
        
        # Answer indicators
        answer_indicators = [
            'well,', 'you know,', 'i think', 'actually,', 'basically,',
            'the answer is', 'what happens is', 'the way it works',
            'so,', 'right,', 'yeah,', 'um,', 'uh,'
        ]
        
        return any(indicator in text_lower for indicator in answer_indicators)
    
    def classify_qa_pair(self, question: str, answer: str) -> str:
        """Classify the type of Q&A pair"""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Check if Levin is involved
        levin_indicators = ['levin', 'michael', 'your research', 'your work', 'your view']
        
        if any(indicator in question_lower for indicator in levin_indicators):
            return 'about_levin'
        elif any(indicator in answer_lower for indicator in levin_indicators):
            return 'levin_explaining'
        else:
            return 'general_discussion'
    
    def organize_by_topics(self, chunks: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]], video_id: str) -> Dict[str, Any]:
        """Organize content by topics for future retrieval"""
        logger.info(f"🏷️ Organizing content by topics for {video_id}...")
        
        topic_organization = {
            'video_id': video_id,
            'pipeline_type': 'conversations_1_on_1',
            'organization_timestamp': datetime.now().isoformat(),
            'topics': {},
            'levin_knowledge_summary': {},
            'retrieval_metadata': {}
        }
        
        # Organize chunks by topic
        for chunk in chunks:
            for topic in chunk.get('topics', []):
                if topic not in topic_organization['topics']:
                    topic_organization['topics'][topic] = {
                        'chunks': [],
                        'qa_pairs': [],
                        'levin_content': [],
                        'guest_content': [],
                        'total_duration': 0,
                        'word_count': 0
                    }
                
                topic_data = topic_organization['topics'][topic]
                topic_data['chunks'].append(chunk['chunk_id'])
                
                if chunk['is_levin']:
                    topic_data['levin_content'].append(chunk['chunk_id'])
                else:
                    topic_data['guest_content'].append(chunk['chunk_id'])
                
                topic_data['total_duration'] += chunk['duration_seconds']
                topic_data['word_count'] += chunk['word_count']
        
        # Organize Q&A pairs by topic
        for qa_pair in qa_pairs:
            for topic in qa_pair.get('topics', []):
                if topic in topic_organization['topics']:
                    topic_organization['topics'][topic]['qa_pairs'].append(qa_pair['pair_id'])
        
        # Create Levin knowledge summary
        levin_chunks = [c for c in chunks if c['is_levin']]
        topic_organization['levin_knowledge_summary'] = {
            'total_levin_chunks': len(levin_chunks),
            'levin_topics': list(set([topic for chunk in levin_chunks for topic in chunk.get('topics', [])])),
            'levin_word_count': sum(chunk['word_count'] for chunk in levin_chunks),
            'levin_duration': sum(chunk['duration_seconds'] for chunk in levin_chunks)
        }
        
        # Add retrieval metadata
        topic_organization['retrieval_metadata'] = {
            'total_chunks': len(chunks),
            'total_qa_pairs': len(qa_pairs),
            'unique_topics': len(topic_organization['topics']),
            'speaker_mapping': self.speaker_mapping,
            'retrieval_notes': [
                "Use topics to find relevant conversations",
                "Levin chunks contain his knowledge and views",
                "Q&A pairs suitable for fine-tuning",
                "Guest content provides context but not Levin's views"
            ]
        }
        
        logger.info(f"✅ Organized content into {len(topic_organization['topics'])} topics")
        
        return topic_organization
    
    def save_outputs(self, video_id: str, chunks: List[Dict[str, Any]], qa_pairs: List[Dict[str, Any]], topic_organization: Dict[str, Any]):
        """Save all extracted outputs"""
        # Save chunks
        chunks_file = self.output_dir / f"{video_id}_chunks.json"
        with open(chunks_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        # Save Q&A pairs
        qa_pairs_file = self.output_dir / f"{video_id}_qa_pairs.json"
        with open(qa_pairs_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        
        # Save topic organization
        topics_file = self.output_dir / f"{video_id}_topics.json"
        with open(topics_file, 'w') as f:
            json.dump(topic_organization, f, indent=2)
        
        # Save summary
        summary = {
            'video_id': video_id,
            'pipeline_type': 'conversations_1_on_1',
            'step': 'step_04_extract_chunks',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_chunks': len(chunks),
                'levin_chunks': len([c for c in chunks if c['is_levin']]),
                'guest_chunks': len([c for c in chunks if not c['is_levin']]),
                'total_qa_pairs': len(qa_pairs),
                'levin_qa_pairs': len([qa for qa in qa_pairs if qa['levin_involved']]),
                'unique_topics': len(topic_organization['topics']),
                'speaker_mapping': self.speaker_mapping
            },
            'files_generated': {
                'chunks': str(chunks_file),
                'qa_pairs': str(qa_pairs_file),
                'topics': str(topics_file)
            }
        }
        
        summary_file = self.output_dir / f"{video_id}_chunking_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"💾 Saved outputs for {video_id}:")
        logger.info(f"   - Chunks: {chunks_file}")
        logger.info(f"   - Q&A Pairs: {qa_pairs_file}")
        logger.info(f"   - Topics: {topics_file}")
        logger.info(f"   - Summary: {summary_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Step 4: Extract semantic chunks from conversations")
    parser.add_argument("--video-id", help="Process specific video ID only")
    
    args = parser.parse_args()
    
    try:
        extractor = ConversationsChunkExtractor()
        
        if args.video_id:
            # Process specific video
            transcript_file = Path("step_03_transcription") / f"{args.video_id}_transcript.json"
            if transcript_file.exists():
                success = extractor.process_single_video(args.video_id, transcript_file)
                if success:
                    logger.info(f"✅ Successfully processed {args.video_id}")
                else:
                    logger.error(f"❌ Failed to process {args.video_id}")
            else:
                logger.error(f"Transcript file not found for {args.video_id}")
        else:
            # Process all videos
            extractor.process_all_videos()
        
    except Exception as e:
        logger.error(f"Failed to extract chunks: {e}")
        raise


if __name__ == "__main__":
    main()
