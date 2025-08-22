#!/usr/bin/env python3
"""
Step 4: Multi-Speaker Transcript Processing for 1-on-2 Conversations
- Extract transcript and speakers from AssemblyAI
- Identify 3 speakers interactively (Levin + 2 others)
- Generate labeled transcript with multi-speaker context
- Extract and enhance Q&A pairs for 3-speaker dynamics
- Create rich semantic chunks for RAG with scientific taxonomy
- Multiple output formats for different use cases
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for validation
class EnhancedQAResponse(BaseModel):
    """Pydantic model for validating enhanced Q&A responses."""
    enhanced_text: str = Field(
        ..., 
        min_length=50,
        description="Enhanced text combining context and Michael Levin's complete answer"
    )
    topics: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=4,
        description="List of 1-4 specific scientific topics that describe the content"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "enhanced_text": "When asked about defining intelligence, Michael Levin explains: Okay. What I use is a definition...",
                "topics": ["intelligence_definition", "goal_directed_behavior"]
            }
        }

class LevinChunkResponse(BaseModel):
    """Pydantic model for validating Levin semantic chunk responses."""
    text: str = Field(..., description="The exact text spoken by Michael Levin")
    topic: str = Field(..., description="Main scientific topic")
    insight: str = Field(..., description="Key insight or knowledge conveyed")
    context: str = Field(..., description="Conversation context around this chunk")
    scientific_topics: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=4,
        description="Specific scientific topics from the taxonomy"
    )

class MultiSpeakerTranscriptProcessor:
    """Advanced transcript processor for 1-on-2 conversations with 3-speaker dynamics"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.input_dir = Path("step_03_transcription")
        self.output_dir = Path("step_04_extract_chunks")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different outputs
        (self.output_dir / "finetune_data").mkdir(exist_ok=True)
        (self.output_dir / "enhanced_chunks").mkdir(exist_ok=True)
        
        # Speaker mappings for 1-on-2 conversations
        self.speaker_mappings = {}
        self.speaker_mappings_file = self.output_dir / "speaker_mappings.json"
        self.load_speaker_mappings()
        
        # Scientific topic taxonomy for Michael Levin's work
        self.scientific_topics = [
            "intelligence_definition", "scale_free_cognition", "unconventional_intelligence", "seti_suti",
            "bioelectricity", "morphogenesis", "regeneration", "development", "healing",
            "xenobots", "anthrobots", "synthetic_biology", "living_machines",
            "goal_directed_behavior", "emergent_properties", "collective_intelligence",
            "plasticity", "adaptation", "problem_solving", "cybernetics",
            "computational_biology", "information_theory", "complexity",
            "morphospace_navigation", "pattern_formation", "self_organization",
            "evolutionary_competency", "morphogenetic_code", "bioelectric_circuits",
            "anatomical_decision_making", "regenerative_medicine", "developmental_biology"
        ]
    
    def load_speaker_mappings(self):
        """Load existing speaker mappings"""
        if self.speaker_mappings_file.exists():
            try:
                with open(self.speaker_mappings_file, 'r') as f:
                    self.speaker_mappings = json.load(f)
                logger.info(f"Loaded existing speaker mappings for {len(self.speaker_mappings)} videos")
            except Exception as e:
                logger.error(f"Error loading speaker mappings: {e}")
                self.speaker_mappings = {}
    
    def save_speaker_mappings(self):
        """Save speaker mappings to file"""
        try:
            with open(self.speaker_mappings_file, 'w') as f:
                json.dump(self.speaker_mappings, f, indent=2)
            logger.info("Speaker mappings saved successfully")
        except Exception as e:
            logger.error(f"Error saving speaker mappings: {e}")
    
    def identify_speakers_interactive(self, video_id: str, speakers: List[str]) -> Dict[str, Dict]:
        """Interactive speaker identification for conversations with 2+ speakers"""
        if len(speakers) < 2:
            raise ValueError(f"Need at least 2 speakers, found {len(speakers)}")
        
        print(f"\n🎭 SPEAKER IDENTIFICATION FOR 1-ON-2 CONVERSATION")
        print(f"Video: {video_id}")
        print(f"Available speakers: {', '.join(speakers)}")
        print("=" * 60)
        
        # Step 1: Identify Michael Levin
        print("\n🎯 STEP 1: IDENTIFY MICHAEL LEVIN")
        print("=" * 40)
        levin_speaker = None
        while levin_speaker not in speakers:
            levin_speaker = input(f"Which speaker is Michael Levin? ({', '.join(speakers)}): ").upper()
            if levin_speaker not in speakers:
                print(f"❌ Invalid speaker. Please choose from: {', '.join(speakers)}")
        
        print(f"✅ Michael Levin identified as Speaker {levin_speaker}")
        
        # Step 2: Identify other speakers
        other_speakers = [s for s in speakers if s != levin_speaker]
        speaker_info = {}
        
        for i, speaker_id in enumerate(other_speakers, 1):
            print(f"\n🎭 STEP {i+1}: IDENTIFY SPEAKER {i+1}")
            print("=" * 40)
            print(f"Speaker ID: {speaker_id}")
            
            name = input(f"Enter the name for Speaker {speaker_id}: ").strip()
            role = input(f"Enter the role/organization for {name}: ").strip()
            context = input(f"Any additional context about {name}? (optional): ").strip()
            
            speaker_info[speaker_id] = {
                "speaker_id": speaker_id,
                "name": name,
                "role": role,
                "is_levin": False,
                "speaker_order": i + 1,
                "additional_context": context if context else None,
                "identified_at": datetime.now().isoformat()
            }
            
            print(f"✅ Speaker {speaker_id} identified as: {name} ({role})")
        
        # Add Levin's information
        speaker_info[levin_speaker] = {
            "speaker_id": levin_speaker,
            "name": "Michael Levin",
            "role": "Biologist, Tufts University",
            "is_levin": True,
            "speaker_order": min(2, len(speakers)),  # Dynamic order based on speaker count
            "additional_context": None,
            "identified_at": datetime.now().isoformat()
        }
        
        print(f"\n✅ All speakers identified successfully!")
        print("=" * 60)
        
        return speaker_info
    
    def process_transcript_file(self, transcript_file: Path) -> Dict[str, Any]:
        """Process a single transcript file with 3-speaker support"""
        video_id = transcript_file.stem.replace('_transcript', '')
        
        try:
            with open(transcript_file, 'r') as f:
                transcript_data = json.load(f)
            
            # Extract speaker information
            speakers = list(set(item.get('speaker', '') for item in transcript_data.get('utterances', [])))
            speakers = [s for s in speakers if s]  # Remove empty speakers
            
            if len(speakers) < 2:
                logger.warning(f"Video {video_id} has {len(speakers)} speakers, need at least 2")
                return {"error": f"Need at least 2 speakers, found {len(speakers)}"}
            
            # Log speaker count for transparency
            logger.info(f"Video {video_id} has {len(speakers)} speakers: {speakers}")
            
            # Note: Currently working with {len(speakers)} speakers due to transcription configuration
            # TODO: Re-run transcription step with 3-speaker configuration for full 1-on-2 support
            
            # Check if speakers are already identified
            if video_id not in self.speaker_mappings:
                logger.info(f"Identifying speakers for video {video_id}")
                speaker_info = self.identify_speakers_interactive(video_id, speakers)
                self.speaker_mappings[video_id] = speaker_info
                self.save_speaker_mappings()
            else:
                speaker_info = self.speaker_mappings[video_id]
                logger.info(f"Using existing speaker mappings for video {video_id}")
            
            # Process transcript with speaker context
            processed_chunks = self.create_semantic_chunks(transcript_data, speaker_info, video_id)
            
            # Enhance chunks with LLM analysis for better topic extraction
            logger.info(f"Enhancing {len(processed_chunks)} chunks with LLM analysis...")
            enhanced_chunks = self.enhance_chunks_with_llm(processed_chunks, video_id)
            
            # Generate Q&A pairs
            qa_pairs = self.generate_qa_pairs(transcript_data, speaker_info, video_id)
            
            # Create multi-speaker analysis
            multi_speaker_analysis = self.analyze_multi_speaker_dynamics(transcript_data, speaker_info, video_id)
            
            return {
                "video_id": video_id,
                "speakers": speaker_info,
                "chunks": enhanced_chunks,  # Use enhanced chunks with LLM topics
                "qa_pairs": qa_pairs,
                "multi_speaker_analysis": multi_speaker_analysis
            }
            
        except Exception as e:
            logger.error(f"Error processing transcript {transcript_file}: {e}")
            return {"error": str(e)}
    
    def create_semantic_chunks(self, transcript_data: Dict, speaker_info: Dict, video_id: str) -> List[Dict]:
        """Create semantic chunks with 3-speaker context"""
        chunks = []
        utterances = transcript_data.get('utterances', [])
        
        # Group utterances by speaker and create chunks
        current_chunk = []
        chunk_id = 0
        
        for utterance in utterances:
            speaker_id = utterance.get('speaker', '')
            text = utterance.get('text', '')
            start = utterance.get('start', 0)
            end = utterance.get('end', 0)
            
            if not speaker_id or not text:
                continue
            
            # Get speaker context
            speaker_context = speaker_info.get(speaker_id, {})
            
            # Create chunk entry
            chunk_entry = {
                "speaker_id": speaker_id,
                "speaker_name": speaker_context.get('name', 'Unknown'),
                "speaker_role": speaker_context.get('role', 'Unknown'),
                "is_levin": speaker_context.get('is_levin', False),
                "speaker_order": speaker_context.get('speaker_order', 0),
                "text": text,
                "start_time": start,
                "end_time": end,
                "timestamp": f"{start:.2f}s - {end:.2f}s"
            }
            
            current_chunk.append(chunk_entry)
            
            # Create chunk when we have enough content or speaker changes
            if len(current_chunk) >= 3 or (len(current_chunk) > 0 and len(' '.join([c['text'] for c in current_chunk])) > 500):
                chunk_data = self.create_chunk_data(current_chunk, chunk_id, video_id, speaker_info)
                chunks.append(chunk_data)
                current_chunk = []
                chunk_id += 1
        
        # Add remaining content as final chunk
        if current_chunk:
            chunk_data = self.create_chunk_data(current_chunk, chunk_id, video_id, speaker_info)
            chunks.append(chunk_data)
        
        return chunks
    
    def create_chunk_data(self, chunk_entries: List[Dict], chunk_id: int, video_id: str, speaker_info: Dict) -> Dict:
        """Create structured chunk data with multi-speaker context"""
        # Combine all text
        combined_text = ' '.join([entry['text'] for entry in chunk_entries])
        
        # Get speaker statistics
        speakers_in_chunk = list(set(entry['speaker_id'] for entry in chunk_entries))
        levin_involved = any(entry['is_levin'] for entry in chunk_entries)
        
        # Determine conversation type
        if len(speakers_in_chunk) == 1:
            conversation_type = "monologue"
        elif len(speakers_in_chunk) == 2:
            conversation_type = "dialogue"
        else:
            conversation_type = "multi_speaker_discussion"
        
        # Create chunk data
        chunk_data = {
            "chunk_id": f"{video_id}_chunk_{chunk_id:03d}",
            "semantic_text": combined_text,
            "speaker": speakers_in_chunk[0] if len(speakers_in_chunk) == 1 else "multiple",
            "speaker_name": "Multiple Speakers" if len(speakers_in_chunk) > 1 else speaker_info[speakers_in_chunk[0]]['name'],
            "speaker_role": "Multiple Roles" if len(speakers_in_chunk) > 1 else speaker_info[speakers_in_chunk[0]]['role'],
            "is_levin": levin_involved,
            "speaker_order": "multiple" if len(speakers_in_chunk) > 1 else speaker_info[speakers_in_chunk[0]]['speaker_order'],
            "speaker_additional_context": None,
            "speaker_identified_at": datetime.now().isoformat(),
            "start_time": min(entry['start_time'] for entry in chunk_entries),
            "end_time": max(entry['end_time'] for entry in chunk_entries),
            "timestamp": f"{min(entry['start_time'] for entry in chunk_entries):.2f}s - {max(entry['end_time'] for entry in chunk_entries):.2f}s",
            # Topics will be populated by LLM enhancement later
            "primary_topics": [],
            "secondary_topics": [],
            "key_terms": [],
            "conversation_context": {
                "speaker_context": {
                    "speakers_involved": speakers_in_chunk,
                    "speaker_details": {s: speaker_info[s] for s in speakers_in_chunk},
                    "conversation_type": conversation_type
                },
                "levin_context": {
                    "levin_involved": levin_involved,
                    "levin_role": "primary_speaker" if levin_involved else "not_involved",
                    "levin_contribution_type": "providing_expert_insights" if levin_involved else None,
                    "levin_expertise_area": None,  # Will be filled by AI analysis
                    "levin_insights": []
                },
                "multi_speaker_context": {
                    "total_speakers": len(speakers_in_chunk),
                    "speaker_interaction_type": conversation_type,
                    "conversation_flow": [f"{entry['speaker_id']}: {entry['text'][:50]}..." for entry in chunk_entries[:3]],
                    "collaboration_pattern": self.determine_collaboration_pattern(chunk_entries, speaker_info)
                }
            },
            "chunk_entries": chunk_entries
        }
        
        return chunk_data
    
    def determine_collaboration_pattern(self, chunk_entries: List[Dict], speaker_info: Dict) -> str:
        """Determine the collaboration pattern in a chunk"""
        if len(chunk_entries) < 2:
            return "single_speaker"
        
        # Analyze speaker sequence
        speaker_sequence = [entry['speaker_id'] for entry in chunk_entries]
        
        # Check for patterns
        if len(set(speaker_sequence)) == 1:
            return "monologue"
        elif len(set(speaker_sequence)) == 2:
            return "dialogue"
        elif len(set(speaker_sequence)) == 3:
            return "round_table_discussion"
        else:
            return "complex_interaction"
    
    def generate_qa_pairs(self, transcript_data: Dict, speaker_info: Dict, video_id: str) -> List[Dict]:
        """Generate Q&A pairs with 3-speaker support"""
        qa_pairs = []
        utterances = transcript_data.get('utterances', [])
        
        # Find question-answer patterns
        for i in range(len(utterances) - 1):
            current_utterance = utterances[i]
            next_utterance = utterances[i + 1]
            
            # Check if current utterance is a question
            if self.is_question(current_utterance['text']):
                # Find the answer (next utterance from different speaker)
                if current_utterance['speaker'] != next_utterance['speaker']:
                    qa_pair = self.create_qa_pair(
                        current_utterance, next_utterance, speaker_info, video_id, len(qa_pairs)
                    )
                    qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def is_question(self, text: str) -> bool:
        """Check if text contains a question"""
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)
    
    def create_qa_pair(self, question_utterance: Dict, answer_utterance: Dict, speaker_info: Dict, video_id: str, pair_id: int) -> Dict:
        """Create a Q&A pair with multi-speaker context"""
        question_speaker = question_utterance['speaker']
        answer_speaker = answer_utterance['speaker']
        
        # Get speaker contexts
        question_speaker_info = speaker_info.get(question_speaker, {})
        answer_speaker_info = speaker_info.get(answer_speaker, {})
        
        # Check if Levin is involved
        levin_involved = question_speaker_info.get('is_levin', False) or answer_speaker_info.get('is_levin', False)
        
        qa_pair = {
            "pair_id": f"{video_id}_qa_{pair_id:03d}",
            "question": {
                "speaker": question_speaker,
                "text": question_utterance['text'],
                "is_levin": question_speaker_info.get('is_levin', False),
                "speaker_name": question_speaker_info.get('name', 'Unknown'),
                "speaker_role": question_speaker_info.get('role', 'Unknown'),
                "speaker_order": question_speaker_info.get('speaker_order', 0),
                "timestamp": f"{question_utterance['start']:.2f}s - {question_utterance['end']:.2f}s"
            },
            "answer": {
                "speaker": answer_speaker,
                "text": answer_utterance['text'],
                "is_levin": answer_speaker_info.get('is_levin', False),
                "speaker_name": answer_speaker_info.get('name', 'Unknown'),
                "speaker_role": answer_speaker_info.get('role', 'Unknown'),
                "speaker_order": answer_speaker_info.get('speaker_order', 0),
                "timestamp": f"{answer_utterance['start']:.2f}s - {answer_utterance['end']:.2f}s"
            },
            "levin_involved": levin_involved,
            "multi_speaker_dynamics": {
                "interaction_pattern": "question_response",
                "collaboration_type": "academic_discussion",
                "speaker_roles": ["questioner", "responder"]
            }
        }
        
        return qa_pair
    
    def analyze_multi_speaker_dynamics(self, transcript_data: Dict, speaker_info: Dict, video_id: str) -> Dict:
        """Analyze multi-speaker dynamics in the conversation"""
        utterances = transcript_data.get('utterances', [])
        
        # Count speaker contributions
        speaker_contributions = {}
        for utterance in utterances:
            speaker_id = utterance.get('speaker', '')
            if speaker_id:
                if speaker_id not in speaker_contributions:
                    speaker_contributions[speaker_id] = {
                        "utterance_count": 0,
                        "total_words": 0,
                        "is_levin": speaker_info.get(speaker_id, {}).get('is_levin', False)
                    }
                
                speaker_contributions[speaker_id]["utterance_count"] += 1
                speaker_contributions[speaker_id]["total_words"] += len(utterance.get('text', '').split())
        
        # Analyze conversation flow
        conversation_flow = []
        for i, utterance in enumerate(utterances[:10]):  # First 10 utterances for flow analysis
            speaker_id = utterance.get('speaker', '')
            speaker_name = speaker_info.get(speaker_id, {}).get('name', 'Unknown')
            conversation_flow.append(f"{speaker_name}: {utterance.get('text', '')[:50]}...")
        
        analysis = {
            "video_id": video_id,
            "total_utterances": len(utterances),
            "speaker_contributions": speaker_contributions,
            "conversation_flow_sample": conversation_flow,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return analysis
    
    def create_enhanced_rag_chunks(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Create enhanced RAG chunks from Q&A pairs with timestamp information"""
        logger.info("Creating enhanced RAG chunks...")
        
        enhanced_chunks = []
        
        for qa in qa_pairs:
            # Create enhanced chunk with Q&A information embedded
            # Extract text from nested question/answer structures
            question_obj = qa.get('question', {})
            answer_obj = qa.get('answer', {})
            
            # Get the actual text content
            question_text = question_obj.get('text', '') if isinstance(question_obj, dict) else str(question_obj)
            answer_text = answer_obj.get('text', '') if isinstance(answer_obj, dict) else str(answer_obj)
            
            # Get speaker information
            questioner = question_obj.get('speaker_name', '') if isinstance(question_obj, dict) else ''
            answerer = answer_obj.get('speaker_name', '') if isinstance(answer_obj, dict) else ''
            
            # Determine if Levin is involved
            is_levin = (isinstance(answer_obj, dict) and answer_obj.get('is_levin', False)) or \
                       (isinstance(question_obj, dict) and question_obj.get('is_levin', False))
            
            # Get timestamps
            start_time = question_obj.get('timestamp', '') if isinstance(question_obj, dict) else ''
            end_time = answer_obj.get('timestamp', '') if isinstance(answer_obj, dict) else ''
            
            enhanced_chunk = {
                'chunk_id': qa.get('pair_id', f"{qa.get('video_id', 'unknown')}_qa_{len(enhanced_chunks)}"),
                'semantic_text': f"{question_text} {answer_text}".strip(),
                'speaker': questioner,
                'speaker_name': questioner,
                'is_levin': is_levin,
                'start_time': start_time,
                'end_time': end_time,
                'timestamp': start_time,
                'conversation_context': {
                    'question': question_text,
                    'answer': answer_text,
                    'questioner': questioner,
                    'answerer': answerer,
                    'youtube_link': qa.get('youtube_link', ''),
                    'qa_type': qa.get('qa_type', 'conversational'),
                    'extraction_method': qa.get('extraction_method', 'pattern_recognition')
                },
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'pipeline_type': 'conversations_1_on_2',
                    'enhancement_level': 'qa_enhanced'
                }
            }
            
            enhanced_chunks.append(enhanced_chunk)
        
        logger.info(f"Created {len(enhanced_chunks)} enhanced RAG chunks")
        return enhanced_chunks
    
    def create_fine_tuning_datasets(self, qa_pairs: List[Dict], video_id: str):
        """Create fine-tuning datasets in multiple formats"""
        logger.info(f"Creating fine-tuning datasets for {video_id}...")
        
        # Create OpenAI format dataset
        openai_data = []
        for qa in qa_pairs:
            openai_entry = {
                'messages': [
                    {'role': 'user', 'content': qa.get('question', '')},
                    {'role': 'assistant', 'content': qa.get('answer', '')}
                ]
            }
            openai_data.append(openai_entry)
        
        openai_file = self.output_dir / "finetune_data" / f"{video_id}_openai_dataset.json"
        with open(openai_file, 'w') as f:
            json.dump(openai_data, f, indent=2)
        
        # Create Hugging Face format dataset
        hf_data = []
        for qa in qa_pairs:
            hf_entry = {
                'question': qa.get('question', ''),
                'answer': qa.get('answer', ''),
                'questioner': qa.get('questioner', ''),
                'answerer': qa.get('answerer', ''),
                'qa_type': qa.get('qa_type', 'conversational'),
                'start_time': qa.get('start_time', 0),
                'end_time': qa.get('end_time', 0)
            }
            hf_data.append(hf_entry)
        
        hf_file = self.output_dir / "finetune_data" / f"{video_id}_huggingface_dataset.json"
        with open(hf_file, 'w') as f:
            json.dump(hf_data, f, indent=2)
        
        # Create dataset statistics
        stats = {
            'video_id': video_id,
            'total_qa_pairs': len(qa_pairs),
            'question_types': {
                'explicit': len([qa for qa in qa_pairs if qa.get('qa_type') == 'explicit']),
                'synthetic': len([qa for qa in qa_pairs if qa.get('qa_type') == 'synthetic']),
                'conversational': len([qa for qa in qa_pairs if qa.get('qa_type') == 'conversational'])
            },
            'avg_question_length': sum(len(str(qa.get('question', '')).split()) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'avg_answer_length': sum(len(str(qa.get('answer', '')).split()) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'created_at': datetime.now().isoformat()
        }
        
        stats_file = self.output_dir / "finetune_data" / f"{video_id}_dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Created fine-tuning datasets:")
        logger.info(f"   OpenAI format: {openai_file}")
        logger.info(f"   Hugging Face format: {hf_file}")
        logger.info(f"   Statistics: {stats_file}")
        
        return stats
    
    def extract_primary_topics(self, text: str) -> List[str]:
        """Extract primary scientific topics from text"""
        if not text:
            return []
        
        # Define scientific topic keywords
        scientific_topics = {
            'bioelectricity': ['bioelectric', 'bioelectricity', 'electrical', 'voltage', 'current', 'membrane', 'ion', 'bioelectrical'],
            'morphogenesis': ['morphogenesis', 'development', 'embryo', 'regeneration', 'growth', 'pattern', 'developmental'],
            'neuroscience': ['brain', 'neural', 'neuron', 'synapse', 'cognitive', 'memory', 'learning', 'neurological', 'neuropsychiatric'],
            'psychiatry': ['psychiatric', 'mental', 'disorder', 'therapy', 'treatment', 'diagnosis', 'psychology', 'psychologist'],
            'aging': ['aging', 'elderly', 'cognitive decline', 'dementia', 'memory loss', 'aging process'],
            'dissociation': ['dissociative', 'identity', 'trauma', 'memory', 'consciousness', 'dissociation'],
            'regeneration': ['regeneration', 'healing', 'repair', 'recovery', 'regrowth'],
            'evolution': ['evolution', 'adaptation', 'selection', 'mutation', 'fitness'],
            'immunology': ['immune', 'immunology', 'autoimmunity', 'inflammation', 'immune system'],
            'functional_disorders': ['functional', 'neurological disorder', 'fnd', 'functional neurological']
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in scientific_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        # Return top 3-5 topics
        return found_topics[:5] if found_topics else ['general_science']
    
    def extract_secondary_topics(self, text: str) -> List[str]:
        """Extract secondary scientific topics from text"""
        if not text:
            return []
        
        # Define secondary topic keywords
        secondary_topics = {
            'research_methods': ['study', 'experiment', 'analysis', 'data', 'results', 'methodology'],
            'clinical_practice': ['patient', 'clinical', 'treatment', 'therapy', 'diagnosis', 'assessment'],
            'theoretical_framework': ['theory', 'model', 'framework', 'concept', 'hypothesis'],
            'collaboration': ['collaboration', 'partnership', 'team', 'cooperation', 'joint'],
            'future_directions': ['future', 'next', 'potential', 'opportunity', 'development']
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in secondary_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        # Return top 3-5 secondary topics
        return found_topics[:5] if found_topics else []
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key scientific terms from text"""
        if not text:
            return []
        
        # Define stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'yeah', 'um', 'uh', 'so', 'well', 'like', 'you know', 'i mean', 'sort of', 'kind of',
            'i\'m', 'kind', 'sort', 'guess', 'who', 'see', 'lot', 'also', 'you\'d', 'call'
        }
        
        # Split text into words and filter
        words = text.lower().split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top terms
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        key_terms = [word for word, count in sorted_words[:15]]
        
        return key_terms
    
    def enhance_chunks_with_llm(self, chunks: List[Dict], video_id: str) -> List[Dict]:
        """Enhance chunks using LLM analysis for better topic extraction"""
        enhanced_chunks = []
        total_chunks = len(chunks)
        
        logger.info(f"Starting LLM enhancement of {total_chunks} chunks...")
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Enhancing chunk {i+1}/{total_chunks}: {chunk['chunk_id']}")
                enhanced_chunk = self.enhance_single_chunk(chunk, video_id)
                enhanced_chunks.append(enhanced_chunk)
                
                # Add delay between requests to respect rate limits
                if i < total_chunks - 1:  # Don't delay after the last chunk
                    import time
                    delay_seconds = 1.0  # 1 second delay between requests
                    logger.info(f"Waiting {delay_seconds}s before next request...")
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                logger.warning(f"Failed to enhance chunk {chunk['chunk_id']}: {e}")
                # Use basic chunk if enhancement fails
                enhanced_chunks.append(chunk)
        
        logger.info(f"LLM enhancement completed: {len(enhanced_chunks)} chunks processed")
        return enhanced_chunks
    
    def enhance_single_chunk(self, chunk: Dict[str, Any], video_id: str) -> Dict[str, Any]:
        """Enhance a single chunk with LLM analysis"""
        try:
            # Create prompt for LLM analysis
            prompt = f"""
            Analyze this transcript chunk from a scientific conversation and provide:

            1. **Primary Topics** (3-5 main scientific concepts)
            2. **Secondary Topics** (5-8 related concepts)
            3. **Key Terms** (important scientific terminology)
            4. **Content Summary** (2-3 sentence summary)
            5. **Scientific Domain** (e.g., developmental biology, bioelectricity, psychiatry, neuroscience, etc.)

            Transcript chunk:
            {chunk['semantic_text'][:2000]}...

            Respond in JSON format:
            {{
                "primary_topics": ["topic1", "topic2"],
                "secondary_topics": ["topic1", "topic2"],
                "key_terms": ["term1", "term2"],
                "content_summary": "Brief summary",
                "scientific_domain": "Domain name"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a scientific content analyst specializing in developmental biology, bioelectricity, psychiatry, neuroscience, and related fields. You excel at extracting meaningful insights from conversational scientific discussions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse LLM response
            llm_content = response.choices[0].message.content
            enhanced_data = self.parse_llm_response(llm_content)
            
            # Merge with original chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk.update(enhanced_data)
            enhanced_chunk['enhanced'] = True
            
            return enhanced_chunk
            
        except Exception as e:
            logger.warning(f"LLM enhancement failed for chunk {chunk['chunk_id']}: {e}")
            # Return basic chunk if enhancement fails
            return chunk
    
    def parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return {
                    'primary_topics': data.get('primary_topics', []),
                    'secondary_topics': data.get('secondary_topics', []),
                    'key_terms': data.get('key_terms', []),
                    'content_summary': data.get('content_summary', ''),
                    'scientific_domain': data.get('scientific_domain', '')
                }
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        # Fallback to basic extraction
        return {
            'primary_topics': [],
            'secondary_topics': [],
            'key_terms': [],
            'content_summary': '',
            'scientific_domain': ''
        }
    
    def save_outputs(self, video_id: str, processed_data: Dict):
        """Save all outputs for a video"""
        try:
            # Save enhanced chunks
            chunks_file = self.output_dir / f"{video_id}_chunks.json"
            with open(chunks_file, 'w') as f:
                json.dump(processed_data['chunks'], f, indent=2)
            
            # Save Q&A pairs
            qa_file = self.output_dir / f"{video_id}_qa_pairs.json"
            with open(qa_file, 'w') as f:
                json.dump(processed_data['qa_pairs'], f, indent=2)
            
            # Save multi-speaker analysis
            analysis_file = self.output_dir / f"{video_id}_multi_speaker_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(processed_data['multi_speaker_analysis'], f, indent=2)
            
            # Save individual chunk files
            chunks_dir = self.output_dir / f"{video_id}_chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            for chunk in processed_data['chunks']:
                chunk_file = chunks_dir / f"{chunk['chunk_id']}.json"
                with open(chunk_file, 'w') as f:
                    json.dump(chunk, f, indent=2)
            
            # Create and save enhanced RAG chunks
            enhanced_chunks = self.create_enhanced_rag_chunks(processed_data['qa_pairs'])
            enhanced_chunks_file = self.output_dir / "enhanced_chunks" / f"{video_id}_enhanced_rag_chunks.json"
            with open(enhanced_chunks_file, 'w') as f:
                json.dump(enhanced_chunks, f, indent=2)
            logger.info(f"Enhanced RAG chunks saved: {enhanced_chunks_file}")
            
            # Create fine-tuning datasets
            self.create_fine_tuning_datasets(processed_data['qa_pairs'], video_id)
            
            logger.info(f"All outputs saved for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error saving outputs for video {video_id}: {e}")
    
    def run(self):
        """Run the multi-speaker transcript processor"""
        logger.info("🚀 Starting Multi-Speaker Transcript Processor for 1-on-2 Conversations")
        
        # Find transcript files
        transcript_files = list(self.input_dir.glob("*_transcript.json"))
        
        if not transcript_files:
            logger.warning("No transcript files found in step_03_transcription")
            return
        
        logger.info(f"Found {len(transcript_files)} transcript files to process")
        
        # Process each transcript
        for transcript_file in transcript_files:
            logger.info(f"\n📝 Processing {transcript_file.name}")
            
            processed_data = self.process_transcript_file(transcript_file)
            
            if "error" in processed_data:
                logger.error(f"Failed to process {transcript_file.name}: {processed_data['error']}")
                continue
            
            # Save outputs
            self.save_outputs(processed_data['video_id'], processed_data)
            
            logger.info(f"✅ Successfully processed {transcript_file.name}")
        
        logger.info("\n🎉 Multi-Speaker Transcript Processing completed!")

def main():
    """Main entry point"""
    processor = MultiSpeakerTranscriptProcessor()
    processor.run()

if __name__ == "__main__":
    main()
