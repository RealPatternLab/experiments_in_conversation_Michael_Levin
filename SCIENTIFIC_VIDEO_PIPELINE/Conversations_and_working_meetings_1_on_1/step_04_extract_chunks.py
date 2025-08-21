#!/usr/bin/env python3
"""
Step 4: Advanced Transcript Processing for Virtual Levin
- Extract transcript and speakers from AssemblyAI
- Identify speakers interactively
- Generate labeled transcript (like a script)
- Extract and enhance Q/A pairs for fine-tuning
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

class AdvancedTranscriptProcessor:
    """Advanced transcript processor for virtual Levin training with sophisticated chunking"""
    
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.input_dir = Path("step_03_transcription")
        self.output_dir = Path("step_04_extract_chunks")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different outputs
        (self.output_dir / "finetune_data").mkdir(exist_ok=True)
        (self.output_dir / "enhanced_chunks").mkdir(exist_ok=True)
        
        # Speaker mappings
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
            with open(self.speaker_mappings_file, 'r') as f:
                self.speaker_mappings = json.load(f)
            logger.info(f"Loaded speaker mappings for {len(self.speaker_mappings)} videos")
    
    def save_speaker_mappings(self):
        """Save speaker mappings"""
        with open(self.speaker_mappings_file, 'w') as f:
            json.dump(self.speaker_mappings, f, indent=2)
        logger.info("Speaker mappings saved")
    
    def identify_speakers(self, video_id: str, speakers: List[str]) -> Dict[str, str]:
        """Interactively identify speakers"""
        print(f"\n🎭 SPEAKER IDENTIFICATION for {video_id}")
        print(f"Available speakers: {', '.join(speakers)}")
        print("=" * 50)
        
        speaker_names = {}
        
        for speaker_id in speakers:
            print(f"\nSpeaker {speaker_id}:")
            name = input("Enter name: ").strip()
            if not name:
                name = f"Speaker {speaker_id}"
            
            role = input("Enter role/organization: ").strip()
            if not role:
                role = "Unknown"
            
            speaker_names[speaker_id] = {
                'name': name,
                'role': role,
                'is_levin': 'levin' in name.lower() or 'michael' in name.lower()
            }
            
            print(f"✅ {speaker_id} = {name} ({role})")
        
        return speaker_names
    
    def generate_labeled_transcript(self, speakers_data: Dict, speaker_mappings: Dict) -> str:
        """Generate a labeled transcript with speaker names and timestamps"""
        logger.info("Generating labeled transcript with timestamps...")
        
        video_id = speakers_data.get('video_id', 'unknown')
        speaker_turns = speakers_data.get('speaker_turns', [])
        
        labeled_lines = []
        
        for turn in speaker_turns:
            speaker_id = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            start_time = turn.get('start', 0)
            end_time = turn.get('end', 0)
            
            # Get speaker name from mappings
            speaker_info = speaker_mappings.get(video_id, {}).get(speaker_id, {})
            speaker_name = speaker_info.get('name', f'Speaker {speaker_id}')
            
            # Convert timestamps to readable format
            start_timestamp = self._ms_to_timestamp(start_time)
            end_timestamp = self._ms_to_timestamp(end_time)
            
            # Format: [00:06:27 - 00:07:10] Michael Levin says: "text here"
            labeled_line = f"[{start_timestamp} - {end_timestamp}] {speaker_name} says: \"{text}\""
            labeled_lines.append(labeled_line)
        
        return '\n\n'.join(labeled_lines)
    
    def chunk_transcript_for_llm(self, labeled_transcript: str, max_chunk_size: int = 8000) -> List[str]:
        """Intelligently chunk the transcript for LLM processing"""
        logger.info(f"Chunking transcript of {len(labeled_transcript)} characters into LLM-friendly chunks")
        
        # Split by speaker turns (double newlines)
        speaker_turns = labeled_transcript.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for turn in speaker_turns:
            turn_size = len(turn)
            
            # If adding this turn would exceed the limit, save current chunk and start new one
            if current_size + turn_size > max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [turn]
                current_size = turn_size
            else:
                current_chunk.append(turn)
                current_size += turn_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        logger.info(f"Created {len(chunks)} chunks for LLM processing")
        return chunks
    
    def extract_qa_pairs_advanced(self, labeled_transcript: str, max_retries: int = 3) -> List[Dict]:
        """Extract Q/A pairs with sophisticated pattern recognition using chunked processing"""
        logger.info("Extracting Q/A pairs with advanced pattern recognition...")
        
        # Chunk the transcript for LLM processing
        transcript_chunks = self.chunk_transcript_for_llm(labeled_transcript)
        
        all_qa_pairs = []
        
        for i, chunk in enumerate(transcript_chunks):
            logger.info(f"Processing transcript chunk {i+1}/{len(transcript_chunks)} ({len(chunk)} chars)")
            
            # Use Gemini-optimized prompt but ensure proper JSON output format
            system_prompt = """You are a scientific conversation expert tasked with creating high-quality question-answer pairs for a virtual assistant based on the work of researcher Michael Levin. You will be provided with a transcript of a scientific conversation where Michael Levin is a participant. Your task is to generate multiple question-answer pairs based on this transcript."""
            
            user_prompt = f"""
            **Instructions:**

            1. **Identify Key Concepts:** First, identify the key scientific concepts, ideas, and arguments discussed in the provided transcript.

            2. **Generate Q&A Pairs:** Create question-answer pairs where the answers are excerpts from Levin's contributions to the conversation. Follow these guidelines:

                * **Explicit Questions:** If a question in the transcript directly precedes Levin's response, use that question as your question and the corresponding response as the answer.
                * **Synthetic Questions:** If a section of Levin's contribution does not have a directly preceding question, *synthesize a realistic question that Michael Levin might have been asked that would have elicited that particular response*. These questions should be specific and focused, reflecting the scientific context and Levin's known areas of expertise (e.g., bioelectricity, morphogenesis, regenerative medicine). Ensure the synthesized question accurately reflects the content and tone of the response. Avoid generic or overly broad questions.
                * **Contextual Accuracy:** Each question should be directly addressable using only the provided transcript excerpt as context.
                * **Multiple Questions per Response:** For longer responses from Levin, generate multiple questions and answers, breaking down the response into more manageable and focused units. Each Q&A pair should focus on a single scientific idea.
                * **High Quality:** Prioritize accuracy, clarity, and conciseness in both the questions and answers. The Q&A pairs should be informative and useful for training a virtual assistant that convincingly emulates Michael Levin's scientific reasoning and communication style.

            3. **Timestamp Extraction:** For each Q&A pair, you MUST extract the start and end timestamps from the labeled transcript. Look for the timestamp format [HH:MM:SS - HH:MM:SS] that corresponds to Levin's answer. If Levin's answer spans multiple timestamp ranges, use the first and last timestamps.

            4. **Output Format:** You MUST return a JSON object with a "qa_pairs" key containing an array of Q&A pairs. Each Q&A pair object must have these exact keys: "question", "answer", "questioner", "context", "question_type" (explicit/synthetic), "confidence" (high/medium/low), "start_timestamp", "end_timestamp".

            **Example Output Format:**
            {{
                "qa_pairs": [
                    {{
                        "question": "What role does bioelectricity play in morphogenesis?",
                        "answer": "Bioelectricity plays a crucial role in patterning and shaping tissues during development...",
                        "questioner": "Willem Nielsen",
                        "context": "Discussion about cellular automata and biological modeling",
                        "question_type": "explicit",
                        "confidence": "high",
                        "start_timestamp": "00:06:27",
                        "end_timestamp": "00:07:10"
                    }}
                ]
            }}

            **Transcript Chunk:**
            {chunk}

            Generate at least 3-5 Q&A pairs for this chunk. Prioritize generating synthetic questions where necessary to fully capture the richness of Levin's contributions. Remember to return ONLY valid JSON with the exact structure shown above, including timestamps for each Q&A pair."""
            
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    
                    qa_output = json.loads(response.choices[0].message.content)
                    qa_pairs = qa_output.get("qa_pairs", [])
                    
                    # Validate and clean the Q&A pairs
                    validated_pairs = []
                    for pair in qa_pairs:
                        if self._validate_qa_pair(pair):
                            # Add timestamp information from LLM response
                            pair_with_timestamps = self._process_qa_pair_timestamps(pair, i)
                            validated_pairs.append(pair_with_timestamps)
                    
                    if validated_pairs:
                        logger.info(f"  Found {len(validated_pairs)} Q&A pairs in chunk {i+1}")
                        all_qa_pairs.extend(validated_pairs)
                    
                    break  # Success, move to next chunk
                    
                except Exception as e:
                    logger.warning(f"  Attempt {attempt + 1} failed for chunk {i+1}: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to process chunk {i+1} after all attempts")
        
        # Remove duplicates (same question-answer pairs might appear in multiple chunks)
        unique_qa_pairs = self._deduplicate_qa_pairs(all_qa_pairs)
        
        logger.info(f"Extracted {len(unique_qa_pairs)} total validated Q&A pairs from {len(transcript_chunks)} chunks")
        return unique_qa_pairs
    
    def _deduplicate_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Remove duplicate Q&A pairs based on question and answer content"""
        seen = set()
        unique_pairs = []
        
        for pair in qa_pairs:
            # Create a key based on normalized question and answer
            question_key = ' '.join(pair['question'].lower().split())
            answer_key = ' '.join(pair['answer'].lower().split())
            pair_key = f"{question_key}|{answer_key}"
            
            if pair_key not in seen:
                seen.add(pair_key)
                unique_pairs.append(pair)
        
        return unique_pairs
    
    def _validate_qa_pair(self, pair: Dict) -> bool:
        """Validate that a Q&A pair has all required fields"""
        required_fields = [
            'question', 'answer', 'questioner', 'context', 
            'question_type', 'confidence', 'start_timestamp', 'end_timestamp'
        ]
        return all(field in pair and pair[field] for field in required_fields)
    
    def _process_qa_pair_timestamps(self, pair: Dict, chunk_index: int) -> Dict:
        """Process timestamp information from LLM response and convert to proper format"""
        try:
            # Get timestamps from LLM response
            start_timestamp = pair.get('start_timestamp', '00:00:00')
            end_timestamp = pair.get('end_timestamp', '00:00:00')
            
            # Convert timestamps to milliseconds for YouTube links
            start_time_ms = self._timestamp_to_ms(start_timestamp)
            end_time_ms = self._timestamp_to_ms(end_timestamp)
            
            # Create YouTube hyperlink
            video_id = "LKSH4QNqpIE"  # This should be extracted from the transcript
            youtube_link = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time_ms/1000)}s" if start_time_ms else ""
            
            return {
                **pair,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "youtube_link": youtube_link,
                "video_id": video_id,
                "chunk_index": chunk_index
            }
            
        except Exception as e:
            logger.warning(f"Failed to process timestamps for Q&A pair: {e}")
            return self._add_placeholder_timestamps(pair, chunk_index)
    
    def _timestamp_to_ms(self, timestamp: str) -> int:
        """Convert HH:MM:SS timestamp to milliseconds"""
        try:
            if not timestamp or timestamp == "00:00:00":
                return 0
            
            parts = timestamp.split(':')
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                
                total_seconds = (hours * 3600) + (minutes * 60) + seconds
                return total_seconds * 1000
                else:
                return 0
                
        except Exception:
            return 0
    
    def _add_placeholder_timestamps(self, pair: Dict, chunk_index: int) -> Dict:
        """Add placeholder timestamps when extraction fails."""
        return {
            **pair,
            "start_time_ms": None,
            "end_time_ms": None,
            "start_timestamp": "00:00:00",
            "end_timestamp": "00:00:00",
            "youtube_link": "",
            "video_id": "LKSH4QNqpIE",
            "chunk_index": chunk_index
        }
    
    def _ms_to_timestamp(self, milliseconds: int) -> str:
        """Convert milliseconds to HH:MM:SS format."""
        try:
            seconds = int(milliseconds / 1000)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            
        except Exception:
            return "00:00:00"
    
    def enhance_qa_pair_for_rag(self, qa_pair: Dict, max_retries: int = 3) -> Tuple[str, List[str]]:
        """Enhance a Q&A pair for RAG with rich context and topics"""
        question = qa_pair['question']
        answer = qa_pair['answer']
        questioner = qa_pair.get('questioner', 'interviewer')
        
        system_prompt = """You are an expert at creating enhanced content for semantic search systems focused on Michael Levin's scientific work. 

        You must:
        1. PRESERVE MICHAEL LEVIN'S EXACT WORDING AND TONE - keep his authentic voice as much as possible
        2. Only fill in the MINIMAL context needed for someone to understand the question and answer without prior conversation knowledge
        3. Replace unclear references (this, that, it, they) with specific terms, but keep Levin's original phrasing intact
        4. Make the enhanced text completely understandable without prior conversation context
        5. Identify precise scientific topics from the provided taxonomy
        6. Return valid JSON matching the specified schema
        
        CRITICAL: Your goal is to make the content self-contained while preserving Levin's authentic speaking style, tone, and original wording as much as possible."""

        user_prompt = f"""Create enhanced content for a RAG system about Michael Levin's scientific work.

        TASK: Combine a question and Michael Levin's answer into enhanced, searchable text that is COMPLETELY SELF-CONTAINED while preserving Levin's authentic voice.

        ENHANCED TEXT REQUIREMENTS:
        1. Start with a BRIEF context statement that captures the essence of the question and provides minimal necessary background
        2. Follow with "Michael Levin explains:" or "Michael Levin responds:"
        3. Then include Michael Levin's answer with MINIMAL modifications:
           - Keep Levin's EXACT wording, tone, and speaking style as much as possible
           - Only replace unclear references (this, that, it, they) with specific terms when absolutely necessary
           - Preserve all of Levin's original phrasing, scientific terminology, and authentic voice
        4. The enhanced text must be understandable without prior conversation context
        5. Focus on preserving Levin's authentic speaking style while adding minimal context

        EXAMPLE: If Levin says "when you do this, you get no eye at all" but "this" refers to injecting depolarized cells, then write "when you inject depolarized cells into tissue, you get no eye at all" - keeping his original tone and structure.

        BALANCE: Add minimal context for clarity while preserving maximum authenticity of Levin's original response.

        TOPICS REQUIREMENTS:
        Generate 1-4 specific, searchable topics from this taxonomy:
        {', '.join(self.scientific_topics)}

        CONVERSATION:
        Question from {questioner}: {question}

        Michael Levin's Answer: {answer}

        Provide the enhanced_text and topics fields in your JSON response. Remember: preserve Levin's authentic voice while making the content self-contained."""

        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                
                response_text = response.choices[0].message.content.strip()
                response_dict = json.loads(response_text)
                
                # Validate with Pydantic
                validated_response = EnhancedQAResponse(**response_dict)
                
                # Validation: ensure the enhanced text contains the core concepts from Levin's answer
                if not self._validate_enhanced_content_quality(validated_response.enhanced_text):
                    logger.warning(f"⚠️  Attempt {attempt + 1}: Enhanced content quality insufficient, retrying...")
                    continue
                
                logger.info(f"✅ Enhanced chunk created (length: {len(validated_response.enhanced_text)} chars, topics: {validated_response.topics})")
                return validated_response.enhanced_text, validated_response.topics
                
            except Exception as e:
                logger.warning(f"⚠️  Attempt {attempt + 1}: {e}")
                continue
        
        # Fallback: create simple enhanced text
        logger.warning(f"⚠️  LLM enhancement failed after {max_retries} attempts, using fallback format")
        fallback_text = f"Question from {questioner}: {question}\n\nMichael Levin responds: {answer}"
        fallback_topics = self._generate_fallback_topics(qa_pair)
        
        return fallback_text, fallback_topics
    
    def _validate_enhanced_content_quality(self, enhanced_text: str) -> bool:
        """Validate that the enhanced text has sufficient quality and content"""
        # Check if the enhanced text is substantial (not too short)
        if len(enhanced_text.strip()) < 50:
            return False
        
        # Check if it contains key scientific terms that indicate quality content
        scientific_indicators = [
            'michael levin', 'levin', 'explains', 'responds', 'research', 'study',
            'bioelectric', 'morphogenesis', 'regeneration', 'development', 'biology',
            'cellular', 'tissue', 'pattern', 'signal', 'mechanism', 'process'
        ]
        
        enhanced_lower = enhanced_text.lower()
        indicator_count = sum(1 for indicator in scientific_indicators if indicator in enhanced_lower)
        
        # Require at least 2 scientific indicators for quality content
        return indicator_count >= 2
    
    def _generate_fallback_topics(self, qa_pair: Dict) -> List[str]:
        """Generate fallback topics when LLM enhancement fails"""
        text_content = (qa_pair.get('question', '') + ' ' + qa_pair.get('answer', '')).lower()
        
        topic_keywords = {
            'intelligence_definition': ['intelligence', 'define', 'definition'],
            'xenobots': ['xenobot', 'xenobots'],
            'anthrobots': ['anthrobot', 'anthrobots'],
            'morphogenesis': ['morphogenesis', 'development', 'embryo'],
            'regeneration': ['regeneration', 'regrow', 'healing'],
            'bioelectricity': ['bioelectric', 'voltage', 'electric'],
            'plasticity': ['plasticity', 'plastic', 'adapt'],
            'goal_directed_behavior': ['goal', 'purpose', 'directed'],
            'scale_free_cognition': ['scale', 'cognition'],
            'collective_intelligence': ['collective', 'swarm'],
            'problem_solving': ['problem', 'solve', 'solving']
        }
        
        fallback_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                fallback_topics.append(topic)
        
        return fallback_topics[:4] if fallback_topics else ['general_conversation']
    
    def extract_levin_semantic_chunks_advanced(self, labeled_transcript: str, max_retries: int = 3) -> List[Dict]:
        """Extract Levin's semantic chunks with rich scientific taxonomy using chunked processing"""
        logger.info("Extracting Levin's semantic chunks with advanced taxonomy...")
        
        # Chunk the transcript for LLM processing
        transcript_chunks = self.chunk_transcript_for_llm(labeled_transcript)
        
        all_chunks = []
        
        for i, chunk in enumerate(transcript_chunks):
            logger.info(f"Processing transcript chunk {i+1}/{len(transcript_chunks)} ({len(chunk)} chars)")
            
            # Use Gemini-optimized prompt but ensure proper JSON output format
            system_prompt = """You are an expert in creating prompts for AI systems that extract semantic chunks from scientific conversations. Analyze the following conversation transcript, focusing solely on the contributions of "Levin". Identify and extract semantic chunks that represent his unique knowledge, insights, and expertise related to bioelectricity, regenerative biology, and/or morphogenesis."""
            
            user_prompt = f"""
            **Instructions for Chunk Extraction:**

            1. **Identify Key Concepts:** Prioritize chunks that explicitly define, explain, or exemplify Levin's theories, hypotheses, experimental methodologies, or interpretations of results. Avoid purely factual statements easily found elsewhere.

            2. **Focus on Levin's Perspective:** Extract chunks that reveal Levin's personal understanding, opinions, or novel approaches. Exclude statements that are merely summaries or restatements of generally accepted knowledge.

            3. **Maintain Contextual Integrity:** Each chunk should be a self-contained unit of meaning, grammatically correct, and ideally no longer than 2-3 sentences. Ensure that the chunk's meaning is not distorted when taken out of context.

            4. **Prioritize Novel Insights:** Favor chunks that highlight original ideas, innovative approaches, or controversial perspectives unique to Levin's work.

            5. **Use Clear Delimiters:** Begin each extracted chunk with the marker "CHUNK_BEGIN:" and end it with "CHUNK_END:"

            **Output Format:**

            You MUST return a JSON object with a "levin_semantic_chunks" key containing an array of chunks. Each chunk object must have these exact keys: "text", "topic", "insight", "context", "scientific_topics" (array of 1-4 topics from the taxonomy).

            **Example Output Format:**
            {{
                "levin_semantic_chunks": [
                    {{
                        "text": "Our research demonstrates that bioelectric signals act as a morphogenetic field, directing tissue patterning in a fundamentally new way.",
                        "topic": "Bioelectric Morphogenesis",
                        "insight": "Bioelectric signals create morphogenetic fields that direct tissue patterning",
                        "context": "Discussion about bioelectricity in development",
                        "scientific_topics": ["bioelectricity", "morphogenesis", "developmental_biology"]
                    }}
                ]
            }}

            **Input Transcript:** {chunk}

            **Note:** The transcript contains contributions identified as "Levin" and "other researchers". Only analyze the portions attributed to "Levin". Omit any contributions from "other researchers". Focus on extracting high-quality, concise, and informative chunks suitable for embedding in a RAG system for question answering.

            Remember to return ONLY valid JSON with the exact structure shown above."""

            for attempt in range(max_retries):
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                    
                    chunks_output = json.loads(response.choices[0].message.content)
                    levin_chunks = chunks_output.get("levin_semantic_chunks", [])
                    
                    # Validate chunks
                    validated_chunks = []
                    for chunk_data in levin_chunks:
                        try:
                            validated_chunk = LevinChunkResponse(**chunk_data)
                            validated_chunks.append(validated_chunk.model_dump())
                        except ValidationError as e:
                            logger.warning(f"  Invalid chunk in chunk {i+1}: {e}")
                            continue
                    
                    if validated_chunks:
                        logger.info(f"  Found {len(validated_chunks)} Levin chunks in chunk {i+1}")
                        all_chunks.extend(validated_chunks)
                    
                    break  # Success, move to next chunk
                    
                except Exception as e:
                    logger.warning(f"  Attempt {attempt + 1} failed for chunk {i+1}: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to process chunk {i+1} after all attempts")
        
        # Remove duplicates
        unique_chunks = self._deduplicate_levin_chunks(all_chunks)
        
        logger.info(f"Extracted {len(unique_chunks)} total validated Levin semantic chunks from {len(transcript_chunks)} chunks")
        return unique_chunks
    
    def _deduplicate_levin_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate Levin chunks based on text content"""
        seen = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Create a key based on normalized text
            text_key = ' '.join(chunk['text'].lower().split())
            
            if text_key not in seen:
                seen.add(text_key)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def create_enhanced_rag_chunks(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Create enhanced RAG chunks from Q&A pairs with timestamp information"""
        logger.info("Creating enhanced RAG chunks...")
        
        enhanced_chunks = []
        
        for i, qa_pair in enumerate(qa_pairs, 1):
            logger.info(f"Processing Q&A pair {i}/{len(qa_pairs)}")
            
            try:
                enhanced_text, topics = self.enhance_qa_pair_for_rag(qa_pair)
                
                # Create enhanced chunk with timestamp information
                enhanced_chunk = {
                    "id": f"enhanced_conv_chunk_{i:04d}",
                    "text": enhanced_text,
                    "topics": topics,
                    "original_question": qa_pair.get('question', ''),
                    "original_answer": qa_pair.get('answer', ''),
                    "questioner": qa_pair.get('questioner', ''),
                    "answerer": "michael_levin",
                    "video_id": qa_pair.get('video_id', ''),
                    "chunk_type": "enhanced_conversation_rag",
                    "enhancement_method": "gpt4_contextual_with_topics",
                    # Add timestamp information
                    "start_time_ms": qa_pair.get('start_time_ms'),
                    "end_time_ms": qa_pair.get('end_time_ms'),
                    "start_timestamp": qa_pair.get('start_timestamp', '00:00:00'),
                    "end_timestamp": qa_pair.get('end_timestamp', '00:00:00'),
                    "youtube_link": qa_pair.get('youtube_link', ''),
                    "chunk_index": qa_pair.get('chunk_index', 0)
                }
                
                enhanced_chunks.append(enhanced_chunk)
                    
            except Exception as e:
                logger.error(f"Failed to enhance Q&A pair {i}: {e}")
                continue
        
        return enhanced_chunks
    
    def create_fine_tuning_datasets(self, qa_pairs: List[Dict]):
        """Create datasets in different formats for fine-tuning"""
        logger.info("Creating fine-tuning datasets...")
        
        # OpenAI fine-tuning format (JSONL)
        openai_file = self.output_dir / "finetune_data" / "openai_finetune.jsonl"
        openai_file.parent.mkdir(exist_ok=True)
        
        with open(openai_file, 'w') as f:
            for qa in qa_pairs:
                openai_format = {
                    "messages": [
                        {"role": "user", "content": qa['question']},
                        {"role": "assistant", "content": qa['answer']}
                    ]
                }
                f.write(json.dumps(openai_format) + '\n')
        
        # Hugging Face format
        hf_file = self.output_dir / "finetune_data" / "huggingface_dataset.json"
        hf_dataset = []
        for qa in qa_pairs:
            hf_format = {
                "instruction": qa['question'],
                "output": qa['answer'],
                "context": qa.get('context', ''),
                "questioner": qa.get('questioner', 'unknown'),
                "question_type": qa.get('question_type', 'explicit'),
                "confidence": qa.get('confidence', 'medium')
            }
            hf_dataset.append(hf_format)
        
        with open(hf_file, 'w') as f:
            json.dump(hf_dataset, f, indent=2)
        
        # Summary statistics
        stats = {
            'total_qa_pairs': len(qa_pairs),
            'question_types': {},
            'confidence_levels': {},
            'avg_question_length': sum(len(qa['question'].split()) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'avg_answer_length': sum(len(qa['answer'].split()) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0
        }
        
        for qa in qa_pairs:
            q_type = qa.get('question_type', 'explicit')
            confidence = qa.get('confidence', 'medium')
            stats['question_types'][q_type] = stats['question_types'].get(q_type, 0) + 1
            stats['confidence_levels'][confidence] = stats['confidence_levels'].get(confidence, 0) + 1
        
        stats_file = self.output_dir / "finetune_data" / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Created fine-tuning datasets:")
        logger.info(f"   OpenAI format: {openai_file}")
        logger.info(f"   Hugging Face format: {hf_file}")
        logger.info(f"   Statistics: {stats_file}")
        
        return stats
    
    def process_video(self, video_id: str):
        """Process a single video's transcript and speakers data"""
        logger.info(f"Processing video: {video_id}")
        
        transcript_file = self.input_dir / f"{video_id}_transcript.json"
        speakers_file = self.input_dir / f"{video_id}_speakers.json"
        
        if not transcript_file.exists() or not speakers_file.exists():
            logger.error(f"Missing transcript or speakers file for {video_id}")
            return False
        
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        with open(speakers_file, 'r') as f:
            speakers_data = json.load(f)
        
        # 1. Identify speakers interactively (if not already mapped)
        if video_id not in self.speaker_mappings:
            unique_speakers = list(set(turn['speaker'] for turn in speakers_data.get('speaker_turns', [])))
            unique_speakers.sort()
            self.speaker_mappings[video_id] = self.identify_speakers(video_id, unique_speakers)
            self.save_speaker_mappings()
        else:
            logger.info(f"Using existing speaker mappings for {video_id}")
        
        video_speaker_map = self.speaker_mappings.get(video_id, {})
        
        # 2. Generate labeled transcript
        labeled_transcript = self.generate_labeled_transcript(speakers_data, video_speaker_map)
        labeled_transcript_file = self.output_dir / f"{video_id}_labeled_transcript.txt"
        with open(labeled_transcript_file, 'w') as f:
            f.write(labeled_transcript)
        logger.info(f"Labeled transcript saved: {labeled_transcript_file}")
        
        # 3. Extract Q&A pairs with advanced pattern recognition
        qa_pairs = self.extract_qa_pairs_advanced(labeled_transcript)
        
        # Add metadata to Q&A pairs
        for qa in qa_pairs:
            qa['video_id'] = video_id
            qa['extracted_at'] = datetime.now().isoformat()
        
        qa_pairs_file = self.output_dir / f"{video_id}_qa_pairs.json"
        with open(qa_pairs_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        logger.info(f"Q/A pairs saved: {qa_pairs_file}")
        
        # 4. Extract Levin's semantic chunks with rich taxonomy
        levin_chunks = self.extract_levin_semantic_chunks_advanced(labeled_transcript)
        
        # Add metadata to chunks
        for chunk in levin_chunks:
            chunk['video_id'] = video_id
            chunk['extracted_at'] = datetime.now().isoformat()
        
        levin_chunks_file = self.output_dir / f"{video_id}_levin_chunks.json"
        with open(levin_chunks_file, 'w') as f:
            json.dump(levin_chunks, f, indent=2)
        logger.info(f"Levin chunks saved: {levin_chunks_file}")
        
        # 5. Create enhanced RAG chunks
        enhanced_chunks = self.create_enhanced_rag_chunks(qa_pairs)
        enhanced_chunks_file = self.output_dir / "enhanced_chunks" / f"{video_id}_enhanced_rag_chunks.json"
        with open(enhanced_chunks_file, 'w') as f:
            json.dump(enhanced_chunks, f, indent=2)
        logger.info(f"Enhanced RAG chunks saved: {enhanced_chunks_file}")
        
        # 6. Create fine-tuning datasets
        self.create_fine_tuning_datasets(qa_pairs)
        
        logger.info(f"✅ Successfully processed {video_id}")
        return True
    
    def process_all_videos(self):
        """Process all video transcripts in the input directory"""
        transcript_files = list(self.input_dir.glob("*_transcript.json"))
        logger.info(f"Found {len(transcript_files)} transcript files")
        
        for transcript_file in transcript_files:
            video_id = transcript_file.stem.replace('_transcript', '')
            self.process_video(video_id)
        
        logger.info("Processing completed successfully")

def main():
    processor = AdvancedTranscriptProcessor()
    processor.process_all_videos()

if __name__ == "__main__":
    main()
