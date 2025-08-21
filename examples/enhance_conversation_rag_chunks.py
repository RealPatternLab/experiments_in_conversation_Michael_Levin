#!/usr/bin/env python3
"""
Enhance Conversation RAG Chunks

This script takes Q&A pairs and processes them with an LLM to create enhanced RAG chunks
that include contextual questions followed by Michael Levin's complete answers.
"""

import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


class EnhancedChunkResponse(BaseModel):
    """Pydantic model for validating LLM responses."""
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

class ConversationRAGEnhancer:
    def __init__(self, playlist_dir: str):
        self.playlist_dir = Path(playlist_dir)
        self.qa_pairs_file = self.playlist_dir / "finetune_data" / "qa_pairs.json"
        self.rag_chunks_file = self.playlist_dir / "rag_chunks" / "conversation_chunks.json"
        self.enhanced_rag_file = self.playlist_dir / "rag_chunks" / "enhanced_conversation_chunks.json"
        
    def load_qa_pairs(self) -> List[Dict]:
        """Load Q&A pairs from the finetune data."""
        with open(self.qa_pairs_file, 'r') as f:
            return json.load(f)
    
    def enhance_qa_pair(self, qa_pair: Dict, max_retries: int = 3) -> tuple[str, List[str]]:
        """Use LLM to enhance a Q&A pair into a single contextual RAG chunk with validation."""
        question = qa_pair['question']
        answer = qa_pair['answer']
        questioner = qa_pair.get('questioner', 'interviewer')
        
        # Generate JSON schema for the response
        response_schema = EnhancedChunkResponse.model_json_schema()
        
        system_prompt = """You are an expert at creating enhanced content for semantic search systems focused on Michael Levin's scientific work. 

You must:
1. Preserve Michael Levin's exact words (NEVER modify, skip, or rephrase his content)
2. Add helpful context from the question
3. Identify precise scientific topics
4. Return valid JSON matching the specified schema"""

        user_prompt = f"""Create enhanced content for a RAG system about Michael Levin's scientific work.

TASK: Combine a question and Michael Levin's answer into enhanced, searchable text.

ENHANCED TEXT REQUIREMENTS:
1. Start with a brief context statement that captures the essence of the question
2. Follow with "Michael Levin explains:" or "Michael Levin responds:" 
3. Then include Michael Levin's COMPLETE answer (EXACT COPY - do not change, skip, or modify any words)
4. The result should flow naturally and be searchable

TOPICS REQUIREMENTS:
Generate 1-4 specific, searchable topics from these Michael Levin research areas:
- intelligence_definition, scale_free_cognition, unconventional_intelligence, seti_suti
- bioelectricity, morphogenesis, regeneration, development, healing
- xenobots, anthrobots, synthetic_biology, living_machines
- goal_directed_behavior, emergent_properties, collective_intelligence
- plasticity, adaptation, problem_solving, cybernetics
- computational_biology, information_theory, complexity
- morphospace_navigation, pattern_formation, self_organization

CONVERSATION:
Question from {questioner}: {question}

Michael Levin's Answer: {answer}

Provide the enhanced_text and topics fields in your JSON response."""

        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model="gpt-4-1106-preview",  # GPT-4 Turbo with JSON support
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,  # Lower temperature for more consistent JSON
                    max_tokens=4096,  # Increased for long Q&A pairs
                    response_format={"type": "json_object"}  # Force JSON response - supported by gpt-4-turbo
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse and validate with Pydantic
                try:
                    response_dict = json.loads(response_text)
                    validated_response = EnhancedChunkResponse(**response_dict)
                    
                    enhanced_text = validated_response.enhanced_text
                    topics = validated_response.topics
                    
                    # Additional validation: ensure Levin's answer is preserved
                    if not self._validate_answer_preservation(answer, enhanced_text):
                        logger.warning(f"⚠️  Attempt {attempt + 1}: Answer not fully preserved, retrying...")
                        continue
                    
                    logger.info(f"✅ Enhanced chunk created (length: {len(enhanced_text)} chars, topics: {topics})")
                    return enhanced_text, topics
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️  Attempt {attempt + 1}: JSON decode error: {e}")
                    continue
                    
                except ValidationError as e:
                    logger.warning(f"⚠️  Attempt {attempt + 1}: Validation error: {e}")
                    continue
                
            except Exception as e:
                logger.warning(f"⚠️  Attempt {attempt + 1}: API error: {e}")
                continue
        
        # If LLM enhancement fails, create a fallback enhanced text
        logger.warning(f"⚠️  LLM enhancement failed after {max_retries} attempts, using fallback format")
        
        # Create a simple but effective fallback format
        fallback_text = f"Question from {questioner}: {question}\n\nMichael Levin responds: {answer}"
        
        # Generate basic topics based on the original topic and content analysis
        fallback_topics = self._generate_fallback_topics(qa_pair)
        
        logger.info(f"📝 Fallback chunk created (length: {len(fallback_text)} chars, topics: {fallback_topics})")
        return fallback_text, fallback_topics
    
    def _validate_answer_preservation(self, original_answer: str, enhanced_text: str) -> bool:
        """Validate that the original answer is preserved in the enhanced text."""
        # Remove extra whitespace and normalize
        original_normalized = ' '.join(original_answer.split())
        enhanced_normalized = ' '.join(enhanced_text.split())
        
        # Check if the original answer appears in the enhanced text
        # Allow for minor formatting differences
        return original_normalized.lower() in enhanced_normalized.lower()
    
    def _generate_fallback_topics(self, qa_pair: Dict) -> List[str]:
        """Generate fallback topics when LLM enhancement fails."""
        # Start with the original topic if it's meaningful
        original_topic = qa_pair.get('topic', '')
        fallback_topics = []
        
        # Use original topic if it's not generic
        if original_topic and original_topic not in ['general', 'conversation']:
            fallback_topics.append(original_topic)
        
        # Do simple keyword matching for common topics
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
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                if topic not in fallback_topics:
                    fallback_topics.append(topic)
        
        # Ensure we have at least one topic
        if not fallback_topics:
            fallback_topics = ['general_conversation']
        
        # Limit to 4 topics max
        return fallback_topics[:4]
    
    def create_enhanced_rag_chunks(self) -> List[Dict]:
        """Create enhanced RAG chunks from all Q&A pairs."""
        logger.info("🔄 Loading Q&A pairs...")
        qa_pairs = self.load_qa_pairs()
        
        enhanced_chunks = []
        
        for i, qa_pair in enumerate(qa_pairs, 1):
            logger.info(f"📝 Processing Q&A pair {i}/{len(qa_pairs)}")
            
            enhanced_text, topics = self.enhance_qa_pair(qa_pair)
            
            # Now always create a chunk since enhance_qa_pair always returns a result
            enhanced_chunk = {
                "id": f"enhanced_conv_chunk_{i:04d}",
                "text": enhanced_text,
                "topics": topics,  # New: AI-generated topics
                "original_topic": qa_pair.get('topic', 'general'),  # Keep original for comparison
                "original_question": qa_pair['question'],
                "original_answer": qa_pair['answer'],
                "questioner": qa_pair.get('questioner', 'unknown'),
                "answerer": qa_pair.get('answerer', 'michael_levin'),
                "video_id": qa_pair.get('video_id', ''),
                "question_start_time": qa_pair.get('question_start_time', 0),
                "answer_start_time": qa_pair.get('answer_start_time', 0),
                "chunk_type": "enhanced_conversation_rag",
                "enhancement_method": "gpt4_contextual_with_topics"
            }
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def save_enhanced_chunks(self, enhanced_chunks: List[Dict]):
        """Save the enhanced RAG chunks to file."""
        # Create directory if it doesn't exist
        self.enhanced_rag_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.enhanced_rag_file, 'w') as f:
            json.dump(enhanced_chunks, f, indent=2)
        
        logger.info(f"💾 Saved {len(enhanced_chunks)} enhanced chunks to: {self.enhanced_rag_file}")
    
    def generate_comparison_report(self, enhanced_chunks: List[Dict]):
        """Generate a report comparing original vs enhanced chunks."""
        # Load original RAG chunks for comparison
        if self.rag_chunks_file.exists():
            with open(self.rag_chunks_file, 'r') as f:
                original_chunks = json.load(f)
        else:
            original_chunks = []
        
        report = {
            "enhancement_summary": {
                "original_rag_chunks": len(original_chunks),
                "enhanced_chunks": len(enhanced_chunks),
                "qa_pairs_processed": len(enhanced_chunks)
            },
            "avg_lengths": {
                "original_avg_length": sum(len(chunk.get('text', '')) for chunk in original_chunks) / len(original_chunks) if original_chunks else 0,
                "enhanced_avg_length": sum(len(chunk['text']) for chunk in enhanced_chunks) / len(enhanced_chunks) if enhanced_chunks else 0
            },
            "content_types": {
                "original_type": "levin_answers_only",
                "enhanced_type": "contextualized_qa_pairs"
            }
        }
        
        report_file = self.playlist_dir / "rag_chunks" / "enhancement_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Enhancement Report:")
        logger.info(f"   Original RAG chunks: {report['enhancement_summary']['original_rag_chunks']}")
        logger.info(f"   Enhanced chunks: {report['enhancement_summary']['enhanced_chunks']}")
        logger.info(f"   Average length increase: {report['avg_lengths']['enhanced_avg_length'] - report['avg_lengths']['original_avg_length']:.0f} chars")
        logger.info(f"   Report saved: {report_file}")
    
    def process_playlist(self):
        """Main processing function."""
        logger.info(f"🎬 Processing playlist: {self.playlist_dir.name}")
        
        if not self.qa_pairs_file.exists():
            logger.error(f"❌ Q&A pairs file not found: {self.qa_pairs_file}")
            return
        
        # Create enhanced chunks
        enhanced_chunks = self.create_enhanced_rag_chunks()
        
        if not enhanced_chunks:
            logger.error("❌ No enhanced chunks were created")
            return
        
        # Save results
        self.save_enhanced_chunks(enhanced_chunks)
        self.generate_comparison_report(enhanced_chunks)
        
        logger.info("🎉 Enhancement complete!")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance conversation RAG chunks with contextual Q&A")
    parser.add_argument("playlist_dir", help="Path to playlist directory (e.g., 'data/ingested/youtube/podcasts - 1 - on -1')")
    
    args = parser.parse_args()
    
    enhancer = ConversationRAGEnhancer(args.playlist_dir)
    enhancer.process_playlist()


if __name__ == "__main__":
    main() 
