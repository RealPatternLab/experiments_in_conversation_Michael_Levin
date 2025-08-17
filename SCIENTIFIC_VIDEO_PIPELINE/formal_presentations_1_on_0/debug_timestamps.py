#!/usr/bin/env python3
"""
Debug script to test timestamp mapping logic
"""

import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_timestamp_mapping():
    """Debug the timestamp mapping logic"""
    
    # Load the transcript data
    transcript_file = Path("step_03_transcription/FzFFeRVEdUM_transcript.json")
    if not transcript_file.exists():
        logger.error("Transcript file not found")
        return
    
    with open(transcript_file, 'r') as f:
        transcript_data = json.load(f)
    
    # Get words and utterances
    words = transcript_data.get('words', [])
    utterances = transcript_data.get('utterances', [])
    
    logger.info(f"Total words: {len(words)}")
    logger.info(f"Total utterances: {len(utterances)}")
    
    # Show first few words
    logger.info("First 5 words:")
    for i, word in enumerate(words[:5]):
        logger.info(f"  Word {i}: '{word.get('text', '')}' at {word.get('start', 0)}ms - {word.get('end', 0)}ms")
    
    # Show first few utterances
    logger.info("First 3 utterances:")
    for i, utterance in enumerate(utterances[:3]):
        logger.info(f"  Utterance {i}: '{utterance.get('text', '')}' at {utterance.get('start', 0)}ms - {utterance.get('end', 0)}ms")
    
    # Test the sentence mapping logic
    test_sentences = [
        "But But what happens with this, with this eye is that it makes an optic nerve",
        "That optic nerve does not go to the brain",
        "It synapses on the spinal cord or sometimes on the gut, sometimes nowhere at all"
    ]
    
    logger.info("\nTesting sentence mapping:")
    
    # Sort words by start time
    all_words = sorted(words, key=lambda x: x.get('start', 0))
    
    # Create a continuous text stream with timing information
    full_text = ""
    word_timings = []
    
    for word_data in all_words:
        word_text = word_data.get('text', '').strip()
        if word_text:
            full_text += word_text + " "
            word_timings.append({
                'text': word_text,
                'start': word_data.get('start', 0),
                'end': word_data.get('end', 0)
            })
    
    full_text = full_text.strip()
    logger.info(f"Full text length: {len(full_text)} characters")
    logger.info(f"Full text preview: {full_text[:200]}...")
    
    # Test finding sentences in the full text
    for sentence in test_sentences:
        sentence_clean = sentence.strip()
        sentence_start_pos = full_text.find(sentence_clean)
        
        if sentence_start_pos == -1:
            logger.warning(f"❌ Could not find sentence: {sentence_clean[:50]}...")
            continue
        
        logger.info(f"✅ Found sentence at position {sentence_start_pos}: {sentence_clean[:50]}...")
        
        # Find which words correspond to this sentence
        sentence_end_pos = sentence_start_pos + len(sentence_clean)
        sentence_words = []
        current_pos = 0
        
        for word_timing in word_timings:
            word_start = current_pos
            word_end = current_pos + len(word_timing['text']) + 1  # +1 for space
            
            # Check if this word overlaps with our sentence
            if (word_start < sentence_end_pos and word_end > sentence_start_pos):
                sentence_words.append(word_timing)
            
            current_pos = word_end
        
        if sentence_words:
            sentence_start = sentence_words[0]['start']
            sentence_end = sentence_words[-1]['end']
            logger.info(f"  Mapped to: {sentence_start}ms - {sentence_end}ms ({len(sentence_words)} words)")
        else:
            logger.warning(f"  No words mapped to sentence")

if __name__ == "__main__":
    debug_timestamp_mapping()
