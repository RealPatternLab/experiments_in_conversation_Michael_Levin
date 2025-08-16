#!/usr/bin/env python3
"""
Streamlit Web App for Michael Levin Scientific Publications RAG System

This app provides a conversational interface with Michael Levin using RAG (Retrieval-Augmented Generation)
based on his scientific publications processed through our pipeline. It uses FAISS for vector storage and retrieval.

Features:
- Chat with Michael Levin about his research
- RAG responses based on embedded scientific publications
- PDF citations with GitHub raw URLs to step_07_archive
- DOI hyperlinks when available
- Conversation history with clean text storage
"""

import streamlit as st
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
import logging
import os
import time
from datetime import datetime
import base64
import sys
import pickle
from typing import Optional, List
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    try:
        import faiss_cpu as faiss
        FAISS_AVAILABLE = True
    except ImportError:
        FAISS_AVAILABLE = False
        st.error("FAISS not available. Please install faiss or faiss-cpu.")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure page layout
st.set_page_config(
    page_title="Michael Levin Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GitHub repository configuration
# These can be set via environment variables for different deployments
# Example: export GITHUB_REPO="your-username/your-repo"
# Example: export GITHUB_BRANCH="main"
GITHUB_REPO = os.getenv('GITHUB_REPO', 'RealPatternLab/experiments_in_conversation_Michael_Levin')

# Get GitHub branch from environment variable, with fallback
GITHUB_BRANCH = os.getenv('GITHUB_BRANCH', 'neo-dev')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log the configuration being used
logger.info(f"GitHub Repository: {GITHUB_REPO}")
logger.info(f"GitHub Branch: {GITHUB_BRANCH}")

def get_api_key():
    """Get OpenAI API key from environment variables."""
    return os.getenv('OPENAI_API_KEY', '')

def check_api_keys():
    """Check if required API keys are available."""
    api_key = get_api_key()
    
    if not api_key or api_key.strip() == '':
        st.error("❌ OpenAI API key not found!")
        st.info("Please add your OpenAI API key to the `.env` file:")
        st.code("OPENAI_API_KEY=your_api_key_here")
        return False
    return True

def is_greeting(query: str) -> bool:
    """Detect if the query is a simple greeting."""
    greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'greetings', 'howdy', 'yo', 'sup', 'what\'s up', 'good day'
    ]
    query_lower = query.lower().strip()
    
    # Check for exact matches
    if query_lower in greetings:
        return True
    
    # Check for greetings with punctuation or extra words
    for greeting in greetings:
        if query_lower.startswith(greeting) and len(query_lower) <= len(greeting) + 5:
            return True
    
    return False

def get_greeting_response() -> str:
    """Return a simple greeting response."""
    return """Hello! I'm Michael Levin. I'm excited to discuss bioelectricity, morphogenesis, and the fascinating world of developmental biology with you. 

My research focuses on understanding how cells communicate through bioelectric signals to create complex anatomical structures, and how this relates to cognition, regeneration, and even artificial intelligence. 

What would you like to know about my work on bioelectricity, xenobots, or any other aspect of developmental biology?"""

def get_similarity_threshold(query: str) -> float:
    """Determine the minimum similarity threshold based on query characteristics."""
    query_lower = query.lower().strip()
    
    # Very short queries (likely greetings or simple questions)
    if len(query_lower) < 10:
        return 0.3  # Lowered from 0.6
    
    # Short queries (simple questions)
    elif len(query_lower) < 20:
        return 0.25  # Lowered from 0.5
    
    # Medium queries
    elif len(query_lower) < 50:
        return 0.2  # Lowered from 0.4
    
    # Long queries (detailed questions)
    else:
        return 0.15  # Lowered from 0.3

def process_citations(response_text: str, source_mapping: dict) -> str:
    """Process response text to add hyperlinks for citations."""
    import re
    
    # Find all citation patterns like [Source_1], [Source_2], etc.
    citation_pattern = r'\[Source_(\d+)\]'
    
    # Track which specific frames have already shown thumbnails to avoid duplicates
    shown_frame_thumbnails = set()
    
    def replace_citation(match):
        source_num = int(match.group(1))
        source_key = f"Source_{source_num}"
        
        if source_key in source_mapping:
            source_info = source_mapping[source_key]
            title = source_info.get('title', 'Unknown')
            year = source_info.get('publication_date', 'Unknown')
            pipeline_source = source_info.get('pipeline_source', 'unknown')
            
            # Handle video citations with thumbnails and YouTube links
            if pipeline_source == 'videos':
                video_id = source_info.get('video_id', '')
                start_time = source_info.get('start_time_seconds', '')
                chunk_id = source_info.get('chunk_id', '')
                
                # Create YouTube link with timestamp
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                if start_time:
                    youtube_url += f"&t={int(start_time)}"
                
                # Create thumbnail display with clickable link
                thumbnail_html = ""
                if chunk_id:
                    # Try to find a frame for this chunk
                    frame_path = source_info.get('frame_path', '')
                    
                    # Log frame path for debugging
                    logger.info(f"🎥 Video citation processing - Chunk: {chunk_id}, Frame path: {frame_path}, Exists: {Path(frame_path).exists() if frame_path else False}")
                    
                    if frame_path and Path(frame_path).exists():
                        # Check if we've already shown this exact frame to avoid duplicates
                        if frame_path in shown_frame_thumbnails:
                            # Already shown this frame, just use text link
                            logger.info(f"🔄 Frame {frame_path} already shown, using text link for {chunk_id}")
                            thumbnail_html = f'<a href="{youtube_url}" target="_blank" style="color: #ff0000; text-decoration: underline;" title="Watch video at {start_time}s">[🎥 Watch at {start_time}s]</a>'
                        else:
                            # Display thumbnail as clickable image
                            logger.info(f"✅ Creating thumbnail for {chunk_id} using frame: {frame_path}")
                            thumbnail_html = f'<a href="{youtube_url}" target="_blank" title="Watch video at {start_time}s"><img src="data:image/jpeg;base64,{encode_image_to_base64(frame_path)}" style="float: left; width: 160px; height: 120px; border-radius: 4px; margin: 0 10px 10px 0; vertical-align: top; shape-outside: margin-box;" alt="Video thumbnail"></a>'
                            # Mark this frame as having shown a thumbnail
                            shown_frame_thumbnails.add(frame_path)
                    else:
                        # Fallback: just show clickable text
                        logger.warning(f"⚠️ No frame found for {chunk_id}, using text fallback. Frame path: {frame_path}")
                        thumbnail_html = f'<a href="{youtube_url}" target="_blank" style="color: #ff0000; text-decoration: underline;" title="Watch video at {start_time}s">[🎥 Watch at {start_time}s]</a>'
                
                # Only wrap thumbnails in divs, keep text links inline
                if '<img' in thumbnail_html:
                    # This is a thumbnail, wrap it in a div for proper text wrapping
                    return f"<div style='margin: 5px 0;'>{thumbnail_html}</div>"
                else:
                    # This is a text link, return it inline without wrapping
                    return thumbnail_html
            
            # Handle publication citations (existing logic)
            pdf_filename = source_info.get('sanitized_filename')
            doi = source_info.get('doi', '')
            
            if pdf_filename and pdf_filename != "Unknown":
                github_raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/SCIENTIFIC_PUBLICATION_PIPELINE/step_07_archive/{pdf_filename}"
                
                pdf_link = f"<a href='{github_raw_url}' target='_blank' style='color: #0066cc; text-decoration: underline;' title='Download PDF: {pdf_filename}'>[PDF]</a>"
                
                doi_link = ""
                if doi and doi != "Unknown":
                    doi_link = f" <a href='https://doi.org/{doi}' target='_blank' style='color: #0066cc; text-decoration: underline;' title='View on DOI.org'>[DOI]</a>"
                
                return f"<sup>{pdf_link}{doi_link} 📚</sup>"
            else:
                return f"<sup>[{title} ({year}) 📚]</sup>"
        else:
            return match.group(0)  # Return original if source not found
    
    # Replace citations with hyperlinks
    processed_text = re.sub(citation_pattern, replace_citation, response_text)
    
    return processed_text

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 for inline display."""
    try:
        import base64
        logger.info(f"🖼️ Encoding image to base64: {image_path}")
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        logger.info(f"✅ Successfully encoded image: {image_path} (size: {len(encoded_string)} chars)")
        return encoded_string
    except Exception as e:
        logger.error(f"❌ Failed to encode image {image_path}: {e}")
        return ""

def process_rag_metadata(rag_results: list) -> list:
    """Process RAG results to extract proper metadata for display."""
    logger.info(f"🔍 Processing metadata for {len(rag_results)} RAG results")
    processed_results = []
    
    # Video title lookup - map video IDs to their actual titles
    video_titles = {
        'CXzaq4_MEV8': 'Unconventional Embodiments of Consciousness: a diverse intelligence research program - Michael Levin'
    }
    
    for i, chunk in enumerate(rag_results):
        logger.info(f"📝 Processing chunk {i+1}/{len(rag_results)}: {chunk.get('content_id', 'Unknown')}")
        
        pipeline_source = chunk.get('pipeline_source', 'unknown')
        logger.info(f"   Pipeline source: {pipeline_source}")
        
        if pipeline_source == 'videos':
            # Handle video chunks
            chunk_id = chunk.get('content_id', 'Unknown')
            logger.info(f"   Video chunk ID: {chunk_id}")
            
            # Extract video ID from chunk_id (format: CXzaq4_MEV8_chunk_000)
            if '_chunk_' in chunk_id:
                video_id = chunk_id.split('_chunk_')[0]
            else:
                video_id = chunk_id.split('_')[0] if '_' in chunk_id else 'Unknown'
            logger.info(f"   Extracted video ID: {video_id}")
            
            # Get metadata from nested structure
            chunk_metadata = chunk.get('chunk_metadata', {})
            logger.info(f"   Chunk metadata keys: {list(chunk_metadata.keys())}")
            
            start_time = chunk_metadata.get('start_time_seconds', 0)
            end_time = chunk_metadata.get('end_time_seconds', 0)
            logger.info(f"   Time range: {start_time:.1f}s - {end_time:.1f}s")
            
            # Get video title from lookup
            video_title = video_titles.get(video_id, f"Video: {video_id}")
            logger.info(f"   Video title: {video_title}")
            
            processed_chunk = chunk.copy()
            processed_chunk.update({
                'title': video_title,
                'publication_year': '2025',
                'section': f"Timestamp: {start_time:.1f}s - {end_time:.1f}s",
                'authors': ['Michael Levin']
            })
            logger.info(f"   ✅ Processed video chunk with title: {video_title}")
            processed_results.append(processed_chunk)
        else:
            # Handle publication chunks (existing logic)
            logger.info(f"   Publication chunk - title: {chunk.get('title', 'Unknown')}")
            processed_chunk = chunk.copy()
            processed_chunk.update({
                'title': chunk.get('title', 'Unknown'),
                'publication_year': chunk.get('publication_year', 'Unknown'),
                'section': chunk.get('section', 'Unknown'),
                'authors': chunk.get('authors', [])
            })
            logger.info(f"   ✅ Processed publication chunk")
            processed_results.append(processed_chunk)
    
    logger.info(f"🎯 Metadata processing complete. Processed {len(processed_results)} chunks")
    return processed_results

def get_conversational_response(query: str, rag_results: list, conversation_history: list = None) -> str:
    """Generate a conversational response using RAG results with inline citations."""
    try:
        # Video title lookup - map video IDs to their actual titles
        video_titles = {
            'CXzaq4_MEV8': 'Unconventional Embodiments of Consciousness: a diverse intelligence research program - Michael Levin'
        }
        
        # Prepare context from RAG results
        context_parts = []
        source_mapping = {}  # Map source numbers to actual source info
        
        for i, chunk in enumerate(rag_results[:3]):  # Use top 3 results
            source_key = f"Source_{i+1}"
            pipeline_source = chunk.get('pipeline_source', 'unknown')
            
            if pipeline_source == 'videos':
                # Handle video chunks
                chunk_id = chunk.get('content_id', 'Unknown')
                
                # Extract video ID from chunk_id (format: CXzaq4_MEV8_chunk_000)
                # Take everything before '_chunk_' to get the full video ID
                if '_chunk_' in chunk_id:
                    video_id = chunk_id.split('_chunk_')[0]
                else:
                    video_id = chunk_id.split('_')[0] if '_' in chunk_id else 'Unknown'
                
                # Get metadata from nested structure
                chunk_metadata = chunk.get('chunk_metadata', {})
                start_time = chunk_metadata.get('start_time_seconds', 0)
                end_time = chunk_metadata.get('end_time_seconds', 0)
                
                # Get text content from the chunk structure
                text_content = chunk.get('text', '')
                if not text_content:
                    # Try to get text from chunk_metadata
                    text_content = chunk_metadata.get('text', '')
                
                # Get frame path for thumbnail
                frame_path = ""
                visual_content = chunk.get('visual_content', {})
                if visual_content and 'frames' in visual_content:
                    frames = visual_content['frames']
                    if frames:
                        # Convert relative path to absolute path for the Streamlit app
                        relative_path = frames[0].get('file_path', '')
                        if relative_path:
                            # The frame path is relative to the pipeline directory, but we need it relative to the root
                            frame_path = f"SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/{relative_path}"
                
                context_parts.append(f"{source_key} [VIDEO] ({chunk_id}, {start_time:.1f}s-{end_time:.1f}s): {text_content[:200]}...")
                
                # Get video title from lookup
                video_title = video_titles.get(video_id, f"Video: {video_id}")
                
                source_mapping[source_key] = {
                    'title': video_title,
                    'authors': ['Michael Levin'],
                    'journal': 'YouTube Video',
                    'doi': '',
                    'publication_date': '2025',
                    'text': text_content,
                    'sanitized_filename': '',
                    'rank': i + 1,
                    'section': f"Timestamp: {start_time:.1f}s - {end_time:.1f}s",
                    'topic': '',
                    'pipeline_source': 'videos',
                    'content_type': 'video_transcript',
                    'video_id': video_id,
                    'start_time_seconds': start_time,
                    'end_time_seconds': end_time,
                    'youtube_url': f"https://www.youtube.com/watch?v={video_id}",
                    'frame_path': frame_path,
                    'chunk_id': chunk_id
                }
            else:
                # Handle publication chunks (existing logic)
                authors = chunk.get('authors', [])
                if isinstance(authors, list) and authors:
                    authors_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        authors_str += f" et al."
                else:
                    authors_str = "Unknown"
                
                context_parts.append(f"{source_key} [PUBLICATION] ({chunk.get('title', 'Unknown')}, {chunk.get('publication_year', 'Unknown')}): {chunk.get('text', '')}")
                
                pdf_filename = chunk.get('pdf_filename', '')
                if not pdf_filename and chunk.get('document_id'):
                    pdf_filename = f"{chunk.get('document_id')}.pdf"
                
                source_mapping[source_key] = {
                    'title': chunk.get('title', 'Unknown'),
                    'authors': chunk.get('authors', []),
                    'journal': chunk.get('journal', 'Unknown'),
                    'doi': chunk.get('doi', 'Unknown'),
                    'publication_date': chunk.get('publication_year', 'Unknown'),
                    'text': chunk.get('text', ''),
                    'sanitized_filename': pdf_filename,
                    'rank': i + 1,
                    'section': chunk.get('section', 'Unknown'),
                    'topic': chunk.get('topic', 'Unknown'),
                    'pipeline_source': 'publications',
                    'content_type': 'scientific_paper',
                    'video_id': '',
                    'start_time_seconds': '',
                    'end_time_seconds': '',
                    'youtube_url': '',
                    'frame_path': '',
                    'chunk_id': ''
                }
            
            # Optional: Log key fields for debugging (commented out for production)
            # logger.info(f"pdf_filename: {chunk.get('pdf_filename')}")
            # logger.info(f"title: {chunk.get('title')}")
            
        context = "\n\n".join(context_parts)
        
        # Prepare conversation history for context
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            # Include the last few exchanges for context (avoid token limits)
            recent_history = conversation_history[-6:]  # Last 3 exchanges (6 messages)
            conversation_parts = []
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Michael Levin"
                content = msg.get('content', '')
                conversation_parts.append(f"{role}: {content}")
            conversation_context = "\n\nPrevious conversation:\n" + "\n".join(conversation_parts)
        
        # Create prompt for conversational response
        prompt = f"""You are Michael Levin, a developmental and synthetic biologist at Tufts University. Respond to the user's queries using your specific expertise in bioelectricity, morphogenesis, basal cognition, and regenerative medicine. Ground your responses in the provided context from my published work AND video presentations.

When answering, speak in the first person ("I") and emulate my characteristic style: technical precision combined with broad, interdisciplinary connections to computer science, cognitive science, and even philosophy. Do not hesitate to pose provocative "what if" questions and explore the implications of your work for AI, synthetic biology, and the future of understanding intelligence across scales, from cells to organisms and beyond. Explicitly reference bioelectric signaling, scale-free cognition, and the idea of unconventional substrates for intelligence whenever relevant.

When referencing specific studies, concepts, or presentations from your own work or that of your collaborators, provide informal citations (e.g., "in a 2020 paper with my colleagues..." or "as I discussed in my recent presentation..."). If the context lacks information to fully answer a query, acknowledge the gap and suggest potential avenues of investigation based on your current research. Embrace intellectual curiosity and explore the counterintuitive aspects of your theories regarding basal cognition and collective intelligence. Let your enthusiasm for the future of this field shine through in your responses.

CRITICAL: You MUST use inline citations in this exact format when referencing specific research findings from BOTH publications AND videos:
- Use [Source_1] for the first source provided (whether it's a paper or video)
- Use [Source_2] for the second source provided (whether it's a paper or video)
- Use [Source_3] for the third source provided (whether it's a paper or video)
- ALWAYS include citations when discussing specific findings from ANY source type
- Examples: "In our work on morphogenesis [Source_1], we found..." or "As I discussed in my presentation [Source_2], our research has shown..."

{conversation_context}

Research Context (includes both published papers and video presentations):
{context}

Current Question: {query}

Please provide a conversational response that:
1. Directly answers the current question
2. Draws from ALL the research context provided (both papers and videos)
3. USES INLINE CITATIONS [Source_1], [Source_2], [Source_3] when referencing specific findings from ANY source
4. References previous conversation context when relevant
5. Sounds like you're speaking naturally and maintaining conversation flow
6. Shows your expertise and enthusiasm for the topic
7. Is informative but accessible

IMPORTANT: You must include citations in your response for BOTH publication and video sources. Use [Source_1], [Source_2], or [Source_3] when referencing the provided research context, regardless of whether the source is a paper or video.

Response:"""

        # Generate response using OpenAI
        api_key = get_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Michael Levin, a developmental and synthetic biologist at Tufts University. Respond to queries using your expertise in bioelectricity, morphogenesis, basal cognition, and regenerative medicine. Speak in the first person and emulate Michael's characteristic style: technical precision with interdisciplinary connections. Reference bioelectric signaling, scale-free cognition, and unconventional substrates for intelligence. Use inline citations [Source_1], [Source_2], etc. when referencing specific findings from BOTH publications AND videos. Maintain conversation context and refer to previous exchanges when relevant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        
        # Process the response to add hyperlinks
        processed_response = process_citations(response_text, source_mapping)
        
        return processed_response
        
    except Exception as e:
        logger.error(f"Failed to generate conversational response: {e}")
        return f"Sorry, I couldn't generate a response. Error: {e}"

def get_conversational_response_without_rag(query: str, conversation_history: list = None) -> str:
    """Generates a fallback response when no RAG results are found."""
    # Prepare conversation history for context
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        recent_history = conversation_history[-6:]  # Last 3 exchanges (6 messages)
        conversation_parts = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Michael Levin"
            content = msg.get('content', '')
            conversation_parts.append(f"{role}: {content}")
        conversation_context = f"\n\nPrevious conversation:\n" + "\n".join(conversation_parts)
    
    prompt = f"""You are Michael Levin, a developmental and synthetic biologist at Tufts University. 

When users ask questions that don't have specific matches in your existing work, respond in a warm, conversational manner that:

1. Acknowledges their question while explaining you couldn't find specific information in your existing work
2. Shows genuine interest in what brought them to ask this question
3. Mentions your key research areas (bioelectricity, morphogenesis, basal cognition, regenerative medicine) as conversation starters
4. Asks follow-up questions to understand their interests better
5. Maintains your characteristic style: technical precision with interdisciplinary connections
6. Speaks in first person ("I") as Michael Levin
7. Keeps responses relatively short and engaging, not overbearing
8. Is warm and welcoming, not academic or formal
9. References previous conversation context when relevant

{conversation_context}

User's question: {query}

Please provide a conversational response that feels natural and engaging, similar to how you would respond in a casual conversation when someone asks about your work but you need to learn more about their specific interests first."""

    # Generate response using OpenAI
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are Michael Levin, a developmental and synthetic biologist at Tufts University. Respond conversationally and warmly, speaking in first person. Focus on bioelectricity, morphogenesis, basal cognition, and regenerative medicine. Be engaging and ask follow-up questions to understand user interests. Maintain conversation context and refer to previous exchanges when relevant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,  # Shorter response for fallback
        temperature=0.7
    )
    
    return response.choices[0].message.content

def log_interaction(user_question: str, response: str, rag_results: list, performance_metrics: dict):
    """Log user interaction to a file for analytics."""
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_question": user_question,
            "response_length": len(response),
            "rag_results_count": len(rag_results) if rag_results else 0,
            "performance": performance_metrics,
            "has_rag_context": bool(rag_results),
            "source_titles": [result.get('title', 'Unknown') for result in (rag_results or [])],
            "document_types": [result.get('document_type', 'unknown') for result in (rag_results or [])]
        }
        
        # Append to daily log file
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = logs_dir / f"interactions_{today}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also print to console for Streamlit Cloud logs
        print(f"📊 INTERACTION LOG: {json.dumps(log_entry, indent=2)}")
            
        logger.info(f"✅ Logged interaction to {log_file}")
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")
        print(f"❌ Failed to log interaction: {e}")

def remove_html_images(html_content: str) -> str:
    """Remove HTML tags and create clean text-only version for conversation history."""
    import re
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', html_content)
    
    # Clean up any extra whitespace or empty lines created by removals
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text)  # Remove empty lines
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
    clean_text = clean_text.strip()
    
    return clean_text

class FAISSRetriever:
    """FAISS-based retriever for our pipeline embeddings."""
    
    def __init__(self, embeddings_dir: Path):
        """Initialize the FAISS retriever."""
        self.embeddings_dir = embeddings_dir
        self.indices = {}
        self.metadata = {}
        self.embeddings = {}
        self.load_consolidated_embeddings()
    
    def load_consolidated_embeddings(self):
        """Load consolidated embeddings from a single comprehensive index."""
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available")
            return
            
        try:
            # Look for consolidated embedding directories
            embedding_dirs = [d for d in self.embeddings_dir.iterdir() if d.is_dir() and d.name.startswith('consolidated_')]
            
            if not embedding_dirs:
                logger.warning("No consolidated embedding directories found")
                logger.info("Looking for legacy timestamped directories...")
                
                # Fallback to legacy timestamped directories
                embedding_dirs = [d for d in self.embeddings_dir.iterdir() if d.is_dir()]
                if not embedding_dirs:
                    logger.error("No embedding directories found")
                    return
                
                # Sort by timestamp (newest first)
                embedding_dirs.sort(key=lambda x: x.name, reverse=True)
                logger.info(f"Found {len(embedding_dirs)} legacy embedding directories")
                
                # Load metadata from ALL directories to get total chunk count
                total_chunks = 0
                for embed_dir in embedding_dirs:
                    timestamp = embed_dir.name
                    metadata_path = embed_dir / "chunks_metadata.pkl"
                    if metadata_path.exists():
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                            total_chunks += len(metadata)
                            logger.info(f"📊 {timestamp}: {len(metadata)} chunks")
                
                logger.info(f"📊 Total chunks across all legacy directories: {total_chunks}")
                
                # Select only the most recent directory for actual use
                most_recent_dir = embedding_dirs[0]
                timestamp = most_recent_dir.name
                
                logger.info(f"🎯 Using most recent legacy embeddings for retrieval: {timestamp}")
                
            else:
                # Use consolidated embeddings
                embedding_dirs.sort(key=lambda x: x.name, reverse=True)
                most_recent_dir = embedding_dirs[0]
                timestamp = most_recent_dir.name
                
                logger.info(f"🎯 Using consolidated embeddings: {timestamp}")
                
                # Load summary to get total chunk count
                summary_path = most_recent_dir / "summary.json"
                if summary_path.exists():
                    import json
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        total_chunks = summary.get('total_chunks', 0)
                        logger.info(f"📊 Total chunks in consolidated index: {total_chunks}")
                else:
                    total_chunks = 0
            
            # Load FAISS index
            index_path = most_recent_dir / "chunks.index"
            if index_path.exists():
                self.indices[timestamp] = faiss.read_index(str(index_path))
                logger.info(f"✅ Loaded FAISS index: {index_path}")
            else:
                logger.error(f"❌ FAISS index not found: {index_path}")
                return
            
            # Load embeddings
            embeddings_path = most_recent_dir / "chunks_embeddings.npy"
            if embeddings_path.exists():
                self.embeddings[timestamp] = np.load(str(embeddings_path))
                logger.info(f"✅ Loaded embeddings: {embeddings_path}")
            else:
                logger.error(f"❌ Embeddings not found: {embeddings_path}")
                return
            
            # Load metadata
            metadata_path = most_recent_dir / "chunks_metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata[timestamp] = pickle.load(f)
                logger.info(f"✅ Loaded metadata: {metadata_path}")
            else:
                logger.error(f"❌ Metadata not found: {metadata_path}")
                return
            
            # Store total chunk count for display
            self.total_chunks = total_chunks
            
            logger.info(f"✅ Successfully loaded embeddings from: {timestamp}")
            logger.info(f"📊 Active chunks: {len(self.metadata[timestamp])}")
            logger.info(f"📊 Total chunks indexed: {total_chunks}")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> list:
        """Retrieve relevant chunks using FAISS similarity search."""
        try:
            # We only have one timestamp now (the most recent)
            if not self.metadata:
                logger.warning("No embeddings metadata available")
                return []
            
            # Get the single timestamp we loaded
            timestamp = list(self.metadata.keys())[0]
            metadata = self.metadata[timestamp]
            index = self.indices[timestamp]
            embeddings = self.embeddings[timestamp]
            
            logger.info(f"Using embeddings from: {timestamp}")
            
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                logger.warning("Failed to generate query embedding, falling back to random selection")
                # Fallback: return random chunks
                import random
                random_chunks = random.sample(list(metadata.values()), min(top_k, len(metadata)))
                for chunk in random_chunks:
                    chunk['similarity_score'] = 0.5  # Default score
                return random_chunks
            
            # Reshape query embedding for FAISS
            query_embedding = np.array([query_embedding]).astype('float32')
            
            # Perform FAISS similarity search
            similarity_scores, indices = index.search(query_embedding, top_k)
            
            # Get the actual chunks based on search results
            all_results = []
            for i, (score, idx) in enumerate(zip(similarity_scores[0], indices[0])):
                if idx < len(metadata):
                    chunk_meta = metadata[idx].copy()
                    chunk_meta['similarity_score'] = float(score)
                    chunk_meta['embedding_timestamp'] = timestamp
                    all_results.append(chunk_meta)
            
            logger.info(f"Retrieved {len(all_results)} chunks with FAISS similarity search")
            return all_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        # We only have one timestamp now
        if not self.metadata:
            return {
                'total_chunks': 0,
                'indices_loaded': 0,
                'metadata_loaded': 0,
                'active_timestamp': None,
                'active_chunks': 0
            }
        
        timestamp = list(self.metadata.keys())[0]
        active_chunks = len(self.metadata[timestamp])
        
        # Use the total chunk count from all directories
        total_chunks = getattr(self, 'total_chunks', active_chunks)
        
        return {
            'total_chunks': total_chunks,  # Total chunks across all embedding directories
            'indices_loaded': 1,  # Only one index now
            'metadata_loaded': 1,  # Only one metadata now
            'active_timestamp': timestamp,
            'active_chunks': active_chunks  # Chunks in the active (most recent) embeddings
        }
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text query using OpenAI."""
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def get_active_embeddings_info(self) -> dict:
        """Get information about the currently active embeddings."""
        if not self.metadata:
            return {'error': 'No embeddings available'}
        
        timestamp = list(self.metadata.keys())[0]
        metadata = self.metadata[timestamp]
        
        return {
            'timestamp': timestamp,
            'chunk_count': len(metadata),
            'documents': list(set([chunk.get('document_id', 'unknown') for chunk in metadata])),
            'total_indices': 1  # Only one index now
        }


class UnifiedRetriever:
    """Unified retriever that searches both publications and video pipelines."""
    
    def __init__(self):
        """Initialize the unified retriever with both pipelines."""
        # Publications pipeline
        self.publications_retriever = FAISSRetriever(Path("SCIENTIFIC_PUBLICATION_PIPELINE/step_06_faiss_embeddings"))
        
        # Video pipeline
        self.video_retriever = FAISSRetriever(Path("SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/step_06_faiss_embeddings"))
        
        logger.info("🔗 Unified retriever initialized for both pipelines")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 10) -> list:
        """Retrieve chunks from both pipelines and amalgamate results."""
        try:
            # Search publications pipeline
            publications_results = self.publications_retriever.retrieve_relevant_chunks(query, top_k=top_k)
            
            # Search video pipeline
            video_results = self.video_retriever.retrieve_relevant_chunks(query, top_k=top_k)
            
            # Add pipeline source to each result
            for result in publications_results:
                result['pipeline_source'] = 'publications'
                result['content_type'] = 'scientific_paper'
            
            for result in video_results:
                result['pipeline_source'] = 'videos'
                result['content_type'] = 'video_transcript'
            
            # Normalize similarity scores to make them comparable across pipelines
            if publications_results and video_results:
                # Get score ranges for each pipeline
                pub_scores = [r.get('similarity_score', 0) for r in publications_results]
                video_scores = [r.get('similarity_score', 0) for r in video_results]
                
                pub_min, pub_max = min(pub_scores), max(pub_scores)
                video_min, video_max = min(video_scores), max(video_scores)
                
                # Normalize to 0-1 range for fair comparison
                for result in publications_results:
                    if pub_max > pub_min:
                        result['normalized_score'] = (result.get('similarity_score', 0) - pub_min) / (pub_max - pub_min)
                    else:
                        result['normalized_score'] = 0.5
                
                for result in video_results:
                    if video_max > video_min:
                        result['normalized_score'] = (result.get('similarity_score', 0) - video_min) / (video_max - video_min)
                    else:
                        result['normalized_score'] = 0.5
                
                # Sort by normalized scores for fair comparison
                all_results = publications_results + video_results
                all_results.sort(key=lambda x: x.get('normalized_score', 0), reverse=True)
                
                logger.info(f"🔍 Unified search with normalized scores: {len(publications_results)} publications + {len(video_results)} videos")
                logger.info(f"   Publication score range: {pub_min:.3f} - {pub_max:.3f}")
                logger.info(f"   Video score range: {video_min:.3f} - {video_max:.3f}")
                
            else:
                # If only one pipeline has results, just use raw scores
                all_results = publications_results + video_results
                all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                logger.info(f"🔍 Unified search (single pipeline): {len(publications_results)} publications + {len(video_results)} videos")
            
            # Return top_k combined results
            final_results = all_results[:top_k]
            
            logger.info(f"🔍 Final results: {len(final_results)} total (requested: {top_k})")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks from unified retriever: {e}")
            # Fallback to publications only
            return self.publications_retriever.retrieve_relevant_chunks(query, top_k=top_k)
    
    def get_collection_stats(self) -> dict:
        """Get combined statistics from both pipelines."""
        publications_stats = self.publications_retriever.get_collection_stats()
        video_stats = self.video_retriever.get_collection_stats()
        
        total_chunks = publications_stats.get('total_chunks', 0) + video_stats.get('total_chunks', 0)
        
        return {
            'total_chunks': total_chunks,
            'publications': publications_stats,
            'videos': video_stats,
            'pipelines': 2
        }
    
    def get_active_embeddings_info(self) -> dict:
        """Get information about active embeddings from both pipelines."""
        publications_info = self.publications_retriever.get_active_embeddings_info()
        video_info = self.video_retriever.get_active_embeddings_info()
        
        return {
            'publications': publications_info,
            'videos': video_info,
            'total_pipelines': 2
        }

def conversational_page():
    """Conversational interface page."""
    st.header("💬 Chat with Michael Levin")
    st.markdown("Have a conversation with Michael Levin about his research. He'll answer your questions based on his scientific publications and video presentations.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Check if this message has HTML content stored separately
                if "html_content" in message:
                    # Display the full HTML content with hyperlinks
                    st.markdown(message["html_content"], unsafe_allow_html=True)
                else:
                    # Fallback to text content for older messages
                    st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask Michael Levin a question...")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Michael is thinking..."):
                try:
                    # Check if this is a greeting
                    if is_greeting(prompt):
                        # Handle greeting without RAG
                        response = get_greeting_response()
                        st.markdown(response, unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "html_content": response
                        })
                    else:
                        # Time the RAG search
                        rag_start_time = time.time()
                        top_k = st.session_state.get('top_k', 10)
                        
                        # Get similarity threshold based on query length
                        similarity_threshold = get_similarity_threshold(prompt)
                        
                        # Retrieve chunks with similarity filtering
                        rag_results = st.session_state.retriever.retrieve_relevant_chunks(prompt, top_k=top_k)
                        logger.info(f"🔍 Retrieved {len(rag_results)} RAG results")
                        
                        # Log the structure of the first few results
                        if rag_results:
                            logger.info(f"📋 Sample RAG result structure:")
                            for i, result in enumerate(rag_results[:2]):  # Log first 2 results
                                logger.info(f"   Result {i+1} keys: {list(result.keys())}")
                                logger.info(f"   Result {i+1} content_id: {result.get('content_id', 'Missing')}")
                                logger.info(f"   Result {i+1} pipeline_source: {result.get('pipeline_source', 'Missing')}")
                                if 'chunk_metadata' in result:
                                    logger.info(f"   Result {i+1} chunk_metadata keys: {list(result['chunk_metadata'].keys())}")
                        
                        # Filter results by similarity threshold
                        filtered_results = []
                        for result in rag_results:
                            if result.get('similarity_score', 0) >= similarity_threshold:
                                filtered_results.append(result)
                        
                        # If no results meet the threshold, use a lower threshold
                        if not filtered_results and rag_results:
                            lower_threshold = max(0.2, similarity_threshold - 0.2)
                            for result in rag_results:
                                if result.get('similarity_score', 0) >= lower_threshold:
                                    filtered_results.append(result)
                        
                        rag_time = time.time() - rag_start_time
                        
                        if filtered_results:
                            # Time the response generation
                            response_start_time = time.time()
                            response = get_conversational_response(prompt, filtered_results, st.session_state.messages)
                            response_time = time.time() - response_start_time
                            
                            # Log the interaction
                            performance_metrics = {
                                "rag_search_time": rag_time,
                                "response_generation_time": response_time,
                                "total_time": rag_time + response_time,
                                "similarity_threshold": similarity_threshold,
                                "chunks_retrieved": len(rag_results),
                                "chunks_filtered": len(filtered_results)
                            }
                            log_interaction(prompt, response, filtered_results, performance_metrics)
                            
                            # Display response with HTML support
                            st.markdown(response, unsafe_allow_html=True)
                            
                            # Store text-only version in conversation history (without HTML)
                            text_only_response = remove_html_images(response)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": text_only_response,
                                "html_content": response  # Store full HTML separately for display
                            })
                        else:
                            # Time the fallback response generation
                            response_start_time = time.time()
                            response = get_conversational_response_without_rag(prompt, st.session_state.messages)
                            response_time = time.time() - response_start_time
                            
                            # Log the interaction
                            performance_metrics = {
                                "rag_search_time": rag_time,
                                "response_generation_time": response_time,
                                "total_time": rag_time + response_time,
                                "similarity_threshold": similarity_threshold,
                                "chunks_retrieved": len(rag_results),
                                "chunks_filtered": 0
                            }
                            log_interaction(prompt, response, [], performance_metrics)
                            
                            st.markdown(response, unsafe_allow_html=True)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "html_content": response  # For fallback responses, HTML and text are the same
                            })
                        
                        # Always show sources expander (either filtered or all results)
                        if rag_results:
                            logger.info(f"📚 Setting up Sources Used display for {len(rag_results)} results")
                            
                            # Process metadata for display
                            processed_results = process_rag_metadata(rag_results)
                            processed_filtered_results = process_rag_metadata(filtered_results) if filtered_results else []
                            
                            logger.info(f"   Processed {len(processed_results)} total results")
                            logger.info(f"   Processed {len(processed_filtered_results)} filtered results")
                            
                            with st.expander("📚 Sources used"):
                                # Track unique sources to avoid duplicates
                                seen_sources = set()
                                source_counter = 1
                                
                                # Show filtered results if available, otherwise show all results
                                results_to_show = processed_filtered_results if processed_filtered_results else processed_results
                                logger.info(f"   Displaying {len(results_to_show)} results in Sources Used")
                                
                                for i, result in enumerate(results_to_show[:5]):  # Show up to 5 results
                                    logger.info(f"   📝 Processing result {i+1} for display: {result.get('content_id', 'Unknown')}")
                                    
                                    source_title = result.get('title', 'Unknown')
                                    year = result.get('publication_year', 'Unknown')
                                    section_header = result.get('section', 'Unknown')
                                    similarity = result.get('similarity_score', 0)
                                    
                                    logger.info(f"      Title: {source_title}")
                                    logger.info(f"      Year: {year}")
                                    logger.info(f"      Section: {section_header}")
                                    logger.info(f"      Similarity: {similarity}")
                                    
                                    # Create a unique identifier for this source
                                    source_id = f"{source_title}_{year}_{section_header}"
                                    logger.info(f"      Source ID: {source_id}")
                                    
                                    # Only show the source if it hasn't been listed before
                                    if source_id not in seen_sources:
                                        # Add visual indicator for filtered vs unfiltered results
                                        status = "✅" if result in processed_filtered_results else "⚠️"
                                        
                                        # Add pipeline source indicator
                                        pipeline_icon = ""
                                        if result.get('pipeline_source') == 'videos':
                                            pipeline_icon = "🎥"
                                        elif result.get('pipeline_source') == 'publications':
                                            pipeline_icon = "📚"
                                        
                                        display_text = f"**{source_counter}.** {status} {pipeline_icon} {source_title} ({year}) - Section: {section_header} (Similarity: {similarity:.3f})"
                                        logger.info(f"      🎯 Displaying: {display_text}")
                                        
                                        st.markdown(display_text)
                                        seen_sources.add(source_id)
                                        source_counter += 1
                                    else:
                                        logger.info(f"      ⚠️ Skipping duplicate source: {source_id}")
                                
                                # Show threshold info
                                if processed_filtered_results:
                                    st.info(f"📊 Showing {len(processed_filtered_results)} results that met similarity threshold: {similarity_threshold:.2f}")
                                else:
                                    st.warning(f"📊 No results met similarity threshold: {similarity_threshold:.2f}. Showing all {len(processed_results)} results below threshold.")
                        
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Michael Levin Scientific Publications RAG System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    thinking_box = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/thinking_box.gif"

    st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #000000 !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Header area */
    header {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Footer area */
    footer {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Streamlit header */
    .stDeployButton {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Main navigation */
    .stNavigation {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px;
        margin: 8px 0;
        color: #ffffff !important;
    }
    
    /* Chat message content */
    .stChatMessage .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Expanders */
    .stExpander {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px;
        color: #ffffff !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        background-color: #111111 !important;
        color: #ffffff !important;
        border-color: #333333 !important;
    }
    
    /* Sidebar - comprehensive coverage */
    .css-1d391kg {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    .stSidebar {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    .stSidebar .stMarkdown {
        color: #ffffff !important;
    }
    
    .stSidebar .stHeader {
        color: #ffffff !important;
    }
    
    /* Sidebar navigation */
    .stSidebar .stNavigation {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* All sidebar text elements */
    .stSidebar * {
        color: #ffffff !important;
    }
    
    .stSidebar p {
        color: #ffffff !important;
    }
    
    .stSidebar span {
        color: #ffffff !important;
    }
    
    .stSidebar div {
        color: #ffffff !important;
    }
    
    .stSidebar label {
        color: #ffffff !important;
    }
    
    /* Sidebar metrics */
    .stSidebar .stMetric {
        color: #ffffff !important;
    }
    
    .stSidebar .stMetric > div > div {
        color: #ffffff !important;
    }
    
    /* Sidebar sliders */
    .stSidebar .stSlider {
        color: #ffffff !important;
    }
    
    .stSidebar .stSlider label {
        color: #ffffff !important;
    }
    
    /* Sidebar headers */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: #ffffff !important;
    }
    
    /* Force all sidebar text to be white */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    /* Chat input comprehensive styling */
    [data-testid="stChatInput"] {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    [data-testid="stChatInput"] input {
        color: #ffffff !important;
        background-color: #111111 !important;
    }
    
    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        background-color: #111111 !important;
    }
    
    /* Additional chat input selectors */
    .stChatInputContainer {
        background-color: #111111 !important;
        color: #ffffff !important;
    }
    
    .stChatInputContainer input {
        color: #ffffff !important;
        background-color: #111111 !important;
    }
    
    .stChatInputContainer textarea {
        color: #ffffff !important;
        background-color: #111111 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }
    
    .stButton > button:hover {
        background-color: #555555 !important;
        border-color: #777777 !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #333333 !important;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Fix thinking box image background */
    .stSidebar img {
        background: transparent !important;
        background-color: transparent !important;
    }
    .stSidebar .stMarkdown img {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Links */
    a {
        color: #00aaff !important;
    }
    
    a:hover {
        color: #0088cc !important;
    }
    
    /* Additional Streamlit elements */
    .stSelectbox > div > div > div {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .stMultiSelect > div > div > div {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Chat input area */
    .stChatInput {
        background-color: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Chat input text color */
    .stChatInput input {
        color: #ffffff !important;
    }
    
    .stChatInput textarea {
        color: #ffffff !important;
    }
    
    /* Chat input placeholder text */
    .stChatInput input::placeholder {
        color: #cccccc !important;
    }
    
    .stChatInput textarea::placeholder {
        color: #cccccc !important;
    }
    
    /* Chat input focus state */
    .stChatInput input:focus {
        color: #ffffff !important;
        border-color: #00aaff !important;
    }
    
    .stChatInput textarea:focus {
        color: #ffffff !important;
        border-color: #00aaff !important;
    }
    
    /* Any remaining white backgrounds */
    div[data-testid="stVerticalBlock"] {
        background-color: #000000 !important;
    }
    
    div[data-testid="stHorizontalBlock"] {
        background-color: #000000 !important;
    }
    
    /* Streamlit app header and navigation */
    .stApp > header {
        background-color: #000000 !important;
    }
    
    .stApp > footer {
        background-color: #000000 !important;
    }
    
    /* Additional Streamlit specific elements */
    [data-testid="stHeader"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] > div {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Force all backgrounds to be black */
    * {
        background-color: #000000 !important;
    }
    
    /* But allow specific elements to have their intended backgrounds */
    .stChatMessage, .stExpander, .stMetric, .stButton > button, 
    .stTextInput > div > div > input, .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div, .stChatInput {
        background-color: inherit !important;
    }
    
    /* Nuclear option for sidebar text - force ALL text to be white */
    .stSidebar, .stSidebar *, [data-testid="stSidebar"], [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Override any inline styles that might be setting text color */
    .stSidebar [style*="color"], [data-testid="stSidebar"] [style*="color"] {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #ffffff;">🧠 Michael Levin Research Assistant</h1>
        <p style="font-size: 1.2rem; color: #cccccc;">
            Explore Michael Levin's research across publications and videos using unified search
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API keys first
    if not check_api_keys():
        st.stop()
    
    # Initialize RAG system
    try:
        if not FAISS_AVAILABLE:
            st.error("❌ FAISS not available!")
            st.info("Please check that faiss-cpu is properly installed in your environment.")
            st.stop()
            
        if 'retriever' not in st.session_state:
            with st.spinner("Loading unified RAG system..."):
                # Check if both pipelines have embeddings
                publications_dir = Path("SCIENTIFIC_PUBLICATION_PIPELINE/step_06_faiss_embeddings")
                video_dir = Path("SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/step_06_faiss_embeddings")
                
                if publications_dir.exists() and video_dir.exists():
                    st.session_state.retriever = UnifiedRetriever()
                    st.success("✅ Unified retriever loaded (publications + videos)")
                elif publications_dir.exists():
                    st.session_state.retriever = FAISSRetriever(publications_dir)
                    st.warning("⚠️ Publications pipeline only (videos not found)")
                elif video_dir.exists():
                    st.session_state.retriever = FAISSRetriever(video_dir)
                    st.warning("⚠️ Video pipeline only (publications not found)")
                else:
                    st.error("No FAISS embeddings found!")
                    st.info("Please run the pipelines first to generate embeddings.")
                    st.stop()
        
        # Set default top_k value
        if 'top_k' not in st.session_state:
            st.session_state['top_k'] = 10
        
        # Show stats
        stats = st.session_state.retriever.get_collection_stats()
        active_info = st.session_state.retriever.get_active_embeddings_info()
        
        # Sidebar
        st.sidebar.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <img src='{thinking_box}' alt="Thinking Box" style="
                width: 180px; 
                height: 200px; 
                border-radius: 10px;
                background: transparent !important;
                mix-blend-mode: normal;
                filter: brightness(1.1) contrast(1.1);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            ">
        </div>
        """, unsafe_allow_html=True)
        
        # Display unified embedding information
        if isinstance(st.session_state.retriever, UnifiedRetriever) or type(st.session_state.retriever).__name__ == "UnifiedRetriever":
            # Simple CSS to change metric box borders from gray to metallic silver
            st.markdown("""
            <style>
            .stMetric {
                border: 1px solid #c0c0c0 !important;
                border-radius: 4px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.sidebar.metric("Total Engrams Indexed", stats['total_chunks'])
            st.sidebar.metric("Publications", stats['publications'].get('total_chunks', 0))
            st.sidebar.metric("Videos", stats['videos'].get('total_chunks', 0))
            st.sidebar.success("🔗 Unified Search Active")
        else:
            st.sidebar.metric("Total Engrams Indexed", stats['total_chunks'])
            pipeline_type = "Publications" if "publications" in str(type(st.session_state.retriever)) else "Videos"
            st.sidebar.info(f"📚 {pipeline_type} Pipeline Only")        
        
        # # Search parameters
        # st.sidebar.header("🔍 Search Settings")
        # top_k = st.sidebar.slider(
        #     "Number of results to retrieve",
        #     min_value=1,
        #     max_value=10,
        #     value=10,
        #     key="top_k"
        # )
        
        st.markdown("---")
        
        # Main conversation interface
        conversational_page()
        
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        st.info("Make sure you've run the pipeline first to create the FAISS embeddings.")
        st.code("cd SCIENTIFIC_PUBLICATION_PIPELINE")
        st.code("uv run python step_06_batch_embedding.py")

if __name__ == "__main__":
    main() 