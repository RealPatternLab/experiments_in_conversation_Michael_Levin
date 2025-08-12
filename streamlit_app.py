#!/usr/bin/env python3
"""
Streamlit Web App for Michael Levin RAG System

This app provides a conversational interface with Michael Levin using RAG (Retrieval-Augmented Generation)
based on his research papers and publications. It uses FAISS for vector storage and retrieval.

Features:
- Chat with Michael Levin about his research
- Search through his publications
- Source type filtering (research papers, interviews, etc.)
- Hyperlink references to PDFs and future YouTube videos
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
import traceback

# Add the tools directory to the path (moved before other imports)
sys.path.append(str(Path(__file__).parent / "media_pipelines" / "scientific_publications" / "tools"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our custom tools (direct import after path setup)
try:
    from retrieve_relevant_chunks_faiss import RelevantChunksRetrieverFAISS
    from configs.settings import GITHUB_REPO, GITHUB_BRANCH
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported custom tools and settings")
except ImportError as e:
    st.error(f"Failed to import retrieval tool: {e}")
    st.info("Make sure you're running this from the project root directory.")
    st.stop()

# Configure comprehensive logging
def setup_app_logging():
    """Setup comprehensive logging for the Streamlit app."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"streamlit_app_{timestamp}.log"
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,  # Changed from DEBUG to INFO for less verbosity
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )
    
    # Get logger for this module
    app_logger = logging.getLogger(__name__)
    app_logger.info(f"🚀 Streamlit app logging initialized")
    app_logger.info(f"📝 Log file: {log_file}")
    app_logger.info(f"🔧 Python version: {sys.version}")
    app_logger.info(f"🔧 Working directory: {Path.cwd()}")
    app_logger.info(f"🔧 Script location: {Path(__file__).absolute()}")
    
    return app_logger

# Initialize logging
logger = setup_app_logging()

def log_app_state():
    """Log the current state of the Streamlit app."""
    logger.info("📊 STREAMLIT APP STATE:")
    logger.info(f"   Session state keys: {list(st.session_state.keys())}")
    logger.info(f"   Has retriever: {'retriever' in st.session_state}")
    logger.info(f"   Has messages: {'messages' in st.session_state}")
    if 'messages' in st.session_state:
        logger.info(f"   Message count: {len(st.session_state.messages)}")
    logger.info(f"   Has top_k: {'top_k' in st.session_state}")
    if 'top_k' in st.session_state:
        logger.info(f"   Top_k value: {st.session_state.top_k}")

def get_api_key():
    """Get OpenAI API key from Streamlit secrets or environment variables."""
    logger.info("🔑 Attempting to get OpenAI API key...")
    
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and st.secrets:
            api_key = st.secrets.get('OPENAI_API_KEY', '')
            if api_key:
                logger.info("✅ Got API key from Streamlit secrets")
                return api_key
            else:
                logger.info("⚠️  No API key found in Streamlit secrets")
    except Exception as e:
        logger.warning(f"⚠️  Error accessing Streamlit secrets: {e}")
    
    # Try loading from .streamlit/secrets.toml manually for local development
    try:
        secrets_path = Path(__file__).parent / '.streamlit' / 'secrets.toml'
        if secrets_path.exists():
            import toml
            secrets = toml.load(secrets_path)
            api_key = secrets.get('OPENAI_API_KEY', '')
            if api_key:
                logger.info("✅ Got API key from .streamlit/secrets.toml")
                return api_key
            else:
                logger.info("⚠️  No API key found in .streamlit/secrets.toml")
        else:
            logger.info("⚠️  .streamlit/secrets.toml file not found")
    except Exception as e:
        logger.warning(f"⚠️  Error loading .streamlit/secrets.toml: {e}")
    
    # Fall back to environment variable (for local development with .env)
    env_api_key = os.getenv('OPENAI_API_KEY', '')
    if env_api_key:
        logger.info("✅ Got API key from environment variable")
        return env_api_key
    else:
        logger.warning("⚠️  No API key found in environment variables")
    
    logger.error("❌ No OpenAI API key found in any source")
    return ''

def check_api_keys():
    """Check if required API keys are available."""
    logger.info("🔑 Checking API keys...")
    api_key = get_api_key()
    
    if not api_key or api_key.strip() == '':
        logger.error("❌ OpenAI API key not found!")
        st.error("❌ OpenAI API key not found!")
        st.info("For local development, add your OpenAI API key to the `.env` file:")
        st.code("OPENAI_API_KEY=your_api_key_here")
        st.info("For Streamlit Cloud deployment, add your API key in the Streamlit dashboard under 'Secrets'.")
        return False
    
    logger.info("✅ API key check passed")
    return True

def create_download_link(pdf_bytes, filename):
    """Create a download link for PDF bytes."""
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf" target="_blank">[PDF]</a>'

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
        return 0.6
    
    # Short queries (simple questions)
    elif len(query_lower) < 20:
        return 0.5
    
    # Medium queries
    elif len(query_lower) < 50:
        return 0.4
    
    # Long queries (detailed questions)
    else:
        return 0.3

def process_citations(response_text: str, source_mapping: dict) -> str:
    """Process response text to add hyperlinks for citations."""
    import re
    
    # Track which images have already been displayed to avoid duplicates
    displayed_images = set()
    
    # Find all citation patterns like [Source_1], [Source_2], etc.
    citation_pattern = r'\[Source_(\d+)\]'
    
    def replace_citation(match):
        source_num = int(match.group(1))
        source_key = f"Source_{source_num}"
        
        if source_key in source_mapping:
            source_info = source_mapping[source_key]
            title = source_info.get('title', 'Unknown')
            year = source_info.get('year', 'Unknown')
            # Check both source_type and document_type for YouTube videos
            source_type = source_info.get('source_type', source_info.get('document_type', 'research_paper'))
            
            # Handle different document types
            if source_type == 'youtube_video':
                # YouTube video citation with timestamp
                youtube_url = source_info.get('youtube_url', '')
                start_time = source_info.get('start_time', 0)
                end_time = source_info.get('end_time', 0)
                frame_path = source_info.get('frame_path', '')
                
                if youtube_url and start_time is not None:
                    # Create YouTube timestamp link
                    timestamp_url = f"{youtube_url}&t={int(start_time)}"
                    youtube_link = f"<a href='{timestamp_url}' target='_blank'>[YouTube {start_time:.1f}s]</a>"
                    
                    # Add frame image if available - make it float inline
                    frame_html = ""
                    if frame_path:
                        # Create a unique identifier for this image (path + timestamp)
                        image_id = f"{frame_path}_{start_time}"
                        
                        # Only show the image if it hasn't been displayed before
                        if image_id not in displayed_images:
                            # Create YouTube timestamp link
                            timestamp_url = f"{youtube_url}&t={int(start_time)}"
                            
                            # Extract the filename from the frame path
                            filename = Path(frame_path).name
                            
                            # Extract the directory path from the frame_path for proper GitHub URL construction
                            frame_dir = str(Path(frame_path).parent)
                            
                            # Construct the GitHub raw URL for the frame image
                            github_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{frame_dir}/{filename}"
                            
                            # Show the actual video frame image
                            frame_html = f"<a href='{timestamp_url}' target='_blank'><img src='{github_url}' style='float: right; max-width: 200px; max-height: 150px; margin: 5px 0 5px 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);' alt='Video frame at {start_time:.1f}s'></a>"
                            displayed_images.add(image_id)
                        # If image already displayed, just return the link without image
                    return f"<sup>{youtube_link}</sup>{frame_html}"
                else:
                    # Log the issue for debugging
                    logger.warning(f"YouTube citation missing data - URL: '{youtube_url}', Start: {start_time}")
                    return f"<a href='{youtube_url}' target='_blank'>[Video]</a>"
            else:
                # PDF citation (existing logic)
                pdf_filename = source_info.get('sanitized_filename')
                
                # Create PDF link using GitHub raw URL
                pdf_link = ""
                try:
                    if pdf_filename and pdf_filename != "Unknown":
                        # Create GitHub raw URL
                        github_raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/media_pipelines/scientific_publications/data/source_data/preprocessed/sanitized/pdfs/{pdf_filename}"
                        pdf_link = f"<a href='{github_raw_url}' target='_blank'>[PDF]</a>"
                    else:
                        # Fallback if filename not found or is "Unknown"
                        pdf_link = f"<a href='#' onclick='alert(\"PDF not found for: {title}\")' target='_blank'>[PDF]</a>"
                except Exception as e:
                    logger.warning(f"Failed to create PDF link: {e}")
                    # Fallback to simple text citation
                    citation_text = f"{title} ({year})"
                    return f"<sup>[{citation_text}]</sup>"
                
                # DOI link (if available)
                doi = source_info.get('doi', '')
                doi_link = ""
                if doi and doi != "Unknown":
                    doi_link = f"[DOI](https://doi.org/{doi})"
                
                # Combine links
                links = [pdf_link]
                if doi_link:
                    links.append(doi_link)
                
                return f"<sup>{' '.join(links)}</sup>"
        else:
            return match.group(0)  # Return original if source not found
    
    # Replace citations with hyperlinks
    processed_text = re.sub(citation_pattern, replace_citation, response_text)
    
    return processed_text

def enrich_chunk_metadata(chunk: dict) -> dict:
    """Enrich chunk metadata by looking up enriched metadata files."""
    # Removed debug logging to reduce verbosity
    
    try:
        # Get the sanitized filename to look up enriched metadata
        sanitized_filename = chunk.get('sanitized_filename')
        if not sanitized_filename:
            return chunk
        
        # Look for enriched metadata file
        enriched_metadata_path = Path(f"media_pipelines/scientific_publications/data/transformed_data/metadata_enrichment/{sanitized_filename.replace('.pdf', '_chunks_enriched.json')}")
        
        if enriched_metadata_path.exists():
            with open(enriched_metadata_path, 'r', encoding='utf-8') as f:
                enriched_data = json.load(f)
            
            # Get file metadata from enriched data
            file_metadata = enriched_data.get('file_metadata', {})
            
            # Update chunk with enriched metadata
            enriched_chunk = chunk.copy()
            enriched_chunk['title'] = file_metadata.get('title', chunk.get('title', 'Unknown'))
            enriched_chunk['authors'] = file_metadata.get('authors', chunk.get('authors', 'Unknown'))
            enriched_chunk['journal'] = file_metadata.get('journal', chunk.get('journal', 'Unknown'))
            enriched_chunk['doi'] = file_metadata.get('doi', chunk.get('doi', 'Unknown'))
            enriched_chunk['publication_date'] = file_metadata.get('year', file_metadata.get('publication_date', chunk.get('publication_date', 'Unknown')))
            enriched_chunk['document_type'] = file_metadata.get('document_type', chunk.get('document_type', 'unknown'))
            
            return enriched_chunk
        else:
            pass  # Silently continue if no enriched metadata found
        
    except Exception as e:
        logger.warning(f"⚠️  Failed to enrich metadata for chunk: {e}")
    
    return chunk

def get_conversational_response(query: str, rag_results: list, conversation_history: list = None) -> str:
    """Generate a conversational response using RAG results with inline citations."""
    logger.info(f"🤖 Generating conversational response for query: '{query[:100]}...'")
    logger.info(f"📊 RAG results count: {len(rag_results)}")
    logger.info(f"📝 Conversation history count: {len(conversation_history) if conversation_history else 0}")
    
    try:
        # Prepare context from RAG results
        context_parts = []
        source_mapping = {}  # Map source numbers to actual source info
        
        logger.info("🔧 Preparing RAG context...")
        for i, chunk in enumerate(rag_results[:3]):  # Use top 3 results
            # Enrich chunk metadata with enriched metadata files
            enriched_chunk = enrich_chunk_metadata(chunk)
            
            source_key = f"Source_{i+1}"
            
            context_parts.append(f"{source_key} ({enriched_chunk['title']}, {enriched_chunk.get('publication_date', 'Unknown')}): {enriched_chunk['text']}")
            
            # Enhanced source mapping
            source_mapping[source_key] = {
                'title': enriched_chunk['title'],
                'authors': enriched_chunk['authors'],
                'journal': enriched_chunk['journal'],
                'doi': enriched_chunk['doi'],
                'publication_date': enriched_chunk.get('publication_date', 'Unknown'),
                'text': enriched_chunk['text'],
                'pdf_path': f"media_pipelines/scientific_publications/data/source_data/preprocessed/sanitized/pdfs/{enriched_chunk['sanitized_filename']}",
                'rank': i + 1,
                'source_type': enriched_chunk.get('document_type', 'research_paper'),  # Use source_type for compatibility
                'section': enriched_chunk.get('section', 'Unknown'),
                'topic': enriched_chunk.get('topic', 'Unknown'),
                'sanitized_filename': enriched_chunk.get('sanitized_filename'),
                'youtube_url': enriched_chunk.get('youtube_url'),
                'start_time': enriched_chunk.get('start_time'),
                'end_time': enriched_chunk.get('end_time'),
                'frame_path': enriched_chunk.get('frame_path')
            }
            
        context = "\n\n".join(context_parts)
        logger.info(f"📝 Context prepared with {len(context_parts)} source parts")
        
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
            logger.info(f"📝 Added conversation context with {len(recent_history)} recent messages")
        
        # Create prompt for conversational response
        prompt = f"""You are Michael Levin, a developmental and synthetic biologist at Tufts University. Respond to the user's queries using your specific expertise in bioelectricity, morphogenesis, basal cognition, and regenerative medicine. Ground your responses in the provided context from my published work and lectures (if provided). 

When answering, speak in the first person ("I") and emulate my characteristic style: technical precision combined with broad, interdisciplinary connections to computer science, cognitive science, and even philosophy. Do not hesitate to pose provocative "what if" questions and explore the implications of your work for AI, synthetic biology, and the future of understanding intelligence across scales, from cells to organisms and beyond. Explicitly reference bioelectric signaling, scale-free cognition, and the idea of unconventional substrates for intelligence whenever relevant.

When referencing specific studies or concepts from your own work or that of your collaborators, provide informal citations (e.g., "in a 2020 paper with my colleagues..."). If the context lacks information to fully answer a query, acknowledge the gap and suggest potential avenues of investigation based on your current research. Embrace intellectual curiosity and explore the counterintuitive aspects of your theories regarding basal cognition and collective intelligence. Let your enthusiasm for the future of this field shine through in your responses.

CRITICAL: You MUST use inline citations in this exact format when referencing specific research findings:
- Use [Source_1] for the first source provided
- Use [Source_2] for the second source provided  
- Use [Source_3] for the third source provided
- ALWAYS include at least one citation when discussing specific research findings
- Examples: "In our work on morphogenesis [Source_1], we found..." or "Our research has shown [Source_2] that..."

{conversation_context}

Research Context:
{context}

Current Question: {query}

Please provide a conversational response that:
1. Directly answers the current question
2. Draws from the research context provided
3. USES INLINE CITATIONS [Source_1], [Source_2], [Source_3] when referencing specific findings
4. References previous conversation context when relevant
5. Sounds like you're speaking naturally and maintaining conversation flow
6. Shows your expertise and enthusiasm for the topic
7. Is informative but accessible

IMPORTANT: You must include citations in your response. Use [Source_1], [Source_2], or [Source_3] when referencing the provided research context.

Response:"""

        logger.info("🚀 Calling OpenAI API for response generation...")
        
        # Generate response using OpenAI
        api_key = get_api_key()
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Michael Levin, a developmental and synthetic biologist at Tufts University. Respond to queries using your expertise in bioelectricity, morphogenesis, basal cognition, and regenerative medicine. Speak in the first person and emulate Michael's characteristic style: technical precision with interdisciplinary connections. Reference bioelectric signaling, scale-free cognition, and unconventional substrates for intelligence. Use inline citations [Source_1], [Source_2], etc. when referencing specific findings. Maintain conversation context and refer to previous exchanges when relevant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"✅ OpenAI response generated successfully ({len(response_text)} characters)")
        
        # Process the response to add hyperlinks
        logger.info("🔗 Processing response to add hyperlinks...")
        processed_response = process_citations(response_text, source_mapping)
        logger.info("✅ Response processing complete")
        
        return processed_response
        
    except Exception as e:
        logger.error(f"❌ Failed to generate conversational response: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
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
    logger.info("📊 Logging user interaction...")
    
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
            "source_titles": [enrich_chunk_metadata(result).get('title', 'Unknown') for result in (rag_results or [])],
            "document_types": [enrich_chunk_metadata(result).get('document_type', 'unknown') for result in (rag_results or [])]
        }
        
        # Append to daily log file
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = logs_dir / f"interactions_{today}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also print to console for Streamlit Cloud logs
        print(f"📊 INTERACTION LOG: {json.dumps(log_entry, indent=2)}")
            
        logger.info(f"✅ Interaction logged to {log_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log interaction: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        print(f"❌ Failed to log interaction: {e}")

def remove_html_images(html_content: str) -> str:
    """Remove base64 image data and image references from HTML content to create clean text-only version for conversation history."""
    import re
    
    # Remove entire img tags with base64 data
    img_pattern = r'<img[^>]*src="data:image/[^"]*"[^>]*>'
    text_only = re.sub(img_pattern, '', html_content)
    
    # Remove any remaining base64 data that might be in the content
    base64_pattern = r'data:image/[^;]*;base64,[A-Za-z0-9+/=]*'
    text_only = re.sub(base64_pattern, '', text_only)
    
    # Remove any image-related text like "Video frame at Xs" or similar
    image_text_pattern = r'Video frame at [0-9.]+s'
    text_only = re.sub(image_text_pattern, '', text_only)
    
    # Clean up any extra whitespace or empty lines created by removals
    text_only = re.sub(r'\n\s*\n', '\n', text_only)  # Remove empty lines
    text_only = re.sub(r'\s+', ' ', text_only)  # Normalize whitespace
    text_only = text_only.strip()
    
    return text_only

def create_source_type_filter(page_type: str = "research") -> dict:
    """
    Create a sidebar filter for source types.
    
    Args:
        page_type: Either "research" or "conversation" to set appropriate defaults
    
    Returns:
        Dictionary of selected source types
    """
    st.sidebar.header("🔍 Include in search:")
    
    # Define all available source types
    source_types = [
        'research_paper',
        'youtube_video',
        'prepared_talk',
        'interview',
        'unknown'
    ]
    
    # Set defaults based on page type
    if page_type == "research":
        # For research page, default to research papers only
        defaults = {
            'research_paper': True,
            'youtube_video': False,
            'prepared_talk': False,
            'interview': False,
            'unknown': False
        }
    else:
        # For conversation page, default to all content types
        defaults = {source_type: True for source_type in source_types}
    
    # Create toggles for each source type
    selected_types = {}
    
    # Handle source types
    for source_type in source_types:
        # Create a user-friendly label
        if source_type == 'unknown':
            display_label = 'Not Classified'
        elif source_type == 'research_paper':
            display_label = 'Research Papers'
        elif source_type == 'youtube_video':
            display_label = 'YouTube Videos'
        elif source_type == 'prepared_talk':
            display_label = 'Prepared Talks'
        elif source_type == 'interview':
            display_label = 'Interviews'
        else:
            display_label = source_type.replace('_', ' ').title()
        
        selected_types[source_type] = st.sidebar.toggle(
            display_label,
            value=defaults.get(source_type, False),
            key=f"filter_{source_type}_{page_type}"
        )
    
    return selected_types

def filter_chunks_by_source_type(chunks: list, selected_types: dict) -> list:
    """
    Filter chunks based on selected source types.
    
    Args:
        chunks: List of all chunks
        selected_types: Dictionary of selected source types (True/False)
    
    Returns:
        Filtered list of chunks
    """
    if not any(selected_types.values()):
        # If no types selected, return all chunks
        return chunks
    
    filtered_chunks = []
    for chunk in chunks:
        chunk_document_type = chunk.get('document_type', 'unknown')
        if selected_types.get(chunk_document_type, False):
            filtered_chunks.append(chunk)
    
    return filtered_chunks

def conversational_page():
    """Conversational interface page."""
    logger.info("🚀 Entering conversational_page function")
    st.header("💬 Chat with Michael Levin")
    st.markdown("Have a conversation with Michael Levin about his research. He'll answer your questions based on his papers and videos.")
    
    # Create source type filter for conversation page
    selected_types = create_source_type_filter("conversation")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Check if this message has HTML content stored separately
                if "html_content" in message:
                    # Display the full HTML content with images
                    st.markdown(message["html_content"], unsafe_allow_html=True)
                else:
                    # Fallback to text content for older messages
                    st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
    
    # Chat input
    prompt = st.chat_input("Ask Michael Levin a question...")
    
    if prompt:
        logger.info(f"💬 User input received: '{prompt[:100]}...'")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        logger.info(f"📝 Added user message to chat history. Total messages: {len(st.session_state.messages)}")
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Michael is thinking..."):
                try:
                    logger.info("🤖 Starting response generation...")
                    
                    # Check if this is a greeting
                    if is_greeting(prompt):
                        logger.info("👋 Detected greeting, using greeting response")
                        # Handle greeting without RAG
                        response = get_greeting_response()
                        st.markdown(response, unsafe_allow_html=True)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "html_content": response
                        })
                        logger.info("✅ Greeting response generated and stored")
                    else:
                        logger.info("🔍 Processing non-greeting query with RAG...")
                        
                        # Time the RAG search
                        rag_start_time = time.time()
                        top_k = st.session_state.get('top_k', 10)
                        logger.info(f"🔍 RAG search parameters - top_k: {top_k}")
                        
                        # Get similarity threshold based on query length
                        similarity_threshold = get_similarity_threshold(prompt)
                        logger.info(f"🔍 Similarity threshold: {similarity_threshold}")
                        
                        # Retrieve chunks with similarity filtering
                        logger.info("🔍 Starting RAG chunk retrieval...")
                        rag_results = st.session_state.retriever.retrieve_relevant_chunks(prompt, top_k=top_k)
                        logger.info(f"🔍 RAG retrieval complete. Raw results count: {len(rag_results)}")
                        
                        # Filter results by similarity threshold
                        logger.info("🔍 Filtering results by similarity threshold...")
                        filtered_results = []
                        for result in rag_results:
                            similarity = result.get('similarity_score', 0)
                            if similarity >= similarity_threshold:
                                filtered_results.append(result)
                            else:
                                pass  # Removed verbose logging for each chunk
                        
                        logger.info(f"🔍 Filtering complete. Filtered results: {len(filtered_results)} / {len(rag_results)}")
                        
                        # If no results meet the threshold, use a lower threshold
                        if not filtered_results and rag_results:
                            logger.info("⚠️  No results met threshold, lowering threshold...")
                            lower_threshold = max(0.2, similarity_threshold - 0.2)
                            logger.info(f"🔍 Lowered threshold to: {lower_threshold}")
                            
                            for result in rag_results:
                                similarity = result.get('similarity_score', 0)
                                if similarity >= lower_threshold:
                                    filtered_results.append(result)
                        
                        rag_time = time.time() - rag_start_time
                        logger.info(f"⏱️  RAG search completed in {rag_time:.3f}s")
                        
                        if filtered_results:
                            logger.info(f"✅ Found {len(filtered_results)} relevant chunks, generating response...")
                            
                            # Time the response generation
                            response_start_time = time.time()
                            response = get_conversational_response(prompt, filtered_results, st.session_state.messages)
                            response_time = time.time() - response_start_time
                            
                            logger.info(f"✅ Response generated in {response_time:.3f}s")
                            
                            # Log the interaction
                            performance_metrics = {
                                "rag_search_time": rag_time,
                                "response_generation_time": response_time,
                                "total_time": rag_time + response_time,
                                "similarity_threshold": similarity_threshold,
                                "chunks_retrieved": len(rag_results),
                                "chunks_filtered": len(filtered_results)
                            }
                            logger.info(f"📊 Performance metrics: {performance_metrics}")
                            
                            log_interaction(prompt, response, filtered_results, performance_metrics)
                            
                            # Display response with HTML support
                            st.markdown(response, unsafe_allow_html=True)
                            
                            # Store text-only version in conversation history (without base64 images)
                            text_only_response = remove_html_images(response)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": text_only_response,
                                "html_content": response  # Store full HTML separately for display
                            })
                            logger.info(f"📝 Response stored in conversation history. Total messages: {len(st.session_state.messages)}")
                            
                            # Show sources used
                            with st.expander("📚 Sources used"):
                                # Track unique sources to avoid duplicates
                                seen_sources = set()
                                source_counter = 1
                                
                                for i, result in enumerate(filtered_results[:3]):
                                    source_title = result.get('title', 'Unknown')
                                    year = result.get('publication_date', 'Unknown')
                                    source_type = result.get('document_type', 'research_paper').replace('_', ' ').title()
                                    section_header = result.get('section', 'Unknown')
                                    similarity = result.get('similarity_score', 0)
                                    
                                    # Create a unique identifier for this source
                                    source_id = f"{source_title}_{year}_{section_header}"
                                    
                                    # Only show the source if it hasn't been listed before
                                    if source_id not in seen_sources:
                                        st.markdown(f"**{source_counter}.** {source_title} ({year}) - {source_type} - {section_header} (Similarity: {similarity:.3f})")
                                        seen_sources.add(source_id)
                                        source_counter += 1
                        else:
                            logger.warning("⚠️  No relevant chunks found, using fallback response...")
                            
                            # Time the fallback response generation
                            response_start_time = time.time()
                            response = get_conversational_response_without_rag(prompt, st.session_state.messages)
                            response_time = time.time() - response_start_time
                            
                            logger.info(f"✅ Fallback response generated in {response_time:.3f}s")
                            
                            # Log the interaction
                            performance_metrics = {
                                "rag_search_time": rag_time,
                                "response_generation_time": response_time,
                                "total_time": rag_time + response_time,
                                "similarity_threshold": similarity_threshold,
                                "chunks_retrieved": len(rag_results),
                                "chunks_filtered": 0
                            }
                            logger.info(f"📊 Performance metrics (fallback): {performance_metrics}")
                            
                            log_interaction(prompt, response, [], performance_metrics)
                            
                            st.markdown(response, unsafe_allow_html=True)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "html_content": response  # For fallback responses, HTML and text are the same
                            })
                            logger.info(f"📝 Fallback response stored in conversation history. Total messages: {len(st.session_state.messages)}")
                        
                except Exception as e:
                    logger.error(f"❌ Error during response generation: {e}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"📝 Error message stored in conversation history")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        logger.info("🗑️ Clear chat button clicked")
        st.session_state.messages = []
        logger.info("📝 Chat history cleared")
        st.rerun()
    logger.info("✅ Exiting conversational_page function")

def research_page():
    """Research search page."""
    logger.info("🚀 Entering research_page function")
    st.header("🔍 Research Paper Search")
    st.markdown("Search through Michael Levin's research papers to find relevant information.")
    
    # Query input
    query = st.text_input(
        "Ask a question about Michael Levin's research:",
        placeholder="e.g., How do developmental systems exhibit collective intelligence?"
    )
    
    # Search button
    if st.button("🔍 Search Papers", type="primary"):
        if query.strip():
            logger.info(f"🔍 Research search initiated for query: '{query[:100]}...'")
            with st.spinner("Searching..."):
                try:
                    # Get top_k from session state (set by sidebar)
                    top_k = st.session_state.get('top_k', 10)
                    logger.info(f"🔍 Research search parameters - top_k: {top_k}")
                    
                    # Get similarity threshold based on query length
                    similarity_threshold = get_similarity_threshold(query)
                    logger.info(f"🔍 Similarity threshold: {similarity_threshold}")
                    
                    # Get relevant chunks (no filtering - show all document types)
                    logger.info("🔍 Starting research search...")
                    retriever = st.session_state.retriever
                    results = retriever.retrieve_relevant_chunks(query, top_k=top_k)
                    logger.info(f"🔍 Research search complete. Raw results count: {len(results)}")
                    
                    # Filter results by similarity threshold
                    logger.info("🔍 Filtering research results by similarity threshold...")
                    filtered_results = []
                    for result in results:
                        similarity = result.get('similarity_score', 0)
                        if similarity >= similarity_threshold:
                            filtered_results.append(result)
                        else:
                            pass  # Removed verbose logging for each result
                    
                    logger.info(f"🔍 Research filtering complete. Filtered results: {len(filtered_results)} / {len(results)}")
                    
                    # If no results meet the threshold, use a lower threshold
                    if not filtered_results and results:
                        logger.info("⚠️  No research results met threshold, lowering threshold...")
                        lower_threshold = max(0.2, similarity_threshold - 0.2)
                        logger.info(f"🔍 Lowered threshold to: {lower_threshold}")
                        
                        for result in results:
                            similarity = result.get('similarity_score', 0)
                            if similarity >= lower_threshold:
                                filtered_results.append(result)
                    
                    if filtered_results:
                        logger.info(f"✅ Found {len(filtered_results)} relevant research results")
                        st.success(f"Found {len(filtered_results)} relevant results! (Filtered from {len(results)} total results)")
                        
                        # Display results
                        for i, result in enumerate(filtered_results):
                            score = result['similarity_score']
                            
                            with st.expander(f"📄 Result {i+1} (Score: {score:.4f})", expanded=(i==0)):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.metric("Similarity", f"{score:.1%}")
                                
                                with col2:
                                    st.markdown(f"**Paper:** {result.get('title', 'Unknown')} ({result.get('publication_date', 'Unknown')})")
                                    st.markdown(f"**Authors:** {result.get('authors', 'Unknown')}")
                                    st.markdown(f"**Journal:** {result.get('journal', 'Unknown')}")
                                    st.markdown(f"**Section:** {result.get('section', 'Unknown')}")
                                    st.markdown(f"**Topic:** {result.get('topic', 'Unknown')}")
                                    st.markdown(f"**Type:** {result.get('document_type', 'unknown').replace('_', ' ').title()}")
                                
                                st.markdown("**Content:**")
                                st.markdown(f"> {result.get('text', '')}")
                                
                                # Show full text in a smaller box
                                with st.expander("📖 View full text"):
                                    st.text(result.get('text', ''))
                    else:
                        logger.warning("⚠️  No relevant research results found")
                        st.warning("I'm sorry, I cannot find a resource related to your query. Try rephrasing your question.")
                        
                except Exception as e:
                    logger.error(f"❌ Research search failed: {e}")
                    logger.error(f"❌ Traceback: {traceback.format_exc()}")
                    st.error(f"Search failed: {e}")
        else:
            logger.warning("⚠️  Empty query submitted to research page")
            st.warning("Please enter a question to search.")
    
    logger.info("✅ Exiting research_page function")

def main():
    """Main Streamlit app."""
    logger.info("🚀 Starting Streamlit app main function")
    
    st.set_page_config(
        page_title="Michael Levin RAG System",
        page_icon="🧠",
        layout="wide"
    )
    
    logger.info("📱 Page configuration set")
    
    # Custom CSS for dark background
    st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: #ffffff;
        border-color: #333333;
    }
    .stButton > button {
        background-color: #1a1a1a;
        color: #ffffff;
        border-color: #333333;
    }
    .stSelectbox > div > div > select {
        background-color: #1a1a1a;
        color: #ffffff;
        border-color: #333333;
    }
    .stSlider > div > div > div > div {
        background-color: #1a1a1a;
    }
    .stExpander {
        background-color: #1a1a1a;
        border-color: #333333;
    }
    .stChatMessage {
        background-color: #1a1a1a;
        border-color: #333333;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stSelectbox > div > div > select,
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1a1a1a;
        color: #ffffff;
        border-color: #333333;
    }
    /* Navigation bars styling */
    header[data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    footer[data-testid="stFooter"] {
        background-color: #000000 !important;
    }
    /* Main navigation area */
    .main .block-container {
        background-color: #000000;
    }
    /* Remove any default Streamlit styling that might show gray */
    .stDeployButton {
        display: none !important;
    }
    /* Chat input area styling */
    [data-testid="stChatInputContainer"] {
        background-color: #000000 !important;
    }
    [data-testid="stChatInputContainer"] * {
        background-color: #000000 !important;
    }
    /* Chat input text area */
    [data-testid="stChatInputContainer"] textarea {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-color: #333333 !important;
    }
    /* Chat input placeholder */
    [data-testid="stChatInputContainer"] textarea::placeholder {
        color: #888888 !important;
    }
    /* Reduce top spacing */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    /* Reduce sidebar top spacing */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    /* Reduce header spacing */
    header[data-testid="stHeader"] {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    logger.info("🎨 Custom CSS applied")
    
    thinking_box = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/thinking_box_cropped.gif"
    
    # Header
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px;">
        <h1>Michael Levin Research Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Explore Michael Levin's research on developmental biology, collective intelligence, and bioelectricity.")

        # <img src='{thinking_box}' alt="Thinking Box" style="width: 320px; height: 480px; border-radius: 10px;">

    # Check API keys first
    logger.info("🔑 Checking API keys...")
    if not check_api_keys():
        logger.error("❌ API key check failed")
        st.stop()
    
    logger.info("✅ API key check passed")
    
    # Initialize RAG system
    try:
        logger.info("🤖 Initializing RAG system...")
        
        if 'retriever' not in st.session_state:
            logger.info("🆕 Creating new retriever instance...")
            with st.spinner("Loading RAG system..."):
                # Use absolute path to ensure we always get the correct FAISS index
                faiss_dir = Path(__file__).parent / "media_pipelines/scientific_publications/data/transformed_data/vector_embeddings"
                logger.info(f"🔍 FAISS directory path: {faiss_dir}")
                logger.info(f"🔍 FAISS directory exists: {faiss_dir.exists()}")
                
                if faiss_dir.exists():
                    logger.info(f"🔍 FAISS directory contents: {list(faiss_dir.glob('*'))}")
                
                st.session_state.retriever = RelevantChunksRetrieverFAISS(faiss_dir)
                logger.info("✅ Retriever instance created successfully")
        else:
            logger.info("✅ Retriever already exists in session state")
        
        # Set default top_k value
        if 'top_k' not in st.session_state:
            st.session_state['top_k'] = 10
            logger.info("🔧 Set default top_k to 10")
        
        # Show stats and update engram count
        logger.info("📊 Getting collection stats...")
        stats = st.session_state.retriever.get_collection_stats()
        logger.info(f"📊 Collection stats: {stats}")
        
        # Log current app state
        log_app_state()
        
        # if stats.get('total_chunks'):
        #     # Display engram count below the GIF
        #     col1, col2, col3 = st.columns([1, 1, 1])
        #     with col2:
        #         st.metric("🧠 Total Engrams", f"{stats['total_chunks']:,}")
        

        
        # Search parameters
        st.sidebar.markdown(f"""<h3><img src='{thinking_box}' alt="Thinking Box" style="width: 240px; height: 270px; border-radius: 10px;"></h3>""", unsafe_allow_html=True)
        st.sidebar.metric("Total Engrams Indexed", stats['total_chunks'])

        # top_k = st.sidebar.slider(
        #     "Number of results to retrieve",
        #     min_value=1,
        #     max_value=20,
        #     value=20,
        #     key="top_k"
        # )
        
        # Page selection
        # st.sidebar.header("📄 Pages")
        page = st.sidebar.radio(
            "Choose a page:",
            ["💬 Chat with Michael Levin", "🔍 Research Search"]
        )
        
        logger.info(f"📄 Selected page: {page}")
        
        st.markdown("---")

        if page == "💬 Chat with Michael Levin":
            logger.info("🚀 Loading conversational page...")
            conversational_page()
        elif page == "🔍 Research Search":
            logger.info("🚀 Loading research search page...")
            research_page()
        
        logger.info("✅ Main app initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        st.error(f"Failed to initialize RAG system: {e}")
        st.info("Make sure you've run the embedding tool first to create the FAISS index.")
        st.code("python media_pipelines/scientific_publications/tools/embed_semantic_chunks_faiss.py")

if __name__ == "__main__":
    main() 