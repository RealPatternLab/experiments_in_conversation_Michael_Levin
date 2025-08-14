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
    
    # Find all citation patterns like [Source_1], [Source_2], etc.
    citation_pattern = r'\[Source_(\d+)\]'
    
    def replace_citation(match):
        source_num = int(match.group(1))
        source_key = f"Source_{source_num}"
        
        if source_key in source_mapping:
            source_info = source_mapping[source_key]
            title = source_info.get('title', 'Unknown')
            year = source_info.get('publication_date', 'Unknown')
            
            # PDF citation with GitHub raw URL
            pdf_filename = source_info.get('sanitized_filename')
            doi = source_info.get('doi', '')
            
            # Create clickable hyperlinks
            if pdf_filename and pdf_filename != "Unknown":
                github_raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/SCIENTIFIC_PUBLICATION_PIPELINE/step_07_archive/{pdf_filename}"
                
                # Create clickable PDF link
                pdf_link = f"<a href='{github_raw_url}' target='_blank' style='color: #0066cc; text-decoration: underline;' title='Download PDF: {pdf_filename}'>[PDF]</a>"
                
                # Create clickable DOI link if available
                doi_link = ""
                if doi and doi != "Unknown":
                    doi_link = f" <a href='https://doi.org/{doi}' target='_blank' style='color: #0066cc; text-decoration: underline;' title='View on DOI.org'>[DOI]</a>"
                
                return f"<sup>{pdf_link}{doi_link}</sup>"
            else:
                # Fallback: show title and year
                return f"<sup>[{title} ({year})]</sup>"
        else:
            return match.group(0)  # Return original if source not found
    
    # Replace citations with hyperlinks
    processed_text = re.sub(citation_pattern, replace_citation, response_text)
    
    return processed_text

def get_conversational_response(query: str, rag_results: list, conversation_history: list = None) -> str:
    """Generate a conversational response using RAG results with inline citations."""
    try:
        # Prepare context from RAG results
        context_parts = []
        source_mapping = {}  # Map source numbers to actual source info
        
        for i, chunk in enumerate(rag_results[:3]):  # Use top 3 results
            source_key = f"Source_{i+1}"
            
            # Format authors properly
            authors = chunk.get('authors', [])
            if isinstance(authors, list) and authors:
                authors_str = ', '.join(authors[:3])  # Show first 3 authors
                if len(authors) > 3:
                    authors_str += f" et al."
            else:
                authors_str = "Unknown"
            
            context_parts.append(f"{source_key} ({chunk.get('title', 'Unknown')}, {chunk.get('publication_year', 'Unknown')}): {chunk.get('text', '')}")
            
            # Enhanced source mapping
            source_mapping[source_key] = {
                'title': chunk.get('title', 'Unknown'),
                'authors': chunk.get('authors', []),
                'journal': chunk.get('journal', 'Unknown'),
                'doi': chunk.get('doi', 'Unknown'),
                'publication_date': chunk.get('publication_year', 'Unknown'),
                'text': chunk.get('text', ''),
                'sanitized_filename': chunk.get('pdf_filename'),  # Use pdf_filename from FAISS metadata
                'rank': i + 1,
                'section': chunk.get('section', 'Unknown'),
                'topic': chunk.get('topic', 'Unknown')
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
        prompt = f"""You are Michael Levin, a developmental and synthetic biologist at Tufts University. Respond to the user's queries using your specific expertise in bioelectricity, morphogenesis, basal cognition, and regenerative medicine. Ground your responses in the provided context from my published work (if provided). 

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
            
            logger.info(f"Using embeddings from: {timestamp}")
            
            # Return top_k chunks with mock similarity scores
            # In a production system, you'd implement actual FAISS similarity search here
            all_results = []
            for i, chunk_meta in enumerate(metadata[:top_k]):
                chunk_meta['similarity_score'] = 0.9 - (i * 0.1)  # Mock scores
                chunk_meta['embedding_timestamp'] = timestamp  # Add timestamp info
                all_results.append(chunk_meta)
            
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

def conversational_page():
    """Conversational interface page."""
    st.header("💬 Chat with Michael Levin")
    st.markdown("Have a conversation with Michael Levin about his research. He'll answer your questions based on his scientific publications.")
    
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
                            
                            # Show sources used
                            with st.expander("📚 Sources used"):
                                # Track unique sources to avoid duplicates
                                seen_sources = set()
                                source_counter = 1
                                
                                for i, result in enumerate(filtered_results[:3]):
                                    source_title = result.get('title', 'Unknown')
                                    year = result.get('publication_year', 'Unknown')
                                    section_header = result.get('section', 'Unknown')
                                    similarity = result.get('similarity_score', 0)
                                    
                                    # Create a unique identifier for this source
                                    source_id = f"{source_title}_{year}_{section_header}"
                                    
                                    # Only show the source if it hasn't been listed before
                                    if source_id not in seen_sources:
                                        st.markdown(f"**{source_counter}.** {source_title} ({year}) - Section: {section_header} (Similarity: {similarity:.3f})")
                                        seen_sources.add(source_id)
                                        source_counter += 1
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
        layout="wide"
    )
    thinking_box = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/thinking_box.gif"

    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stMarkdown {
        color: #262730;
    }
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin: 8px 0;
    }
    .stExpander {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🧠 Michael Levin Research Assistant</h1>
        <p style="font-size: 1.2rem; color: #666;">
            Explore Michael Levin's research on developmental biology, collective intelligence, and bioelectricity
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
            with st.spinner("Loading RAG system..."):
                faiss_dir = Path("SCIENTIFIC_PUBLICATION_PIPELINE/step_06_faiss_embeddings")
                if faiss_dir.exists():
                    st.session_state.retriever = FAISSRetriever(faiss_dir)
                else:
                    st.error("FAISS embeddings directory not found!")
                    st.info("Please run the pipeline first to generate embeddings.")
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
                background: transparent;
                mix-blend-mode: multiply;
            ">
        </div>
        """, unsafe_allow_html=True)
        
        # Display embedding information
        st.sidebar.metric("Total Engrams Indexed", stats['total_chunks'])        
        
        # Search parameters
        st.sidebar.header("🔍 Search Settings")
        top_k = st.sidebar.slider(
            "Number of results to retrieve",
            min_value=1,
            max_value=20,
            value=10,
            key="top_k"
        )
        
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