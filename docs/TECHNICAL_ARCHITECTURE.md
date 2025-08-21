# Conversations Pipeline - Technical Architecture

## 🏗️ **System Architecture Overview**

The Conversations Pipeline is built as a **modular, step-based processing system** that transforms raw YouTube conversations into searchable, analyzable scientific content. The architecture follows a **sequential pipeline pattern** with **state persistence** and **intelligent fallback mechanisms**.

### **Core Design Principles**
1. **Modularity**: Each step is independent and can be run separately
2. **State Persistence**: Progress tracking across sessions and failures
3. **Graceful Degradation**: Fallback mechanisms when advanced features fail
4. **Data Consistency**: Structured outputs with comprehensive metadata
5. **User Interaction**: Intelligent prompts for complex decisions (speaker identification)

## 🔧 **Technical Stack & Dependencies**

### **Core Technologies**
- **Language**: Python 3.11+
- **Package Management**: `uv` (fast Python package manager)
- **Logging**: Custom logging system with automatic cleanup
- **Data Formats**: JSON for metadata, binary for media files

### **External Services**
- **AssemblyAI**: High-quality transcription with speaker diarization
- **OpenAI GPT-4**: Intelligent conversation analysis and content extraction
- **YouTube**: Video source via yt-dlp

### **Key Libraries**
```python
# Core Processing
openai==1.3.0          # GPT-4 API integration
assemblyai==0.21.0     # Transcription service
yt-dlp==2023.11.16     # YouTube video download

# Data Handling
pydantic==2.5.0        # Data validation (planned)
numpy==1.24.0          # Numerical operations
faiss-cpu==1.7.4       # Vector similarity search (planned)

# Media Processing
ffmpeg-python==0.2.0   # Video frame extraction (planned)
Pillow==10.0.0         # Image processing (planned)

# Utilities
python-dotenv==1.0.0   # Environment variable management
pathlib                 # Path handling (built-in)
logging                 # Logging system (built-in)
```

## 🗂️ **Data Architecture**

### **Data Flow Structure**
```
Raw Input (YouTube URLs)
    ↓
Step 1: Playlist Processing
    ↓
Video Metadata & Queue
    ↓
Step 2: Video Download
    ↓
Video Files + Enhanced Metadata
    ↓
Step 3: Transcription
    ↓
Transcripts + Speaker Diarization
    ↓
Step 4: LLM Analysis
    ↓
Semantic Chunks + Q&A Pairs
    ↓
Step 5: Frame Extraction (planned)
    ↓
Video Frames + Timing
    ↓
Step 6: Frame-Chunk Alignment (planned)
    ↓
Aligned Visual-Text Content
    ↓
Step 7: FAISS Embeddings (planned)
    ↓
Searchable Vector Database
    ↓
Step 8: Cleanup (planned)
    ↓
Optimized Storage + Search Interface
```

### **Data Persistence Strategy**
- **Progress Queue**: JSON-based state tracking
- **Processing Logs**: Timestamped log files with automatic cleanup
- **Output Files**: Structured JSON with comprehensive metadata
- **Media Files**: Organized directory structure with naming conventions

## 📁 **File Organization & Naming Conventions**

### **Directory Structure**
```
Conversations_and_working_meetings_1_on_1/
├── docs/                           # Documentation
├── logs/                           # Pipeline logs and progress tracking
├── step_01_raw/                    # Raw input (playlists, URLs)
├── step_02_extracted_playlist_content/  # Downloaded videos and metadata
├── step_03_transcription/          # AssemblyAI transcripts with speaker diarization
├── step_04_extract_chunks/        # LLM-based semantic chunking and Q&A extraction
├── step_05_frames/                 # Video frame extraction (planned)
├── step_06_frame_chunk_alignment/  # Frame-chunk alignment (planned)
├── step_07_faiss_embeddings/      # Vector embeddings and search (planned)
└── step_08_cleanup/               # Intelligent cleanup (planned)
```

### **File Naming Conventions**
- **Videos**: `video_{VIDEO_ID}/video.mp4`
- **Transcripts**: `{VIDEO_ID}_transcript.json`
- **Speakers**: `{VIDEO_ID}_speakers.json`
- **Chunks**: `{VIDEO_ID}_chunks.json`
- **Q&A Pairs**: `{VIDEO_ID}_qa_pairs.json`
- **Frames**: `{VIDEO_ID}_frame_{NUMBER}_{TIMESTAMP}s.jpg`

## 🔄 **Progress Tracking Architecture**

### **Progress Queue Structure**
```json
{
  "pipeline_progress": {
    "VIDEO_ID": {
      "video_title": {...},
      "playlist_ids": {...},
      "step_01_playlist_processing": "completed|pending|failed",
      "step_02_video_download": "completed|pending|failed",
      "step_03_transcription": "completed|pending|failed",
      "step_04_extract_chunks": "completed|pending|failed",
      "speaker_identification": "pending|completed",
      "qa_extraction": "pending|completed",
      "processing_metadata": {...}
    }
  },
  "playlist_progress": {...},
  "speaker_identification": {...},
  "qa_extraction": {...},
  "pipeline_info": {...}
}
```

### **State Management Features**
- **Atomic Updates**: Each step update is atomic
- **History Tracking**: Maintains change history for debugging
- **Metadata Storage**: Rich information about processing results
- **Error Capture**: Stores failure reasons and timestamps
- **Recovery Support**: Can resume from any step

## 🧠 **LLM Integration Architecture**

### **GPT-4 Implementation Details**
```python
class LLMConversationAnalyzer:
    def _get_llm_analysis(self, conversation_text: str, video_id: str) -> Optional[Dict[str, Any]]:
        # Model Configuration
        model = "gpt-4-1106-preview"
        temperature = 0.3  # Balanced creativity and consistency
        max_tokens = 4096  # Sufficient for detailed analysis
        response_format = {"type": "json_object"}  # Structured output
        
        # Prompt Engineering
        system_prompt = "Expert scientific conversation analysis..."
        user_prompt = f"Analyze conversation: {conversation_text}..."
        
        # Error Handling
        try:
            response = openai.chat.completions.create(...)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return None  # Triggers fallback processing
```

### **Analysis Capabilities**
1. **Semantic Chunking**: Identify coherent discussion segments
2. **Q&A Recognition**: Detect question-answer patterns with context
3. **Topic Classification**: Scientific area identification
4. **Conversation Flow**: Understand discussion structure and progression
5. **Context Awareness**: Maintain conversation continuity and relevance

### **Fallback Mechanisms**
```python
def _fallback_processing(self, transcript_data: Dict, speakers_data: Dict, video_id: str):
    """Rule-based processing when LLM analysis fails"""
    # Basic chunk extraction by speaker turns
    # Simple Q&A detection using keyword patterns
    # Reduced quality but guaranteed completion
```

## 🎤 **Speaker Diarization Architecture**

### **AssemblyAI Integration**
```python
class ConversationsVideoTranscriberWebhook:
    def __init__(self):
        self.assemblyai_url = "https://api.assemblyai.com/v2"
        self.webhook_url = "http://localhost:8000/webhook"
    
    def submit_transcription_request(self, video_path: str) -> str:
        transcript_request = {
            "audio_url": audio_url,
            "speaker_labels": True,        # Enable speaker diarization
            "speakers_expected": 2,        # Expect 2 speakers
            "sentiment_analysis": True,    # Analyze sentiment
            "iab_categories": True,        # Content categorization
            "content_safety": False,       # Disable for research content
            "language_detection": True,    # Detect language
            "boost_param": "high"          # High quality processing
        }
```

### **Speaker Identification Strategy**
1. **Automatic Detection**: AssemblyAI identifies Speaker A, Speaker B
2. **Interactive Mapping**: User identifies which speaker is Michael Levin
3. **Guest Researcher**: User provides name and affiliation
4. **Metadata Storage**: Comprehensive speaker information for future queries

## 📊 **Data Processing Architecture**

### **Chunk Extraction Strategy**
```python
class Chunk:
    chunk_id: str           # Unique identifier
    video_id: str           # Source video reference
    text: str               # Content text
    start_time: float       # Start timestamp
    end_time: float         # End timestamp
    speaker: str            # Speaker identifier
    speaker_name: str       # Human-readable name
    speaker_role: str       # Affiliation/role
    is_levin: bool          # Levin-specific flag
    chunk_type: str         # Discussion type
    topics: List[str]       # Scientific topics
    sentiment: str          # Emotional tone
    confidence: float       # Analysis confidence
    conversation_context: Dict  # Contextual information
```

### **Q&A Pair Structure**
```python
class QAPair:
    pair_id: str            # Unique identifier
    video_id: str           # Source video reference
    question: Dict          # Question details
    answer: Dict            # Answer details
    qa_type: str            # Interaction type
    topics: List[str]       # Scientific topics
    levin_involved: bool    # Levin participation flag
    conversation_flow: str  # Flow context
    confidence: float       # Analysis confidence
```

## 🔍 **Search & Retrieval Architecture (Planned)**

### **Embedding Strategy**
```python
class EmbeddingGenerator:
    def __init__(self):
        self.model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dim = 384
    
    def generate_chunk_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        # Combine text, topics, and context
        # Generate embeddings for semantic search
        # Return numpy array of embeddings
    
    def generate_qa_embeddings(self, qa_pairs: List[Dict]) -> np.ndarray:
        # Combine question + answer + context
        # Generate embeddings for Q&A search
        # Return numpy array of embeddings
```

### **FAISS Index Structure**
```python
class SearchInterface:
    def __init__(self):
        self.chunks_index = faiss.IndexFlatIP(384)  # Inner product similarity
        self.qa_index = faiss.IndexFlatIP(384)
        self.combined_index = faiss.IndexFlatIP(384)
    
    def search_chunks(self, query: str, top_k: int = 10) -> List[Dict]:
        # Generate query embedding
        # Search chunks index
        # Return ranked results
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        # Search both chunks and Q&A
        # Combine and rank results
        # Return unified results
```

## 🚨 **Error Handling Architecture**

### **Error Classification**
1. **Transient Errors**: Network issues, temporary API failures
2. **Data Errors**: Corrupted files, invalid formats
3. **User Errors**: Incorrect input, missing files
4. **System Errors**: Resource limitations, permission issues

### **Error Handling Strategy**
```python
class ErrorHandler:
    def handle_transient_error(self, error: Exception, retry_count: int) -> bool:
        # Implement exponential backoff
        # Retry with increasing delays
        # Return success/failure status
    
    def handle_data_error(self, error: Exception, data: Dict) -> Dict:
        # Attempt data recovery
        # Provide fallback data
        # Log error details
    
    def handle_user_error(self, error: Exception) -> str:
        # Provide helpful error messages
        # Suggest solutions
        # Guide user to resolution
```

### **Recovery Mechanisms**
- **Automatic Retry**: Transient errors with exponential backoff
- **Graceful Degradation**: Fallback to simpler processing methods
- **State Preservation**: Maintain progress across failures
- **User Notification**: Clear error messages and resolution steps

## 📈 **Performance Architecture**

### **Processing Pipeline**
- **Parallel Processing**: Independent steps can run concurrently
- **Batch Processing**: Multiple videos processed in sequence
- **Resource Management**: Controlled memory and CPU usage
- **Progress Monitoring**: Real-time status updates

### **Optimization Strategies**
- **Lazy Loading**: Load data only when needed
- **Caching**: Store frequently accessed data
- **Compression**: Reduce storage requirements
- **Cleanup**: Remove intermediate files automatically

## 🔒 **Security & Privacy Architecture**

### **Data Protection**
- **API Key Management**: Environment variables for sensitive data
- **Local Processing**: No data sent to external services unnecessarily
- **Secure Storage**: Local file system with controlled access
- **Audit Logging**: Track all data access and modifications

### **Privacy Considerations**
- **Research Content**: Focus on scientific discussions
- **Public Videos**: Only process publicly available content
- **Data Retention**: Configurable cleanup policies
- **Access Control**: Local system access only

## 🚀 **Scalability Architecture**

### **Horizontal Scaling**
- **Modular Design**: Each step can be distributed
- **Independent Processing**: Steps don't depend on each other
- **Queue Management**: Can process multiple videos simultaneously
- **Resource Isolation**: Each step has its own resources

### **Vertical Scaling**
- **Memory Optimization**: Efficient data structures
- **CPU Utilization**: Multi-threaded processing where possible
- **Storage Efficiency**: Compressed and optimized file formats
- **Network Optimization**: Efficient API usage and caching

## 📚 **Documentation Architecture**

### **Documentation Structure**
- **`README.md`**: Quick start and overview
- **`PIPELINE_OVERVIEW.md`**: High-level pipeline description
- **`IMPLEMENTATION_PLAN.md`**: Development roadmap and tasks
- **`TECHNICAL_ARCHITECTURE.md`**: This technical deep-dive
- **`PIPELINE_ARCHITECTURE.md`**: Step-by-step implementation details
- **`QUICK_REFERENCE.md`**: Commands and troubleshooting

### **Code Documentation**
- **Docstrings**: Comprehensive function documentation
- **Type Hints**: Python type annotations for clarity
- **Inline Comments**: Complex logic explanations
- **Examples**: Usage examples and sample outputs

---

**Status**: Technical architecture documented for Steps 1-4
**Next Phase**: Extend architecture for Steps 5-8
**Focus**: Maintain architectural consistency while adding new capabilities
**Goal**: Scalable, maintainable, and extensible pipeline architecture
