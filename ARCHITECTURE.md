# Michael Levin Scientific Knowledge Base - Multi-Pipeline Architecture

## System Overview

The Michael Levin Scientific Knowledge Base is a sophisticated multi-modal RAG system that processes scientific publications (PDFs), scientific videos, and conversations/working meetings into an interactive knowledge base. The system uses separate processing pipelines for different content types, with plans to evolve toward unified search capabilities while maintaining separation of concerns.

**🔧 Data Standardization**: The system now implements shared data models and base classes to ensure consistency across all pipelines, reducing troubleshooting and improving maintainability.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-PIPELINE SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────────┐ │
│  │  SCIENTIFIC PUBLICATIONS│    │        SCIENTIFIC VIDEOS                │ │
│  │      PIPELINE           │    │         PIPELINE                        │ │
│  │                         │    │                                         │ │
│  │  ┌─────────────────┐    │    │  ┌─────────────────┐                   │ │
│  │  │   PDF Input     │    │    │  │  Video Input    │                   │ │
│  │  │ (step_01_raw)   │    │    │  │(YouTube URLs)   │                   │ │
│  │  └─────────────────┘    │    │  └─────────────────┘                   │ │
│  │           │              │    │           │                            │ │
│  │           ▼              │    │           ▼                            │ │
│  │  ┌─────────────────┐    │    │  ┌─────────────────┐                   │ │
│  │  │ 7-Step Pipeline │    │    │  │ 8-Step Pipeline │                   │ │
│  │  │ (Publications)  │    │    │  │   (Videos)      │                   │ │
│  │  └─────────────────┘    │    │  └─────────────────┘                   │ │
│  │           │              │    │           │                            │ │
│  │           ▼              │    │           ▼                            │ │
│  │  ┌─────────────────┐    │    │  ┌─────────────────┐                   │ │
│  │  │ FAISS Index     │    │    │  │ FAISS Index     │                   │ │
│  │  │ (3072 dim)      │    │    │  │ (3072 dim)      │                   │ │
│  │  └─────────────────┘    │    │  └─────────────────┘                   │ │
│  └─────────────────────────┘    └─────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    CONVERSATIONS PIPELINE                              │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐                                                   │ │
│  │  │  Video Input    │                                                   │ │
│  │  │(YouTube URLs)   │                                                   │ │
│  │  └─────────────────┘                                                   │ │
│  │           │                                                             │ │
│  │           ▼                                                             │ │
│  │  ┌─────────────────┐                                                   │ │
│  │  │ 8-Step Pipeline │                                                   │ │
│  │  │(Conversations)  │                                                   │ │
│  │  │                 │                                                   │ │
│  │  └─────────────────┘                                                   │ │
│  │           │                                                             │ │
│  │           ▼                                                             │ │
│  │  ┌─────────────────┐                                                   │ │
│  │  │ FAISS Index     │                                                   │ │
│  │  │ (3072 dim)      │                                                   │ │
│  │  └─────────────────┘                                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    UNIFIED SEARCH LAYER (Phase 1)                      │ │
│  │  • Query all three pipelines simultaneously                             │ │
│  │  • Fuse and rank results                                                │ │
│  │  • Provide unified user experience                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    MERGE PIPELINE (Phase 2)                            │ │
│  │  • Periodic CRON job merging                                           │ │
│  │  • Unified FAISS index                                                 │ │
│  │  • Cross-modal relationships                                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                  REAL-TIME SYNC (Phase 3)                              │ │
│  │  • File system watching                                                 │ │
│  │  • Incremental updates                                                  │ │
│  │  • Always up-to-date unified index                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Architecture

### 1. Scientific Publications Pipeline

**Location**: `/SCIENTIFIC_PUBLICATION_PIPELINE/`

The publications pipeline processes PDFs through 7 sequential steps:

```
Step 01: Hash Validation (step_01_unique_hashcode_validator.py)
    ↓
Step 02: Metadata Extraction (step_02_metadata_extractor.py)
    ↓
Step 03: Text Extraction (step_03_text_extractor.py)
    ↓
Step 04: Metadata Enrichment (step_04_optional_metadata_enrichment.py)
    ↓
Step 05: Semantic Chunking (step_05_semantic_chunker_split.py)
    ↓
Step 06: Consolidated Embedding (step_06_consolidated_embedding.py)
    ↓
Step 07: Archive Management (step_07_archive.py)
```

**Key Features**:
- **Input**: Scientific PDFs
- **Processing**: Text extraction, semantic chunking
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Output**: FAISS index + metadata for publications

### 2. Scientific Videos Pipeline

**Location**: `/SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/`

The videos pipeline processes YouTube content through 8 sequential steps:

```
Step 01: Playlist Processing (step_01_playlist_processor.py)
    ↓
Step 02: Video Download & Metadata (step_02_video_downloader.py)
    ↓
Step 03: Enhanced Transcription (step_03_transcription.py)
    ↓
Step 04: Semantic Chunking (step_04_extract_chunks.py)
    ↓
Step 05: Frame Extraction (step_05_frame_extractor.py)
    ↓
Step 06: Frame-Chunk Alignment (step_06_frame_chunk_alignment.py)
    ↓
Step 07: Consolidated Embedding (step_07_consolidated_embedding.py)
    ↓
Step 08: Archive Management (step_08_archive.py)
```

**Key Features**:
- **Input**: YouTube videos (formal presentations)
- **Processing**: Transcription, frame extraction, multimodal alignment
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions) + visual features (64 dimensions)
- **Output**: FAISS indices + metadata for videos

### 3. Conversations Pipeline

**Location**: `/SCIENTIFIC_VIDEO_PIPELINE/Conversations_and_working_meetings_1_on_1/`

The conversations pipeline processes YouTube working meetings and conversations through 8 sequential steps:

```
Step 01: Playlist Processing (step_01_playlist_processor.py)
    ↓
Step 02: Video Download & Metadata (step_02_video_downloader.py)
    ↓
Step 03: Enhanced Transcription (step_03_transcription_webhook.py)
    ↓
Step 04: Semantic Chunking (step_04_extract_chunks.py)
    ↓
Step 05: Frame Extraction (step_05_frame_extractor.py)
    ↓
Step 06: Frame-Chunk Alignment (step_06_frame_chunk_alignment.py)
    ↓
Step 07: Consolidated Embedding (step_07_consolidated_embedding.py)
    ↓
Step 08: Archive Management (step_08_cleanup.py)
```

**Key Features**:
- **Input**: YouTube videos (conversations, working meetings)
- **Processing**: Speaker diarization, Q&A extraction, Levin's insights identification
- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions) + visual features
- **Output**: FAISS indices + metadata for conversations with timestamps

## Evolution Strategy

### Design Philosophy

The system is designed with **separation of concerns** and **evolutionary architecture** in mind:

1. **Independent Development**: Each pipeline can evolve independently
2. **Version Control**: Separate pipelines allow for better version control
3. **Experimental Freedom**: Teams can experiment without affecting other pipelines
4. **Gradual Unification**: Move toward unified search as the system becomes robust

### Phase 1: Unified Search Layer (Current Focus)

**Goal**: Keep pipelines separate but provide unified user experience

**Implementation**:
```python
class UnifiedRAGSearch:
    def search(self, query: str):
        # Search all three pipelines simultaneously
        pub_results = self.search_publications(query)
        video_results = self.search_videos(query)
        conversation_results = self.search_conversations(query)
        
        # Fuse and rank results
        return self.fuse_results(pub_results, video_results, conversation_results)
```

**Benefits**:
- ✅ Immediate improvement in user experience
- ✅ No additional storage requirements
- ✅ Low risk implementation
- ✅ Maintains pipeline independence

**Trade-offs**:
- ❌ Slower queries (search 2 DBs)
- ❌ No cross-modal learning

### Phase 2: Periodic Merge Pipeline (Medium-term)

**Goal**: Create unified FAISS index while keeping pipelines separate

**Implementation**:
```python
class KnowledgeBaseMerger:
    def merge_indices(self):
        # Load existing indices (NO re-embedding)
        pub_index = faiss.read_index("publications.faiss")
        video_index = faiss.read_index("videos.faiss")
        conversation_index = faiss.read_index("conversations.faiss")
        
        # Concatenate existing vectors
        unified_vectors = np.vstack([pub_vectors, video_vectors, conversation_vectors])
        
        # Create unified index
        unified_index = faiss.IndexFlatL2(unified_vectors.shape[1])
        unified_index.add(unified_vectors)
```

**Benefits**:
- ✅ Faster queries (single DB)
- ✅ Cross-modal relationships
- ✅ Better ranking algorithms
- ✅ Maintains pipeline independence

**Trade-offs**:
- ❌ Additional storage (3x FAISS files)
- ❌ Slight delay in updates (cron job frequency)

**CRON Schedule**:
```bash
# Daily merge at 2 AM
0 2 * * * cd /path/to/merge_pipeline && python merge_indices.py

# Weekly full rebuild
0 2 * * 0 cd /path/to/merge_pipeline && python full_rebuild.py
```

### Phase 3: Real-Time Sync (Long-term)

**Goal**: Always up-to-date unified index with incremental updates

**Implementation**:
```python
class RealTimeSync:
    def on_new_content(self, content_path):
        # Process new content
        new_embedding = self.process_content(content_path)
        
        # Add to unified index incrementally
        self.unified_index.add(new_embedding.reshape(1, -1))
        
        # Save updated index
        faiss.write_index(self.unified_index, "unified_index.faiss")
```

**Benefits**:
- ✅ Always up-to-date
- ✅ Fast queries
- ✅ Cross-modal learning
- ✅ Real-time updates

**Trade-offs**:
- ❌ Complex implementation
- ❌ More error-prone
- ❌ Higher maintenance overhead

## Current Implementation Status

### ✅ Completed

- **Publications Pipeline**: Full 7-step pipeline operational
- **Videos Pipeline**: Full 8-step pipeline operational
- **Embedding Consistency**: Both pipelines use OpenAI text-embedding-3-large
- **FAISS Indices**: Separate indices for publications and videos

### 🔄 In Progress

- **Phase 1**: Unified search layer development
- **Cross-pipeline testing**: Ensuring compatibility

### 📋 Planned

- **Phase 2**: Periodic merge pipeline (next month)
- **Phase 3**: Real-time sync (long-term)
- **Advanced ranking**: Cross-modal result fusion
- **Knowledge graph**: Cross-references between content types

## Technical Implementation Details

### Embedding Strategy

**Unified Model**: Both pipelines use OpenAI's `text-embedding-3-large` (3072 dimensions)

**Benefits**:
- Consistent semantic space across content types
- High-quality embeddings for scientific content
- Compatible FAISS indices for future merging

### Data Flow

```
Raw Content → Pipeline Processing → Embeddings → FAISS Index → Search Interface
     ↓              ↓                ↓           ↓            ↓
Publications   7-Step Pipeline  3072-dim     pub_index   Unified Search
Videos        8-Step Pipeline   3072-dim     video_index  (Phase 1)
```

### File Structure

```
SCIENTIFIC_PUBLICATION_PIPELINE/
├── step_06_faiss_embeddings/
│   └── publications.faiss
└── [pipeline files]

SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/
├── step_07_faiss_embeddings/
│   └── videos.faiss
└── [pipeline files]

MERGE_PIPELINE/ (Phase 2)
├── merge_indices.py
├── unified_index.faiss
└── unified_metadata.pkl
```

## Frontend Integration

### Current State

- **Separate search interfaces** for publications and videos
- **Independent result displays**
- **No cross-modal search**

### Phase 1 Implementation

```typescript
// Single unified search component
const UnifiedSearch = () => {
  const search = async (query: string) => {
    const results = await fetch('/api/unified-search', {
      method: 'POST',
      body: JSON.stringify({ query })
    });
    
    return results.json(); // Combined publications + videos
  };
  
  return <SearchInterface onSearch={search} />;
};
```

### Future Enhancements

- **Tabbed interface**: All content, Publications only, Videos only
- **Cross-modal results**: Show related publications when searching videos
- **Unified ranking**: Intelligent result ordering across content types

## Maintenance and Operations

### Pipeline Independence

- **Separate processing cycles** for different content types
- **Independent scaling** - can process one without affecting the other
- **Fault isolation** - issues in one pipeline don't break the other

### Monitoring

- **Pipeline health checks** for each pipeline independently
- **Embedding quality metrics** for both content types
- **Search performance monitoring** across unified and separate indices

### Backup Strategy

- **Separate backups** for each pipeline
- **Unified index backups** (Phase 2+)
- **Metadata versioning** for rollback capabilities

## Data Standardization System

### Overview

The system now implements a comprehensive data standardization approach using shared data models and base classes to ensure consistency across all pipelines.

### Shared Data Models

**Location**: `utils/pipeline_data_models.py`

**Components**:
- **BaseRAGEntry**: Common structure for all content types
- **PipelineType**: Enumeration for different pipeline types
- **Standardized Fields**: Consistent metadata, visual content, and temporal information
- **Validation Functions**: Automatic data structure validation

### Base Classes

**Location**: `utils/base_frame_chunk_aligner.py`

**Purpose**: Provide common functionality while allowing pipeline-specific customization

**Benefits**:
- **Code Reuse**: Common functionality centralized
- **Consistency**: Standardized output formats
- **Maintainability**: Single point for updates
- **Validation**: Built-in data structure validation

### Pipeline Integration

**Video Pipelines**:
- **Formal Presentations**: Ready for refactoring to use base classes
- **Conversations**: Ready for refactoring to use base classes
- **Future Pipelines**: Easy to add using inheritance pattern

**Publications Pipeline**:
- **Data Alignment**: Follows shared data standards
- **Metadata Consistency**: Uses common naming conventions
- **Output Compatibility**: Maintains consistency with video pipelines

### Migration Path

1. **Phase 1**: ✅ Create shared models and base classes
2. **Phase 2**: Refactor existing pipelines to use base classes
3. **Phase 3**: Implement data validation across all pipelines
4. **Phase 4**: Add new pipeline types using standardized approach

### Benefits

- **Reduced Troubleshooting**: Consistent data structures across pipelines
- **Easier Maintenance**: Common code centralized
- **Future-Proof**: Easy to add new pipeline types
- **Quality Assurance**: Automatic validation catches issues early

## Conclusion

This multi-pipeline architecture provides the best of both worlds:

1. **Immediate benefits** through pipeline independence and experimentation
2. **Clear evolution path** toward unified search capabilities
3. **Risk mitigation** through gradual integration
4. **Scalability** for adding new content types in the future
5. **Data consistency** through shared models and base classes

The system is designed to evolve naturally as requirements become clearer and the technology becomes more robust, while maintaining the flexibility to experiment and iterate on each pipeline independently. The new standardization system ensures that all pipelines generate consistent, validated outputs that work seamlessly in the unified RAG system.
