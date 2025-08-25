# Unified Video Pipeline Framework

This directory contains the unified framework for scientific video processing pipelines, designed to eliminate code duplication and ensure consistency across different pipeline types.

## 🏗️ Architecture Overview

The unified framework consists of:

- **`base_pipeline.py`** - Abstract base class that all pipelines inherit from
- **`video_pipeline_scripts/`** - Unified step implementations (steps 1-8)
- **`pipeline_configs/`** - YAML configuration files for each pipeline type
- **`config_loader.py`** - Utility for loading and managing pipeline configurations

## 🎯 Benefits

### **Before (Duplicated Code)**
- 4 separate pipelines with duplicated steps 1-8
- Maintenance nightmare - bug fixes needed in 4 places
- Feature drift - some pipelines got updates, others didn't
- Inconsistent behavior across pipelines
- Testing complexity - same logic tested 4 times

### **After (Unified Framework)**
- Single source of truth for all pipeline logic
- Easy maintenance - fix once, applies everywhere
- Consistent behavior across all pipelines
- Easy testing - test core logic once
- Feature parity - new features automatically available to all pipelines
- Configuration-driven differences

## 📁 Directory Structure

```
core/
├── base_pipeline.py                    # Abstract base class
├── config_loader.py                    # Configuration management
├── video_pipeline_scripts/             # Unified step implementations
│   ├── step_01_playlist_processor.py  # ✅ Playlist processing
│   ├── step_02_video_downloader.py    # ✅ Video downloading with fallbacks
│   ├── step_03_transcription.py       # ✅ Transcription handling
│   ├── step_04_extract_chunks.py      # ✅ LLM-enhanced chunking
│   ├── step_05_frame_extractor.py     # ✅ Frame extraction
│   ├── step_06_frame_chunk_alignment.py # ✅ Frame-chunk alignment
│   ├── step_07_faiss_embeddings.py   # ✅ FAISS embedding generation
│   └── step_08_cleanup.py            # ✅ Pipeline cleanup
├── pipeline_configs/                   # Pipeline-specific configurations
│   ├── formal_presentations.yaml       # ✅ Single speaker, formal content
│   ├── conversations_1_on_1.yaml      # ✅ Two speakers, conversational
│   └── conversations_1_on_2.yaml      # ✅ Three speakers, conversational
└── README.md                           # This file
```

## 🔧 Pipeline Types

### **1. Formal Presentations (`formal_presentations`)**
- **Speaker Count**: 1 (Michael Levin)
- **Content Type**: Formal scientific presentations
- **Best Features**: 
  - Progress queue integration
  - Centralized logging
  - Robust error handling
  - Aggressive cleanup strategy
  - LLM enhancement for topics

### **2. Conversations 1-on-1 (`conversations_1_on_1`)**
- **Speaker Count**: 2 (Levin + 1 other)
- **Content Type**: Conversational scientific discussions
- **Best Features**:
  - Multi-speaker support
  - Speaker identification and mapping
  - Conversation flow analysis
  - Q&A pair generation

### **3. Conversations 1-on-2 (`conversations_1_on_2`)**
- **Speaker Count**: 3 (Levin + 2 others)
- **Content Type**: Multi-speaker scientific conversations
- **Best Features**:
  - Advanced multi-speaker dynamics
  - LLM-powered content summaries
  - Enhanced topic extraction
  - Single-frame alignment strategy
  - Comprehensive Q&A generation

## 🚀 Usage

### **Loading a Pipeline Configuration**

```python
from core.config_loader import load_pipeline_config

# Load configuration for conversations 1-on-2
config = load_pipeline_config('conversations_1_on_2')

# Use the configuration
pipeline_type = config['pipeline_type']
speaker_count = config['speaker_count']
llm_enabled = config['llm_enhancement']
```

### **Using Unified Step Scripts**

```python
from core.video_pipeline_scripts.step_04_extract_chunks import UnifiedChunkExtractor

# Initialize with pipeline configuration
extractor = UnifiedChunkExtractor(config, progress_queue)

# Process all transcripts
extractor.process_all_transcripts()
```

### **Creating Custom Pipeline Types**

```python
from core.config_loader import PipelineConfigLoader

loader = PipelineConfigLoader()

# Create custom configuration
custom_config = loader.create_custom_config(
    pipeline_type='custom_pipeline',
    speaker_count=4,
    llm_enhancement=True,
    chunking_strategy='custom'
)

# Save configuration
loader.save_config('custom_pipeline', custom_config)
```

## 🔄 Migration Path

### **Phase 1: Core Framework (Complete)**
- ✅ Base pipeline class
- ✅ Configuration system
- ✅ All unified step implementations (1-8)
- ✅ Configuration files for all pipeline types

### **Phase 2: Testing & Migration**
- 🔄 Test unified system with real data
- 🔄 Convert existing pipelines to use unified scripts
- 🔄 Remove duplicated code

### **Phase 3: Pipeline Wrappers**
- 🔄 Convert existing pipelines to use unified scripts
- 🔄 Test unified system with real data
- 🔄 Remove duplicated code

### **Phase 4: Advanced Features**
- 🔄 Interactive speaker identification
- 🔄 Advanced Q&A generation
- 🔄 Multi-pipeline coordination

## 🧪 Testing the Framework

### **Test Configuration Loading**

```bash
cd SCIENTIFIC_VIDEO_PIPELINE/core
python config_loader.py
```

### **Test Unified Step 4**

```bash
cd SCIENTIFIC_VIDEO_PIPELINE
python -c "
from core.config_loader import load_pipeline_config
from core.video_pipeline_scripts.step_04_extract_chunks import UnifiedChunkExtractor

# Load config
config = load_pipeline_config('conversations_1_on_2')
print(f'Loaded config for: {config[\"pipeline_name\"]}')

# Test initialization
extractor = UnifiedChunkExtractor(config)
print('✅ Unified extractor initialized successfully')
"
```

## 📊 Configuration Comparison

| Feature | Formal Presentations | Conversations 1-on-1 | Conversations 1-on-2 |
|---------|---------------------|----------------------|----------------------|
| **Speaker Count** | 1 | 2 | 3 |
| **LLM Enhancement** | ✅ | ✅ | ✅ |
| **Q&A Generation** | ❌ | ✅ | ✅ |
| **Frame Strategy** | Single frame | Single frame | Single frame |
| **Cleanup Strategy** | Aggressive | Aggressive | Aggressive |
| **Progress Tracking** | ✅ | ✅ | ✅ |

## 🎉 Next Steps

1. **Test the unified framework** with existing data
2. **Implement remaining unified steps** (1-3, 5-8)
3. **Migrate existing pipelines** to use unified scripts
4. **Add advanced features** like interactive speaker identification
5. **Create pipeline coordination tools** for multi-pipeline workflows

## 🤝 Contributing

When adding new features:

1. **Add to base class** if it's common to all pipelines
2. **Add to configuration** if it's pipeline-specific
3. **Add to unified scripts** if it's step-specific
4. **Update this README** to document changes

## 📝 Notes

- **Backward Compatibility**: Existing pipelines continue to work during migration
- **Configuration Validation**: All configurations are validated for required fields
- **Error Handling**: Robust error handling with fallbacks for missing components
- **Logging**: Centralized logging with pipeline-specific contexts
- **Progress Tracking**: Consistent progress tracking across all pipeline types

---

*This unified framework represents a significant improvement in maintainability and consistency for the scientific video processing pipeline system.*
