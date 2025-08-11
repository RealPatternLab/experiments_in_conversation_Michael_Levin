# Michael Levin QA Engine - Refactored Architecture

This repository has been completely refactored to provide a clean, organized structure for creating digital twins of scientific public figures through multiple specialized media pipelines.

## 🎯 Project Goal

Create comprehensive digital twins of specific scientific public figures by processing and organizing:
- Scientific publications and research papers
- YouTube presentations and lectures
- Podcast conversations and interviews
- Other media types as needed

Each media type is processed into:
- **Semantic chunks** for RAG (Retrieval-Augmented Generation) applications
- **Q&A pairs** for fine-tuning language models

## 🏗️ New Architecture

### Archive
- **Location**: `archive/`
- **Purpose**: Contains all previous code and implementations
- **Use**: Reference for existing functionality and gradual migration

### Media Pipelines
- **Location**: `media_pipelines/`
- **Structure**: Each media type has its own complete pipeline
- **Benefits**: Independent development, specialized tools, clear data flow

#### Current Pipelines
1. **Scientific Publications** - PDF processing and academic content
2. **YouTube Videos - Formal Solo Presentations** - Lectures and academic talks
3. **YouTube Videos - Podcasts (1-on-0)** - Solo podcast episodes
4. **YouTube Videos - Podcasts (1-on-1)** - Two-person conversations
5. **YouTube Videos - Podcasts (1-on-2)** - Three-person discussions
6. **YouTube Videos - Podcasts (1-on-3)** - Four-person conversations
7. **YouTube Videos - Podcasts (1-on-4)** - Five-person panel discussions
8. **Raw Ingestion** - Common ingestion layer for all media types

### Global Tools
- **Location**: `global_tools/`
- **Purpose**: Tools and utilities applicable to all media types
- **Examples**: Database utilities, embedding functions, common data structures

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required dependencies (see individual pipeline requirements)
- API keys for external services (OpenAI, AssemblyAI, etc.)

### Development Workflow
1. **Choose a pipeline** based on your media type
2. **Review the pipeline README** for specific requirements
3. **Implement pipeline-specific tools** in the `tools/` subdirectory
4. **Add global utilities** to `global_tools/` if applicable
5. **Test independently** without affecting other pipelines

## 📁 Directory Structure

```
michael-levin-qa-engine-1/
├── archive/                    # All previous code and implementations
├── media_pipelines/            # Specialized media processing pipelines
│   ├── scientific_publications/
│   │   ├── pipeline/          # Main processing scripts
│   │   ├── tools/             # Pipeline-specific tools
│   │   └── data/
│   │       ├── source_data/   # Raw input files
│   │       └── transformed_data/ # Processed outputs
│   ├── youtube_videos_*/      # Various YouTube video types
│   └── raw_ingestion/         # Common ingestion layer
├── global_tools/               # Tools for all media types
├── .git/                       # Version control
└── .gitignore                  # Git ignore rules
```

## 🔧 Pipeline Development

Each pipeline follows a consistent structure:
- **`pipeline/`**: Main orchestration and processing scripts
- **`tools/`**: Specialized utilities for that media type
- **`data/source_data/`**: Raw input files and preprocessed data
- **`data/transformed_data/`**: Final outputs (chunks, embeddings, etc.)

## 📚 Documentation

- **Pipeline Overview**: `media_pipelines/README.md`
- **Individual Pipeline Docs**: Each pipeline has its own README
- **Archive Documentation**: `archive/` contains all previous documentation

## 🤝 Contributing

1. **Choose a pipeline** to work on
2. **Follow the established structure** for that pipeline
3. **Keep pipelines independent** - avoid cross-pipeline dependencies
4. **Document your changes** in the appropriate README files
5. **Test thoroughly** before committing

## 🔄 Migration from Archive

The `archive/` directory contains all previous functionality. To migrate specific features:

1. **Identify the feature** in the archive
2. **Determine the appropriate pipeline** for that media type
3. **Refactor the code** to fit the new pipeline structure
4. **Update dependencies** to use global tools where appropriate
5. **Test thoroughly** in the new structure
6. **Remove from archive** once migration is complete

## 📈 Benefits of New Architecture

- **Clean Separation**: Each media type has its own pipeline
- **Independent Development**: Teams can work on different pipelines
- **Specialized Tools**: Media-specific processing without conflicts
- **Clear Data Flow**: Source and transformed data are separated
- **Easy Testing**: Each pipeline can be tested independently
- **Scalability**: New media types can be added as new pipelines
- **Maintainability**: Clear structure makes code easier to understand and modify

## 🚧 Status

This is a **work in progress**. The archive contains all previous functionality, and pipelines are being developed incrementally. Each pipeline will be fully functional before moving to the next.

## 📞 Support

For questions about the new architecture or help with pipeline development, please refer to the individual pipeline documentation or create an issue in this repository. 