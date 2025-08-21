# Scientific Publication Pipeline

This pipeline processes scientific publications and research papers, extracting semantic content and creating RAG-ready outputs for the unified search system.

## 🏗️ **Pipeline Architecture**

This pipeline follows a specialized workflow optimized for text-based scientific content:

1. **Step 1**: PDF processing and metadata extraction
2. **Step 2**: Text extraction and cleaning
3. **Step 3**: Semantic chunking and content analysis
4. **Step 4**: Content enhancement and metadata enrichment
5. **Step 5**: FAISS embedding generation

## 🔧 **Data Standardization**

**IMPORTANT**: This pipeline is part of a larger effort to standardize data structures across all pipelines.

### Shared Data Models
- **Location**: `utils/pipeline_data_models.py`
- **Purpose**: Ensures consistent data structures across all pipelines
- **Note**: This pipeline doesn't use video-specific base classes but follows shared data standards

### Current Status
- **Data structures** should align with shared model standards
- **Output formats** should be consistent with other pipelines
- **Metadata fields** should follow common naming conventions

## 📁 **Directory Structure**

```
SCIENTIFIC_PUBLICATION_PIPELINE/
├── step_01_metadata/               # PDF metadata extraction
├── step_02_metadata/               # Enhanced metadata processing
├── step_03_extracted_text/         # Text extraction from PDFs
├── step_04_extract_chunks/         # Semantic chunking
├── step_05_semantic_chunks/        # Enhanced semantic chunks
├── step_06_faiss_embeddings/       # Vector embeddings for RAG
├── step_07_archive/                # Processed PDF storage
├── logs/                           # Pipeline execution logs
├── pipeline_progress_queue.py      # Progress tracking
├── restore_progress_queue.py       # Progress restoration
└── README.md                       # This file
```

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- OpenAI API key (for GPT-4 enhancement)
- PDF processing libraries (PyPDF2, pdfplumber)
- Scientific computing libraries (numpy, scipy)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Pipeline
```bash
# Run individual steps
uv run python step_01_metadata.py
uv run python step_02_metadata.py
uv run python step_03_extracted_text.py
uv run python step_04_extract_chunks.py
uv run python step_05_semantic_chunks.py
uv run python step_06_faiss_embeddings.py

# Or run the complete pipeline
uv run python run_publications_pipeline.py
```

## 🔍 **Key Features**

### PDF Processing
- Automated PDF metadata extraction
- Text extraction with formatting preservation
- Scientific notation and equation handling
- Reference and citation extraction

### Semantic Analysis
- Uses GPT-4 for content understanding
- Extracts key scientific concepts
- Identifies research methodologies
- Generates content summaries

### Content Enhancement
- Metadata enrichment with scientific taxonomies
- Cross-referencing with existing knowledge
- Quality scoring and validation
- RAG-ready content preparation

## 📊 **Data Outputs**

### Step 4: Semantic Chunks
- **Location**: `step_04_extract_chunks/`
- **Format**: JSON with semantic content and metadata
- **Structure**: Follows shared data model standards

### Step 5: Enhanced Chunks
- **Location**: `step_05_semantic_chunks/`
- **Format**: JSON with enriched content and metadata
- **Structure**: Enhanced version of step 4 chunks

### Step 6: FAISS Embeddings
- **Location**: `step_06_faiss_embeddings/`
- **Format**: FAISS index + metadata pickle files
- **Integration**: Unified with other pipelines in Streamlit frontend

## 🔄 **Standardization Integration**

### Data Structure Alignment
- **Metadata fields** should match shared model conventions
- **Output formats** should be consistent with video pipelines
- **Validation** should use shared validation functions where applicable

### Cross-Pipeline Consistency
- **Field naming** should follow established patterns
- **Data types** should be consistent across pipelines
- **Metadata structure** should align with shared standards

## 🧪 **Testing and Validation**

### Data Validation
```python
# For applicable data structures
from utils.pipeline_data_models import validate_pipeline_data

# Validate against appropriate models
# Note: Publications pipeline may use different validation patterns
```

### Quality Assurance
- **Content completeness** checks
- **Metadata accuracy** validation
- **Scientific terminology** verification
- **Cross-reference** validation

## 📝 **Development Guidelines**

### Adding New Features
1. **Follow shared data standards** where applicable
2. **Maintain consistency** with other pipeline outputs
3. **Document pipeline-specific requirements** clearly
4. **Use common metadata patterns** when possible

### Data Structure Changes
1. **Align with shared models** where relevant
2. **Maintain backward compatibility** where possible
3. **Test consistency** with other pipeline outputs
4. **Update documentation** to reflect changes

### Pipeline Extensions
1. **Follow established patterns** for similar functionality
2. **Use consistent naming conventions**
3. **Implement appropriate validation**
4. **Document any deviations** from shared standards

## 🔗 **Related Documentation**

- **Shared Models**: `utils/README.md`
- **Main Architecture**: `ARCHITECTURE.md`
- **Quick Start Guide**: `QUICKSTART.md`
- **Video Pipeline Standards**: See video pipeline READMEs

## 🎯 **Next Steps for Developers**

1. **Review shared data standards** in `utils/README.md`
2. **Align output structures** with other pipelines where possible
3. **Implement consistent metadata patterns**
4. **Test cross-pipeline compatibility**
5. **Document any pipeline-specific requirements**

## 🆘 **Troubleshooting**

### Common Issues
- **Data structure inconsistencies**: Check against shared model standards
- **Metadata field mismatches**: Align with common naming conventions
- **Output format variations**: Ensure consistency with other pipelines

### Getting Help
- **Check shared model documentation** in `utils/README.md`
- **Review other pipeline implementations** for patterns
- **Consult main architecture documentation**
- **Test cross-pipeline compatibility**

## 🔄 **Pipeline-Specific Considerations**

### Publication Characteristics
- **Text-based content**: No video or audio processing
- **Scientific structure**: Abstracts, methods, results, conclusions
- **Reference systems**: Citations and bibliographies
- **Mathematical content**: Equations and scientific notation

### Optimization Strategies
- **Chunk sizing**: Optimized for scientific paper structure
- **Metadata extraction**: Focus on academic and research context
- **Content enhancement**: Emphasize scientific accuracy and completeness
- **Cross-referencing**: Link related concepts and references

This pipeline is part of a larger effort to standardize data structures across all processing pipelines, ensuring consistency, maintainability, and reduced troubleshooting in the unified RAG system. While it doesn't use video-specific base classes, it follows shared data standards and maintains consistency with other pipeline outputs.
