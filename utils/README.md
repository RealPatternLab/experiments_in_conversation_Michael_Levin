# Pipeline Data Models & Base Classes

This directory contains shared data models and base classes that ensure consistency across different video pipelines while maintaining flexibility for pipeline-specific requirements.

## 🎯 **Why This Solution?**

**Problem**: Different video pipelines were generating inconsistent data structures, causing troubleshooting issues in the Streamlit frontend.

**Solution**: 
- **Pydantic models** for type safety and validation
- **Base classes** for common functionality
- **Inheritance** for pipeline-specific extensions
- **Factory functions** for creating appropriate data types

## 🏗️ **Architecture**

```
BaseFrameChunkAligner (Abstract Base Class)
├── FormalPresentationsFrameChunkAligner
└── ConversationsFrameChunkAligner
```

## 📁 **Files**

### Core Models
- **`pipeline_data_models.py`** - Pydantic models for data structures
- **`base_frame_chunk_aligner.py`** - Abstract base class for frame-chunk alignment

### Examples & Tests
- **`conversations_aligner_example.py`** - Example implementation for conversations pipeline
- **`test_data_validation.py`** - Test script demonstrating validation
- **`requirements.txt`** - Dependencies for the utils directory

## 🚀 **Usage**

### 1. Install Dependencies
```bash
cd utils
pip install -r requirements.txt
```

### 2. Import and Use
```python
from utils.pipeline_data_models import PipelineType, validate_pipeline_data
from utils.base_frame_chunk_aligner import BaseFrameChunkAligner

# Validate existing data
is_valid = validate_pipeline_data(data, PipelineType.CONVERSATIONS)

# Create new aligner
class MyPipelineAligner(BaseFrameChunkAligner):
    def __init__(self):
        super().__init__(PipelineType.FORMAL_PRESENTATIONS)
    
    # Implement required abstract methods...
```

### 3. Refactor Existing Pipeline
```python
# Before: Standalone class with duplicated code
class FrameChunkAligner:
    def __init__(self):
        # Lots of duplicated initialization code...
    
    def save_alignments(self, ...):
        # Duplicated saving logic...

# After: Inherit from base class
class ConversationsFrameChunkAligner(BaseFrameChunkAligner):
    def __init__(self):
        super().__init__(PipelineType.CONVERSATIONS)
    
    # Only implement pipeline-specific logic
    def find_all_chunks(self):
        # Conversations-specific chunk finding...
```

## 🔍 **Data Validation**

The system automatically validates data structures:

```python
# This will raise validation errors for:
# - Missing required fields
# - Wrong data types
# - Invalid nested structures

try:
    validate_pipeline_data(data, PipelineType.CONVERSATIONS)
    print("✅ Data is valid")
except Exception as e:
    print(f"❌ Validation failed: {e}")
```

## 🎨 **Pipeline-Specific Extensions**

Add pipeline-specific fields through inheritance:

```python
class ConversationsRAGEntry(BaseRAGEntry):
    conversation_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation-specific context (Q&A, speakers, etc.)"
    )
```

## 📊 **Benefits**

1. **Consistency**: All pipelines generate the same base structure
2. **Validation**: Automatic data validation catches errors early
3. **Maintainability**: Common code is centralized
4. **Flexibility**: Pipeline-specific needs are accommodated
5. **Type Safety**: IDE support and runtime validation
6. **Reduced Complexity**: Less code duplication

## 🔄 **Migration Path**

1. **Phase 1**: Create shared models (✅ Done)
2. **Phase 2**: Refactor conversations pipeline to use base class
3. **Phase 3**: Refactor formal presentations pipeline to use base class
4. **Phase 4**: Update Streamlit app to expect consistent structures
5. **Phase 5**: Add new pipeline types easily

## 🧪 **Testing**

Run the test script to see validation in action:

```bash
cd utils
python test_data_validation.py
```

## 📝 **Adding New Pipeline Types**

1. **Add to PipelineType enum**:
```python
class PipelineType(str, Enum):
    FORMAL_PRESENTATIONS = "formal_presentations"
    CONVERSATIONS = "conversations"
    WORKING_MEETINGS = "working_meetings"
    NEW_PIPELINE = "new_pipeline"  # Add here
```

2. **Create pipeline-specific RAG entry**:
```python
class NewPipelineRAGEntry(BaseRAGEntry):
    new_pipeline_field: str = Field("", description="Pipeline-specific field")
```

3. **Update factory functions** to handle the new type

4. **Create pipeline-specific aligner** inheriting from `BaseFrameChunkAligner`

## 🎯 **Next Steps**

1. **Refactor existing pipelines** to use the base class
2. **Update Streamlit app** to expect consistent structures
3. **Add validation** to pipeline outputs
4. **Test with real data** to ensure compatibility
5. **Document any pipeline-specific requirements**

This system provides a solid foundation for consistent, maintainable video pipeline development while preserving the flexibility needed for different content types.
