# 🎭 Speaker Identification System

## Overview

The enhanced Conversations Video Pipeline now includes an **automatic speaker identification system** that will request user input to properly identify and label all speakers in each video. This system ensures that:

- **Michael Levin is correctly identified** in all conversations
- **Guest researchers are properly labeled** with names and roles
- **Rich semantic context** is preserved for each speaker
- **Speaker mappings are stored** for future use across the pipeline

## 🚀 **How It Works**

### **1. Automatic Detection**
When processing a video transcript, the system automatically:
- Detects the presence of speaker diarization data
- Identifies unique speakers (A, B, C, etc.)
- Checks if speaker mappings already exist for the video

### **2. User Input Request**
For new videos, the system will use a **much more intuitive approach**:

1. **First, identify Michael Levin**: The system asks which speaker (A, B, C, etc.) is Michael Levin
2. **Then, identify other speakers**: For each remaining speaker, provide their name and role

```
🎯 MICHAEL LEVIN IDENTIFICATION
Video: LKSH4QNqpIE
Available speakers: A, B
==================================================
Which speaker is Michael Levin? (A, B): B
✅ Michael Levin identified as Speaker B
==================================================

🎭 SPEAKER IDENTIFICATION
Video: LKSH4QNqpIE
Speaker ID: A
==================================================
Enter the name for Speaker A: Willem Nielsen
Enter the role/organization for Willem Nielsen: Wolfram Institute
Any additional context about Willem Nielsen? (optional): 
✅ Speaker A identified as: Willem Nielsen (Wolfram Institute)
==================================================
```

This approach is **much clearer** because:
- You immediately know which speaker is Michael Levin
- No confusing "Is X Michael Levin?" questions
- Clear separation between Levin and other speakers
- More efficient identification process

### **3. Automatic Storage**
Once identified, speaker mappings are automatically:
- Stored in `step_04_extract_chunks/speaker_mappings.json`
- Used for all future processing of that video
- Integrated into semantic chunks and Q&A pairs

## 📊 **Speaker Information Collected**

For each speaker, the system collects:

| Field | Description | Example |
|-------|-------------|---------|
| **speaker_id** | AssemblyAI speaker identifier | "A", "B", "C" |
| **name** | Full name of the speaker | "Michael Levin" |
| **role** | Role or organization | "Biologist, Tufts University" |
| **is_levin** | Whether this is Michael Levin | `true` or `false` |
| **additional_context** | Any extra information | "Expert in bioelectricity" |
| **identified_at** | Timestamp of identification | "2025-08-19T11:30:00" |

## 🔧 **Integration with Semantic Processing**

### **Enhanced Chunks**
Each semantic chunk now includes rich speaker context:

```json
{
  "chunk_id": "LKSH4QNqpIE_chunk_000",
  "semantic_text": "Full transcript content...",
  "speaker": "B",
  "speaker_name": "Michael Levin",
  "speaker_role": "Biologist, Tufts University",
  "is_levin": true,
  "speaker_additional_context": null,
  "speaker_identified_at": "2025-08-19T11:30:00",
  "conversation_context": {
    "speaker_context": {
      "speaker_id": "B",
      "speaker_name": "Michael Levin",
      "speaker_role": "Biologist, Tufts University",
      "is_levin": true,
      "additional_context": null,
      "identified_at": "2025-08-19T11:30:00"
    },
    "levin_context": {
      "levin_involved": true,
      "levin_role": "primary_speaker",
      "levin_contribution_type": "providing_expert_insights",
      "levin_expertise_area": "regeneration_biology",
      "levin_insights": ["Specific insights from Levin..."]
    }
  }
}
```

### **Enhanced Q&A Pairs**
Q&A extraction now includes proper speaker attribution:

```json
{
  "pair_id": "LKSH4QNqpIE_qa_001",
  "question": {
    "speaker": "A",
    "text": "Question text...",
    "is_levin": false,
    "speaker_name": "Willem Nielsen",
    "speaker_role": "Wolfram Institute"
  },
  "answer": {
    "speaker": "B",
    "text": "Answer text...",
    "is_levin": true,
    "speaker_name": "Michael Levin",
    "speaker_role": "Biologist, Tufts University"
  },
  "levin_involved": true
}
```

## 🎯 **Benefits for Research Analysis**

### **Michael Levin Focus**
- **Automatic identification** of Levin's contributions
- **Expertise area mapping** (regeneration, bioelectricity, etc.)
- **Insight extraction** from Levin's explanations
- **Contribution type analysis** (questions, insights, responses)

### **Guest Researcher Context**
- **Proper attribution** of questions and insights
- **Role-based analysis** (academic, industry, etc.)
- **Collaboration patterns** identification
- **Cross-speaker interaction** analysis

### **Enhanced Search & Retrieval**
- **Speaker-specific queries** ("Show me Levin's insights on planaria")
- **Role-based filtering** ("Find responses from biologists")
- **Collaboration analysis** ("Show conversations between Levin and Wolfram researchers")

## 📁 **File Structure**

```
step_04_extract_chunks/
├── speaker_mappings.json          # Global speaker mappings
├── LKSH4QNqpIE_chunks.json       # Enhanced chunks with speaker context
├── LKSH4QNqpIE_qa_pairs.json     # Q&A pairs with speaker attribution
├── LKSH4QNqpIE_qa_summary.json   # Q&A summary with speaker statistics
└── LKSH4QNqpIE_chunks/           # Individual chunk files
    ├── LKSH4QNqpIE_chunk_000.json
    ├── LKSH4QNqpIE_chunk_001.json
    └── ...
```

## 🚀 **Usage Instructions**

### **First Time Processing**
1. Run step 4: `uv run python step_04_extract_chunks.py`
2. System will automatically detect new videos
3. For each speaker, provide:
   - **Name**: Full name of the person
   - **Role**: Their role or organization
   - **Levin Check**: Whether this is Michael Levin (y/n)
   - **Additional Context**: Any extra information (optional)

### **Subsequent Processing**
- Speaker mappings are automatically loaded
- No additional input required
- All processing uses stored speaker information

### **Adding New Videos**
- Each new video will trigger speaker identification
- Mappings are stored per video
- No need to re-identify speakers for existing videos

## 🔍 **Example Speaker Mappings**

```json
{
  "LKSH4QNqpIE": {
    "A": {
      "speaker_id": "A",
      "name": "Willem Nielsen",
      "role": "Wolfram Institute",
      "is_levin": false,
      "additional_context": null,
      "identified_at": "2025-08-19T11:30:00"
    },
    "B": {
      "speaker_id": "B",
      "name": "Michael Levin",
      "role": "Biologist, Tufts University",
      "is_levin": true,
      "additional_context": null,
      "identified_at": "2025-08-19T11:30:00"
    }
  }
}
```

## 🧪 **Testing the System**

You can test the speaker identification system using:

```bash
uv run python test_speaker_identification.py
```

This will demonstrate the interactive speaker identification process.

## ✅ **Success Metrics**

- **Speaker Identification**: 100% automatic detection and user input
- **Levin Recognition**: Automatic identification of Michael Levin
- **Context Preservation**: Rich speaker context in all semantic chunks
- **Q&A Attribution**: Proper speaker attribution in all Q&A pairs
- **Storage Efficiency**: Persistent speaker mappings for future use

## 🔮 **Future Enhancements**

1. **Batch Processing**: Identify speakers for multiple videos at once
2. **Speaker Validation**: Cross-reference with known researcher databases
3. **Automatic Role Detection**: Use AI to suggest roles based on content
4. **Speaker Analytics**: Track speaker patterns across multiple videos
5. **Collaboration Networks**: Map researcher interaction patterns

## 🎉 **Conclusion**

The new speaker identification system transforms the Conversations Video Pipeline from a basic transcript processor into a **comprehensive research analysis tool** that:

- **Automatically requests** speaker identification for new videos
- **Preserves rich context** about who said what and why
- **Focuses on Michael Levin** while capturing guest researcher contributions
- **Enables advanced queries** based on speaker, role, and expertise
- **Provides enterprise-grade** semantic content for AI training and research

This system ensures that every conversation is properly attributed and contextualized, making the pipeline's output invaluable for researchers studying Michael Levin's work and the broader field of developmental biology and bioelectricity.
