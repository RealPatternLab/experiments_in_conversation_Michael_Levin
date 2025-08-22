# 🎭 Speaker Identification System (1-on-2 Conversations)

## Overview

The enhanced **1-on-2 Conversations Video Pipeline** includes an **automatic speaker identification system** that handles **three-speaker dynamics** featuring Michael Levin and 2 other researchers. This system ensures that:

- **Michael Levin is correctly identified** in all conversations
- **Two guest researchers are properly labeled** with names and roles
- **Rich semantic context** is preserved for each speaker
- **Speaker mappings are stored** for future use across the pipeline
- **Multi-speaker interaction patterns** are captured and analyzed

## 🚀 **How It Works for 1-on-2 Conversations**

### **1. Automatic Detection**
When processing a video transcript, the system automatically:
- Detects the presence of speaker diarization data
- Identifies unique speakers (A, B, C)
- Checks if speaker mappings already exist for the video
- Validates that exactly 3 speakers are present

### **2. Enhanced User Input Request**
For new videos, the system uses an **intuitive three-step approach**:

#### **Step 1: Identify Michael Levin**
```
🎯 MICHAEL LEVIN IDENTIFICATION
Video: [VIDEO_ID]
Available speakers: A, B, C
==================================================
Which speaker is Michael Levin? (A, B, C): B
✅ Michael Levin identified as Speaker B
==================================================
```

#### **Step 2: Identify Second Speaker**
```
🎭 SPEAKER 2 IDENTIFICATION
Video: [VIDEO_ID]
Speaker ID: A
==================================================
Enter the name for Speaker A: [NAME]
Enter the role/organization for Speaker A: [ROLE]
Any additional context about Speaker A? (optional): [CONTEXT]
✅ Speaker A identified as: [NAME] ([ROLE])
```

#### **Step 3: Identify Third Speaker**
```
🎭 SPEAKER 3 IDENTIFICATION
Video: [VIDEO_ID]
Speaker ID: C
==================================================
Enter the name for Speaker C: [NAME]
Enter the role/organization for Speaker C: [ROLE]
Any additional context about Speaker C? (optional): [CONTEXT]
✅ Speaker C identified as: [NAME] ([ROLE])
```

### **3. Automatic Storage and Integration**
Once identified, speaker mappings are automatically:
- Stored in `step_04_extract_chunks/speaker_mappings.json`
- Used for all future processing of that video
- Integrated into semantic chunks and Q&A pairs
- Applied to multi-speaker interaction analysis

## 📊 **Speaker Information Collected for 1-on-2**

For each speaker, the system collects:

| Field | Description | Example |
|-------|-------------|---------|
| **speaker_id** | AssemblyAI speaker identifier | "A", "B", "C" |
| **name** | Full name of the speaker | "Michael Levin" |
| **role** | Role or organization | "Biologist, Tufts University" |
| **is_levin** | Whether this is Michael Levin | `true` or `false` |
| **speaker_order** | Order in conversation (1st, 2nd, 3rd) | 1, 2, or 3 |
| **additional_context** | Any extra information | "Expert in bioelectricity" |
| **identified_at** | Timestamp of identification | "2025-08-19T11:30:00" |

## 🔧 **Integration with Multi-Speaker Semantic Processing**

### **Enhanced Chunks with 3-Speaker Context**
Each semantic chunk now includes rich multi-speaker context:

```json
{
  "chunk_id": "[VIDEO_ID]_chunk_000",
  "semantic_text": "Full transcript content...",
  "speaker": "B",
  "speaker_name": "Michael Levin",
  "speaker_role": "Biologist, Tufts University",
  "is_levin": true,
  "speaker_order": 2,
  "speaker_additional_context": null,
  "speaker_identified_at": "2025-08-19T11:30:00",
  "conversation_context": {
    "speaker_context": {
      "speaker_id": "B",
      "speaker_name": "Michael Levin",
      "speaker_role": "Biologist, Tufts University",
      "is_levin": true,
      "speaker_order": 2,
      "additional_context": null,
      "identified_at": "2025-08-19T11:30:00"
    },
    "levin_context": {
      "levin_involved": true,
      "levin_role": "primary_speaker",
      "levin_contribution_type": "providing_expert_insights",
      "levin_expertise_area": "regeneration_biology",
      "levin_insights": ["Specific insights from Levin..."]
    },
    "multi_speaker_context": {
      "total_speakers": 3,
      "speaker_interaction_type": "collaborative_discussion",
      "conversation_flow": ["A asks question", "B (Levin) responds", "C elaborates"],
      "collaboration_pattern": "question_response_elaboration"
    }
  }
}
```

### **Enhanced Q&A Pairs with Multi-Speaker Attribution**
Q&A extraction now includes proper multi-speaker attribution:

```json
{
  "pair_id": "[VIDEO_ID]_qa_001",
  "question": {
    "speaker": "A",
    "text": "Question text...",
    "is_levin": false,
    "speaker_name": "[SPEAKER_NAME]",
    "speaker_role": "[SPEAKER_ROLE]",
    "speaker_order": 1
  },
  "answer": {
    "speaker": "B",
    "text": "Answer text...",
    "is_levin": true,
    "speaker_name": "Michael Levin",
    "speaker_role": "Biologist, Tufts University",
    "speaker_order": 2
  },
  "elaboration": {
    "speaker": "C",
    "text": "Elaboration text...",
    "is_levin": false,
    "speaker_name": "[SPEAKER_NAME]",
    "speaker_role": "[SPEAKER_ROLE]",
    "speaker_order": 3
  },
  "levin_involved": true,
  "multi_speaker_dynamics": {
    "interaction_pattern": "question_response_elaboration",
    "collaboration_type": "academic_discussion",
    "speaker_roles": ["questioner", "expert_responder", "elaborator"]
  }
}
```

## 🎯 **Benefits for Multi-Speaker Research Analysis**

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

### **Multi-Speaker Dynamics**
- **Conversation flow analysis** across 3 speakers
- **Collaboration pattern identification** (Q&A, debate, discussion)
- **Speaker role evolution** throughout conversation
- **Collective insight generation** tracking

### **Enhanced Search & Retrieval**
- **Speaker-specific queries** ("Show me Levin's insights on planaria")
- **Role-based filtering** ("Find responses from biologists")
- **Collaboration analysis** ("Show conversations between Levin and Wolfram researchers")
- **Multi-speaker patterns** ("Find 3-way discussions about regeneration")

## 📁 **File Structure for 1-on-2**

```
step_04_extract_chunks/
├── speaker_mappings.json          # Global speaker mappings
├── [VIDEO_ID]_chunks.json         # Enhanced chunks with 3-speaker context
├── [VIDEO_ID]_qa_pairs.json       # Q&A pairs with multi-speaker attribution
├── [VIDEO_ID]_qa_summary.json     # Q&A summary with speaker statistics
├── [VIDEO_ID]_multi_speaker_analysis.json  # Multi-speaker dynamics analysis
└── [VIDEO_ID]_chunks/             # Individual chunk files
    ├── [VIDEO_ID]_chunk_000.json
    ├── [VIDEO_ID]_chunk_001.json
    └── ...
```

## 🚀 **Usage Instructions for 1-on-2**

### **First Time Processing**
1. Run step 4: `uv run python step_04_extract_chunks.py`
2. System will automatically detect new videos with 3 speakers
3. For each speaker, provide:
   - **Name**: Full name of the person
   - **Role**: Their role or organization
   - **Levin Check**: Whether this is Michael Levin (y/n)
   - **Additional Context**: Any extra information (optional)

### **Subsequent Processing**
- Speaker mappings are automatically loaded
- No additional input required
- All processing uses stored speaker information
- Multi-speaker dynamics are automatically analyzed

### **Adding New Videos**
- Each new video will trigger speaker identification
- Mappings are stored per video
- No need to re-identify speakers for existing videos

## 🔍 **Example Speaker Mappings for 1-on-2**

```json
{
  "[VIDEO_ID]": {
    "A": {
      "speaker_id": "A",
      "name": "[SPEAKER_NAME]",
      "role": "[SPEAKER_ROLE]",
      "is_levin": false,
      "speaker_order": 1,
      "additional_context": null,
      "identified_at": "2025-08-19T11:30:00"
    },
    "B": {
      "speaker_id": "B",
      "name": "Michael Levin",
      "role": "Biologist, Tufts University",
      "is_levin": true,
      "speaker_order": 2,
      "additional_context": null,
      "identified_at": "2025-08-19T11:30:00"
    },
    "C": {
      "speaker_id": "C",
      "name": "[SPEAKER_NAME]",
      "role": "[SPEAKER_ROLE]",
      "is_levin": false,
      "speaker_order": 3,
      "additional_context": null,
      "identified_at": "2025-08-19T11:30:00"
    }
  }
}
```

## 🧪 **Testing the Multi-Speaker System**

You can test the speaker identification system using:

```bash
uv run python test_speaker_identification.py
```

This will demonstrate the interactive 3-speaker identification process.

## ✅ **Success Metrics for 1-on-2**

- **Speaker Identification**: 100% automatic detection and user input for 3 speakers
- **Levin Recognition**: Automatic identification of Michael Levin
- **Multi-Speaker Context**: Rich context for all 3 speakers in semantic chunks
- **Q&A Attribution**: Proper speaker attribution in all Q&A pairs
- **Collaboration Analysis**: Automatic detection of multi-speaker interaction patterns
- **Storage Efficiency**: Persistent speaker mappings for future use

## 🔮 **Future Enhancements for Multi-Speaker**

1. **Batch Processing**: Identify speakers for multiple videos at once
2. **Speaker Validation**: Cross-reference with known researcher databases
3. **Automatic Role Detection**: Use AI to suggest roles based on content
4. **Speaker Analytics**: Track speaker patterns across multiple videos
5. **Collaboration Networks**: Map researcher interaction patterns
6. **Multi-Speaker Dynamics**: Analyze conversation flow and collaboration types
7. **Speaker Role Evolution**: Track how speaker roles change throughout conversations

## 🎉 **Conclusion**

The new **1-on-2 speaker identification system** transforms the Conversations Video Pipeline into a **comprehensive multi-speaker research analysis tool** that:

- **Automatically requests** speaker identification for 3-speaker conversations
- **Preserves rich context** about who said what and why
- **Focuses on Michael Levin** while capturing guest researcher contributions
- **Enables advanced queries** based on speaker, role, and expertise
- **Provides enterprise-grade** semantic content for AI training and research
- **Analyzes multi-speaker dynamics** and collaboration patterns

This system ensures that every 3-speaker conversation is properly attributed, contextualized, and analyzed for collaboration patterns, making the pipeline's output invaluable for researchers studying Michael Levin's work and the broader field of developmental biology and bioelectricity.
