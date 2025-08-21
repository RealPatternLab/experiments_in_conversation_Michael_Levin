#!/usr/bin/env python3
"""
Generate optimized prompts using Gemini for the Michael Levin RAG system.
This script will create prompts for:
1. Semantic chunking from Levin's perspective
2. Q&A pair generation with synthetic questions
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def get_semantic_chunking_prompt():
    """Get Gemini's recommendation for semantic chunking prompt"""
    
    prompt = """You are an expert in creating prompts for AI systems that extract semantic chunks from scientific conversations.

I am building a RAG (Retrieval Augmented Generation) system to emulate the researcher Michael Levin. We want semantic chunks from the conversations he has had in the past that are relevant to pull into the answers people might ask the virtual Michael Levin.

We have transcripts labeled as Levin and other researchers. Now we need to send this off to an LLM and prompt it to create semantic chunks that would be worth embedding from Levin's POV.

What prompt should we use for this? 

The prompt should:
1. Focus on extracting chunks that represent Levin's knowledge, insights, and expertise
2. Be suitable for embedding in a RAG system
3. Capture the essence of what Levin knows and can explain
4. Be clear and specific enough for an LLM to follow
5. Result in chunks that would be useful for answering questions about Levin's work

Please provide a complete, ready-to-use prompt that we can directly use in our pipeline."""

    print("🤖 Asking Gemini for semantic chunking prompt...")
    print("=" * 60)
    
    response = model.generate_content(prompt)
    
    print("📝 Gemini's recommended semantic chunking prompt:")
    print("=" * 60)
    print(response.text)
    print("=" * 60)
    
    return response.text

def get_qa_generation_prompt():
    """Get Gemini's recommendation for Q&A generation prompt"""
    
    prompt = """You are an expert in creating prompts for AI systems that generate Question/Answer pairs from scientific conversations.

I am building a RAG system to emulate the researcher Michael Levin. We need to generate Q&A pairs for training and fine-tuning purposes.

We want an LLM to digest the entire labeled transcripts and then generate a bunch of Question/Answer pairs for Levin. Levin may go on answering a question or adding to the conversation and there may be no exact question that generated that response. Instead, we want to generate synthetic questions that would have generated this type of response from Levin.

What prompt should we use for this?

The prompt should:
1. Generate realistic questions that Levin might be asked
2. Create synthetic questions when Levin's responses don't have explicit questions
3. Ensure the Q&A pairs are useful for training a virtual Levin
4. Be clear and specific enough for an LLM to follow
5. Result in high-quality training data

Please provide a complete, ready-to-use prompt that we can directly use in our pipeline."""

    print("\n🤖 Asking Gemini for Q&A generation prompt...")
    print("=" * 60)
    
    response = model.generate_content(prompt)
    
    print("📝 Gemini's recommended Q&A generation prompt:")
    print("=" * 60)
    print(response.text)
    print("=" * 60)
    
    return response.text

def save_prompts_to_file(semantic_prompt, qa_prompt):
    """Save the generated prompts to a JSON file"""
    
    prompts = {
        "semantic_chunking_prompt": semantic_prompt,
        "qa_generation_prompt": qa_prompt,
        "generated_at": "2025-08-19",
        "description": "Gemini-generated prompts for Michael Levin RAG system"
    }
    
    output_file = Path("gemini_generated_prompts.json")
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"\n💾 Prompts saved to: {output_file}")
    return output_file

def main():
    """Main function to generate and save prompts"""
    
    print("🚀 Generating optimized prompts using Gemini for Michael Levin RAG system")
    print("=" * 80)
    
    try:
        # Get semantic chunking prompt
        semantic_prompt = get_semantic_chunking_prompt()
        
        # Get Q&A generation prompt
        qa_prompt = get_qa_generation_prompt()
        
        # Save prompts to file
        output_file = save_prompts_to_file(semantic_prompt, qa_prompt)
        
        print(f"\n✅ Successfully generated and saved prompts!")
        print(f"📁 Output file: {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating prompts: {e}")
        raise

if __name__ == "__main__":
    main()
