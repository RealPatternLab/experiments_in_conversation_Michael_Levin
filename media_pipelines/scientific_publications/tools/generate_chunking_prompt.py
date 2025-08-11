#!/usr/bin/env python3
"""
Generate optimal prompt for semantic chunking using Gemini Pro.
This tool asks Gemini to create a prompt for chunking scientific papers.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent))

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: pip install google-generativeai python-dotenv")
    sys.exit(1)

def generate_chunking_prompt() -> str:
    """
    Ask Gemini Pro to generate an optimal prompt for semantic chunking.
    
    Returns:
        str: The generated prompt
    """
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("Please add your Google API key to the .env file")
        return None
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # The request to Gemini
    request = """
You are an expert at creating prompts for Large Language Models. I need you to generate an optimal prompt for semantic chunking of scientific papers.

Here's what I want to achieve:

I have a collection of scientific papers written by Michael Levin, a prominent researcher in developmental biology, bioelectricity, and regenerative medicine. I want to use Gemini Pro to extract self-contained, semantically meaningful chunks from these papers that are useful for knowledge retrieval. These chunks may cross page boundaries and should preserve logical flow and complete ideas.

I want Gemini to return a JSON array where each chunk object contains:
- `text`: The semantically meaningful paragraph or group of sentences (ideally 100–300 words)
- `section`: Which part of the paper the chunk is from (e.g., Abstract, Introduction, Methods, Results, Discussion, Conclusion)
- `topic`: A concise topic or concept that this chunk is about (e.g., "Bioelectric signaling in regeneration", "Gap junctions", "Computational modeling")
- `chunk_summary`: A one-sentence summary of the chunk's core message
- `position_in_section`: Indicate if it appears at the beginning, middle, or end of the section
- `certainty_level`: Your confidence (High, Medium, Low) that the chunk expresses Levin's core view or a key claim
- `citation_context`: If relevant, describe whether the chunk is referring to prior work, presenting new results, or drawing conclusions

Important considerations:
- The author is Michael Levin, a leading researcher in bioelectricity, developmental biology, and regenerative medicine
- Do not include captions, footnotes, or unrelated visual figure references unless they are semantically necessary
- The chunks should be useful for knowledge retrieval and question answering
- We will use Pydantic to enforce the response format, so the prompt should be clear about the expected JSON structure
- The papers cover topics like bioelectricity, regeneration, developmental biology, computational modeling, and cellular communication
- Some papers may have complex layouts, figures, and references
- Focus on Levin's key contributions and insights in each chunk

Please generate the optimal prompt that I should send to Gemini Pro along with each PDF. Make it comprehensive, clear, and effective for this specific task, emphasizing that we're working with Michael Levin's scientific papers.
"""

    try:
        print("🤖 Asking Gemini Pro to generate the optimal chunking prompt...")
        
        response = model.generate_content(request)
        
        if response.text:
            print("✅ Generated prompt successfully!")
            return response.text
        else:
            print("❌ No response from Gemini")
            return None
            
    except Exception as e:
        print(f"❌ Error generating prompt: {e}")
        return None

def save_prompt(prompt: str, output_file: str = "data/chunking_prompt.txt") -> None:
    """
    Save the generated prompt to a file.
    
    Args:
        prompt: The generated prompt
        output_file: Path to save the prompt
    """
    if not prompt:
        print("❌ No prompt to save")
        return
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    print(f"💾 Prompt saved to: {output_path}")
    print(f"📝 Prompt length: {len(prompt)} characters")

def main():
    """Main function to generate and save the chunking prompt."""
    print("🚀 Generating optimal semantic chunking prompt...")
    print("=" * 60)
    
    prompt = generate_chunking_prompt()
    
    if prompt:
        print("\n📋 Generated Prompt:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        
        save_prompt(prompt)
        
        print("\n✅ Prompt generation complete!")
        print("📁 Next steps:")
        print("   1. Review the generated prompt in data/chunking_prompt.txt")
        print("   2. Create the semantic chunking tool using this prompt")
        print("   3. Test with a sample PDF")
    else:
        print("❌ Failed to generate prompt")
        sys.exit(1)

if __name__ == "__main__":
    main() 