#!/usr/bin/env python3
"""
Label and Prepare Conversation Chunks

This tool helps you:
1. Review and label which speaker is Michael Levin
2. Prepare chunks for different use cases (RAG vs fine-tuning)
3. Create Q&A pairs for fine-tuning
4. Generate embeddings for RAG chunks
"""

import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationChunkLabeler:
    def __init__(self, base_dir: str = "data/ingested/youtube"):
        self.base_dir = Path(base_dir)
        
        # These will be set when we find playlist directories
        self.current_playlist_dir = None
        self.conversations_dir = None
        self.labeled_dir = None
        self.rag_dir = None
        self.finetune_dir = None
    
    def find_playlist_directories(self) -> List[Path]:
        """Find all playlist directories containing conversations."""
        playlist_dirs = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name not in ['videos', 'transcripts', 'frames', 'metadata', 'conversations']:
                # Check if this looks like a playlist directory
                conversations_dir = item / "conversations"
                if conversations_dir.exists():
                    playlist_dirs.append(item)
        
        return playlist_dirs
    
    def set_playlist(self, playlist_dir: Path):
        """Set the current playlist to work with."""
        self.current_playlist_dir = playlist_dir
        self.conversations_dir = playlist_dir / "conversations"
        self.labeled_dir = playlist_dir / "labeled"
        self.rag_dir = playlist_dir / "rag_chunks"
        self.finetune_dir = playlist_dir / "finetune_data"
        
        # Create output directories
        for dir_path in [self.labeled_dir, self.rag_dir, self.finetune_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_video_manifests(self) -> List[Dict]:
        """Get all video manifests for review."""
        if not self.conversations_dir:
            return []
        
        manifests = []
        
        for video_dir in self.conversations_dir.iterdir():
            if video_dir.is_dir():
                manifest_path = video_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                        manifest['video_dir'] = str(video_dir)
                        manifests.append(manifest)
        
        return manifests
    
    def display_speaker_summary(self, manifests: List[Dict]):
        """Display summary of speakers across all videos."""
        logger.info("📊 Speaker Summary Across All Videos:")
        logger.info("=" * 50)
        
        all_speakers = set()
        speaker_counts = {}
        
        for manifest in manifests:
            video_id = manifest['video_id']
            speakers = manifest['speakers']
            
            logger.info(f"\n📹 {video_id}:")
            logger.info(f"   Speakers: {speakers}")
            logger.info(f"   Total chunks: {manifest['total_chunks']}")
            
            for speaker in speakers:
                all_speakers.add(speaker)
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        logger.info(f"\n🎭 Unique speakers across all videos: {sorted(all_speakers)}")
        logger.info(f"📈 Speaker frequency: {speaker_counts}")
    
    def interactive_speaker_labeling(self, manifests: List[Dict]) -> Dict[str, str]:
        """Interactively label which speakers are Michael Levin, one video at a time."""
        logger.info("\n🏷️  Interactive Speaker Labeling")
        logger.info("=" * 40)
        
        # Load existing speaker labels first
        speaker_labels = self.load_speaker_labels() or {}
        
        print("\nPlease identify the speakers for each video:")
        print("For each speaker, enter 'y' if Michael Levin, 'n' for other person, or 's' to skip")
        print("=" * 80)
        
        # Process one video at a time
        for manifest in manifests:
            video_id = manifest['video_id']
            speakers = manifest['speakers']
            
            # Load video metadata for context
            metadata_file = self.base_dir / "metadata" / f"{video_id}_metadata.json"
            video_title = "Unknown Title"
            video_channel = "Unknown Channel"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        video_title = metadata.get('title', 'Unknown Title')
                        video_channel = metadata.get('channel', 'Unknown Channel')
                except:
                    pass
            
            print(f"\n🎬 VIDEO: {video_id}")
            print(f"📺 Title: {video_title}")
            print(f"📡 Channel: {video_channel}")
            print(f"🎭 Speakers found: {speakers}")
            print("-" * 60)
            
            # Label speakers for this specific video (treat each video independently)
            for speaker in speakers:
                # Create a unique key for this speaker in this specific video
                speaker_key = f"{video_id}_{speaker}"
                
                # Skip if we already labeled this speaker
                if speaker_key in speaker_labels:
                    print(f"✅ Speaker '{speaker}' already labeled as: {speaker_labels[speaker_key]}")
                    continue
                
                print(f"\n🎭 SPEAKER: '{speaker}'")
                print(f"   📹 Video: {video_id}")
                print(f"   📺 Title: {video_title}")
                
                while True:
                    response = input(f"\n❓ Is speaker '{speaker}' Michael Levin? (y/n/s): ").lower().strip()
                    if response in ['y', 'yes']:
                        speaker_labels[speaker_key] = 'levin'
                        print(f"✅ Marked '{speaker}' in {video_id} as Michael Levin")
                        break
                    elif response in ['n', 'no']:
                        # Ask for the actual name
                        while True:
                            name = input(f"👤 Who is speaker '{speaker}'? (enter name): ").strip()
                            if name:
                                # Clean the name for consistency (lowercase, underscores)
                                clean_name = name.lower().replace(' ', '_').replace('.', '')
                                speaker_labels[speaker_key] = clean_name
                                print(f"✅ Marked '{speaker}' in {video_id} as {name}")
                                break
                            else:
                                print("Please enter a name (or 's' to skip)")
                        break
                    elif response in ['s', 'skip']:
                        speaker_labels[speaker_key] = 'unknown'
                        print(f"⏭️  Skipped '{speaker}' in {video_id} (will need manual review)")
                        break
                    else:
                        print("Please enter 'y' for Levin, 'n' for other person, or 's' to skip")
                
                print("-" * 40)
            
            print(f"✅ Completed labeling for video: {video_id}")
            print("=" * 80)
        
        # Save speaker labels
        labels_file = self.labeled_dir / "speaker_labels.json"
        with open(labels_file, 'w') as f:
            json.dump(speaker_labels, f, indent=2)
        
        logger.info(f"💾 Saved speaker labels to: {labels_file}")
        return speaker_labels
    
    def load_speaker_labels(self) -> Optional[Dict[str, str]]:
        """Load existing speaker labels."""
        labels_file = self.labeled_dir / "speaker_labels.json"
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                return json.load(f)
        return None
    
    def apply_speaker_labels(self, speaker_labels: Dict[str, str]):
        """Apply speaker labels to all chunks."""
        logger.info("🏷️  Applying speaker labels to chunks...")
        
        labeled_count = 0
        total_chunks = 0
        
        for video_dir in self.conversations_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            video_id = video_dir.name
            output_dir = self.labeled_dir / video_id
            output_dir.mkdir(exist_ok=True)
            
            # Process each chunk file
            for chunk_file in video_dir.glob("*.json"):
                if chunk_file.name == "manifest.json":
                    continue
                
                with open(chunk_file, 'r') as f:
                    chunk = json.load(f)
                
                total_chunks += 1
                
                # Apply speaker label using video-specific key
                speaker = chunk['speaker']
                speaker_key = f"{video_id}_{speaker}"
                
                if speaker_key in speaker_labels:
                    chunk['is_levin'] = (speaker_labels[speaker_key] == 'levin')
                    chunk['speaker_label'] = speaker_labels[speaker_key]
                    chunk['speaker_name'] = speaker_labels[speaker_key] if speaker_labels[speaker_key] != 'levin' else 'michael_levin'
                    labeled_count += 1
                else:
                    chunk['is_levin'] = None
                    chunk['speaker_label'] = 'unknown'
                    chunk['speaker_name'] = 'unknown'
                
                # Save labeled chunk
                output_file = output_dir / chunk_file.name
                with open(output_file, 'w') as f:
                    json.dump(chunk, f, indent=2)
        
        logger.info(f"✅ Applied labels to {labeled_count}/{total_chunks} chunks")
    
    def create_rag_chunks(self):
        """Create chunks suitable for RAG (semantic search)."""
        logger.info("🔍 Creating RAG chunks...")
        
        rag_chunks = []
        chunk_id = 0
        
        for video_dir in self.labeled_dir.iterdir():
            if not video_dir.is_dir() or video_dir.name == "speaker_labels.json":
                continue
            
            for chunk_file in video_dir.glob("*.json"):
                with open(chunk_file, 'r') as f:
                    chunk = json.load(f)
                
                # Only include chunks that are substantial and from Levin
                if (chunk.get('is_levin') == True and 
                    chunk.get('word_count', 0) > 20):
                    
                    chunk_id += 1
                    rag_chunk = {
                        'id': f"conv_chunk_{chunk_id:04d}",
                        'text': chunk['text'],
                        'speaker': chunk['speaker'],
                        'speaker_name': chunk.get('speaker_name', 'unknown'),
                        'is_levin': chunk['is_levin'],
                        'video_id': chunk['video_id'],
                        'original_chunk_id': chunk['chunk_id'],
                        'topic': chunk.get('topic', 'general'),
                        'summary': chunk.get('summary', ''),
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'word_count': chunk['word_count'],
                        'chunk_type': 'conversation_rag'
                    }
                    
                    rag_chunks.append(rag_chunk)
        
        # Save RAG chunks
        rag_file = self.rag_dir / "conversation_chunks.json"
        with open(rag_file, 'w') as f:
            json.dump(rag_chunks, f, indent=2)
        
        logger.info(f"✅ Created {len(rag_chunks)} RAG chunks: {rag_file}")
        return rag_chunks
    
    def create_qa_pairs(self) -> List[Dict]:
        """Create Q&A pairs for fine-tuning."""
        logger.info("❓ Creating Q&A pairs for fine-tuning...")
        
        qa_pairs = []
        
        for video_dir in self.labeled_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            # Load all chunks for this video
            chunks = []
            for chunk_file in video_dir.glob("*.json"):
                with open(chunk_file, 'r') as f:
                    chunk = json.load(f)
                    chunks.append(chunk)
            
            # Sort by start time
            chunks.sort(key=lambda x: x['start_time'])
            
            # Look for question-answer patterns
            for i, chunk in enumerate(chunks):
                if chunk.get('chunk_type') == 'question' and not chunk.get('is_levin', False):
                    # This is a question from someone other than Levin
                    question = chunk['text'].strip()
                    
                    # Look for Levin's response in the next few chunks
                    for j in range(i + 1, min(i + 4, len(chunks))):
                        next_chunk = chunks[j]
                        if (next_chunk.get('is_levin', False) and 
                            next_chunk.get('word_count', 0) > 10):
                            
                            answer = next_chunk['text'].strip()
                            
                            qa_pair = {
                                'question': question,
                                'answer': answer,
                                'questioner': chunk.get('speaker_name', 'unknown'),
                                'answerer': 'michael_levin',
                                'question_chunk_id': chunk['chunk_id'],
                                'answer_chunk_id': next_chunk['chunk_id'],
                                'video_id': chunk['video_id'],
                                'topic': chunk.get('topic', 'general'),
                                'question_start_time': chunk['start_time'],
                                'answer_start_time': next_chunk['start_time'],
                                'confidence': 'auto_detected'
                            }
                            
                            qa_pairs.append(qa_pair)
                            break
        
        # Save Q&A pairs
        qa_file = self.finetune_dir / "qa_pairs.json"
        with open(qa_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        
        logger.info(f"✅ Created {len(qa_pairs)} Q&A pairs: {qa_file}")
        return qa_pairs
    
    def create_fine_tuning_datasets(self, qa_pairs: List[Dict]):
        """Create datasets in different formats for fine-tuning."""
        logger.info("📚 Creating fine-tuning datasets...")
        
        # OpenAI fine-tuning format (JSONL)
        openai_file = self.finetune_dir / "openai_finetune.jsonl"
        with open(openai_file, 'w') as f:
            for qa in qa_pairs:
                openai_format = {
                    "messages": [
                        {"role": "user", "content": qa['question']},
                        {"role": "assistant", "content": qa['answer']}
                    ]
                }
                f.write(json.dumps(openai_format) + '\n')
        
        # Hugging Face format
        hf_file = self.finetune_dir / "huggingface_dataset.json"
        hf_dataset = []
        for qa in qa_pairs:
            hf_format = {
                "instruction": qa['question'],
                "output": qa['answer'],
                "topic": qa['topic'],
                "video_id": qa['video_id']
            }
            hf_dataset.append(hf_format)
        
        with open(hf_file, 'w') as f:
            json.dump(hf_dataset, f, indent=2)
        
        # Summary statistics
        stats = {
            'total_qa_pairs': len(qa_pairs),
            'topics': {},
            'avg_question_length': sum(len(qa['question'].split()) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'avg_answer_length': sum(len(qa['answer'].split()) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0
        }
        
        for qa in qa_pairs:
            topic = qa['topic']
            stats['topics'][topic] = stats['topics'].get(topic, 0) + 1
        
        stats_file = self.finetune_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Created fine-tuning datasets:")
        logger.info(f"   OpenAI format: {openai_file}")
        logger.info(f"   Hugging Face format: {hf_file}")
        logger.info(f"   Statistics: {stats_file}")
        
        return stats
    
    def generate_summary_report(self):
        """Generate a summary report of the entire labeling process."""
        logger.info("📊 Generating summary report...")
        
        report = {
            'processing_date': str(datetime.datetime.now()),
            'directories': {
                'conversations': str(self.conversations_dir),
                'labeled': str(self.labeled_dir),
                'rag': str(self.rag_dir),
                'finetune': str(self.finetune_dir)
            },
            'statistics': {}
        }
        
        # Count files and chunks
        total_videos = len(list(self.conversations_dir.iterdir())) if self.conversations_dir.exists() else 0
        total_labeled_chunks = 0
        levin_chunks = 0
        other_chunks = 0
        
        if self.labeled_dir.exists():
            for video_dir in self.labeled_dir.iterdir():
                if video_dir.is_dir():
                    for chunk_file in video_dir.glob("*.json"):
                        with open(chunk_file, 'r') as f:
                            chunk = json.load(f)
                            total_labeled_chunks += 1
                            if chunk.get('is_levin'):
                                levin_chunks += 1
                            elif chunk.get('is_levin') is False:
                                other_chunks += 1
        
        report['statistics'] = {
            'total_videos': total_videos,
            'total_labeled_chunks': total_labeled_chunks,
            'levin_chunks': levin_chunks,
            'other_chunks': other_chunks,
            'unlabeled_chunks': total_labeled_chunks - levin_chunks - other_chunks
        }
        
        # Add file counts
        if self.rag_dir.exists():
            rag_files = list(self.rag_dir.glob("*.json"))
            report['statistics']['rag_files'] = len(rag_files)
        
        if self.finetune_dir.exists():
            ft_files = list(self.finetune_dir.glob("*.json*"))
            report['statistics']['finetune_files'] = len(ft_files)
        
        report_file = self.conversations_dir / "processing_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Summary Report:")
        logger.info(f"   Total videos: {report['statistics']['total_videos']}")
        logger.info(f"   Labeled chunks: {report['statistics']['total_labeled_chunks']}")
        logger.info(f"   Levin chunks: {report['statistics']['levin_chunks']}")
        logger.info(f"   Other speaker chunks: {report['statistics']['other_chunks']}")
        logger.info(f"   Report saved: {report_file}")
        
        return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Label and prepare conversation chunks")
    parser.add_argument("--base-dir", default="data/ingested/youtube",
                       help="Base directory containing playlist data")
    parser.add_argument("--action", choices=['label', 'process', 'all'], default='all',
                       help="Action to perform")
    parser.add_argument("--skip-labeling", action="store_true",
                       help="Skip interactive labeling (use existing labels)")
    
    args = parser.parse_args()
    
    # Create labeler
    labeler = ConversationChunkLabeler(args.base_dir)
    
    # Find available playlists
    playlist_dirs = labeler.find_playlist_directories()
    
    if not playlist_dirs:
        logger.error("❌ No playlist directories with conversations found!")
        return
    
    # Show available playlists
    print("\n📋 Available playlists with conversation data:")
    for i, playlist_dir in enumerate(playlist_dirs, 1):
        print(f"   {i}. {playlist_dir.name}")
    
    # Select playlist
    if len(playlist_dirs) == 1:
        selected_playlist = playlist_dirs[0]
        print(f"\n✅ Using playlist: {selected_playlist.name}")
    else:
        while True:
            try:
                choice = int(input(f"\nSelect playlist (1-{len(playlist_dirs)}): ")) - 1
                if 0 <= choice < len(playlist_dirs):
                    selected_playlist = playlist_dirs[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                return
    
    # Set the playlist
    labeler.set_playlist(selected_playlist)
    
    # Get manifests
    manifests = labeler.get_video_manifests()
    
    if not manifests:
        logger.error(f"❌ No video manifests found in {selected_playlist.name}")
        return
    
    logger.info(f"🎬 Found {len(manifests)} videos to process in {selected_playlist.name}")
    
    if args.action in ['label', 'all'] and not args.skip_labeling:
        # Show speaker summaries
        labeler.display_speaker_summary(manifests)
        
        # Interactive labeling
        labeler.interactive_speaker_labeling(manifests)
    
    if args.action in ['process', 'all']:
        # Apply labels and create outputs
        speaker_labels = labeler.load_speaker_labels()
        if speaker_labels:
            labeler.apply_speaker_labels(speaker_labels)
        labeler.create_rag_chunks()
        
        # Create Q&A pairs
        qa_pairs = labeler.create_qa_pairs()
        
        # Create fine-tuning datasets
        labeler.create_fine_tuning_datasets(qa_pairs)
        
        # Generate summary report
        labeler.generate_summary_report()
    
    logger.info("🎉 Processing complete!")


if __name__ == "__main__":
    import datetime
    main() 
