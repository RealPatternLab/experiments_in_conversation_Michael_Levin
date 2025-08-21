#!/usr/bin/env python3
"""
Check what's actually indexed in the Streamlit instance running on port 8501
"""

import pickle
from pathlib import Path

def check_streamlit_indices():
    """Check what videos and chunks are indexed in the Streamlit instance"""
    
    print("🔍 CHECKING STREAMLIT INDEX CONTENT")
    print("=" * 50)
    
    # Check formal presentations pipeline
    print("\n📹 FORMAL PRESENTATIONS PIPELINE:")
    print("-" * 30)
    
    formal_faiss_dir = Path("SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/step_07_faiss_embeddings")
    if formal_faiss_dir.exists():
        latest_run = max([d for d in formal_faiss_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        print(f"✅ Latest run: {latest_run.name}")
        
        metadata_file = latest_run / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"   Total entries: {len(metadata)}")
            
            # Count videos
            video_counts = {}
            for entry in metadata:
                if isinstance(entry, dict):
                    video_id = entry.get('video_id', 'unknown')
                    video_counts[video_id] = video_counts.get(video_id, 0) + 1
            
            print(f"   Videos indexed: {len(video_counts)}")
            for video_id, count in sorted(video_counts.items()):
                print(f"     - {video_id}: {count} chunks")
        else:
            print("   ❌ No metadata file found")
    else:
        print("   ❌ FAISS directory not found")
    
    # Check conversations pipeline
    print("\n💬 CONVERSATIONS PIPELINE:")
    print("-" * 30)
    
    conversations_faiss_dir = Path("SCIENTIFIC_VIDEO_PIPELINE/Conversations_and_working_meetings_1_on_0/step_07_faiss_embeddings")
    if conversations_faiss_dir.exists():
        latest_run = max([d for d in conversations_faiss_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        print(f"✅ Latest run: {latest_run.name}")
        
        metadata_file = latest_run / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"   Total entries: {len(metadata)}")
            
            # Count videos
            video_counts = {}
            for entry in metadata:
                if isinstance(entry, dict):
                    video_id = entry.get('video_id', 'unknown')
                    video_counts[video_id] = video_counts.get(video_id, 0) + 1
            
            print(f"   Videos indexed: {len(video_counts)}")
            for video_id, count in sorted(video_counts.items()):
                print(f"     - {video_id}: {count} chunks")
        else:
            print("   ❌ No metadata file found")
    else:
        print("   ❌ FAISS directory not found")
    
    # Check publications pipeline
    print("\n📚 PUBLICATIONS PIPELINE:")
    print("-" * 30)
    
    publications_faiss_dir = Path("SCIENTIFIC_PUBLICATION_PIPELINE/step_06_faiss_embeddings")
    if publications_faiss_dir.exists():
        latest_run = max([d for d in publications_faiss_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        print(f"✅ Latest run: {latest_run.name}")
        
        metadata_file = latest_run / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"   Total entries: {len(metadata)}")
            
            # Count publications
            pub_counts = {}
            for entry in metadata:
                if isinstance(entry, dict):
                    pub_id = entry.get('publication_id', entry.get('doi', 'unknown'))
                    pub_counts[pub_id] = pub_counts.get(pub_id, 0) + 1
            
            print(f"   Publications indexed: {len(pub_counts)}")
            for pub_id, count in list(pub_counts.items())[:5]:  # Show first 5
                print(f"     - {pub_id}: {count} chunks")
            if len(pub_counts) > 5:
                print(f"     ... and {len(pub_counts) - 5} more")
        else:
            print("   ❌ No metadata file found")
    else:
        print("   ❌ FAISS directory not found")
    
    print(f"\n🎯 SUMMARY FOR STREAMLIT (port 8501):")
    print("=" * 40)
    
    total_videos = 0
    total_chunks = 0
    
    # Count formal presentations
    if formal_faiss_dir.exists():
        latest_run = max([d for d in formal_faiss_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        metadata_file = latest_run / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            total_chunks += len(metadata)
            video_counts = {}
            for entry in metadata:
                if isinstance(entry, dict):
                    video_id = entry.get('video_id', 'unknown')
                    video_counts[video_id] = video_counts.get(video_id, 0) + 1
            total_videos += len(video_counts)
    
    # Count conversations
    if conversations_faiss_dir.exists():
        latest_run = max([d for d in conversations_faiss_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        metadata_file = latest_run / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            total_chunks += len(metadata)
            video_counts = {}
            for entry in metadata:
                if isinstance(entry, dict):
                    video_id = entry.get('video_id', 'unknown')
                    video_counts[video_id] = video_counts.get(video_id, 0) + 1
            total_videos += len(video_counts)
    
    print(f"   Total videos indexed: {total_videos}")
    print(f"   Total chunks indexed: {total_chunks}")
    print(f"   Streamlit should be able to search across all these chunks")

if __name__ == "__main__":
    check_streamlit_indices()
