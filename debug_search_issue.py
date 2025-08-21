#!/usr/bin/env python3
"""
Debug script to investigate why 1qRIetbuoH4 appears in every search result
"""

import pickle
import numpy as np
from pathlib import Path

def debug_search_issue():
    """Debug why 1qRIetbuoH4 appears in every search result"""
    
    print("🐛 DEBUGGING SEARCH ISSUE: 1qRIetbuoH4 in every result")
    print("=" * 60)
    
    # Check the formal presentations FAISS index
    faiss_dir = Path("SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/step_07_faiss_embeddings")
    latest_run = max([d for d in faiss_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    
    print(f"🔍 Examining FAISS index: {latest_run.name}")
    
    # Load metadata
    metadata_file = latest_run / "metadata.pkl"
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"📊 Total entries: {len(metadata)}")
    
    # Analyze content distribution
    video_stats = {}
    content_lengths = []
    
    for entry in metadata:
        if isinstance(entry, dict):
            video_id = entry.get('video_id', 'unknown')
            text = entry.get('text', '')
            
            if video_id not in video_stats:
                video_stats[video_id] = {
                    'count': 0,
                    'total_length': 0,
                    'avg_length': 0,
                    'min_length': float('inf'),
                    'max_length': 0
                }
            
            video_stats[video_id]['count'] += 1
            video_stats[video_id]['total_length'] += len(text)
            video_stats[video_id]['min_length'] = min(video_stats[video_id]['min_length'], len(text))
            video_stats[video_id]['max_length'] = max(video_stats[video_id]['max_length'], len(text))
            
            content_lengths.append(len(text))
    
    print(f"\n📹 VIDEO STATISTICS:")
    print("-" * 30)
    
    for video_id, stats in sorted(video_stats.items()):
        avg_length = stats['total_length'] / stats['count'] if stats['count'] > 0 else 0
        print(f"   {video_id}:")
        print(f"     - Chunks: {stats['count']}")
        print(f"     - Avg content length: {avg_length:.1f} chars")
        print(f"     - Min length: {stats['min_length']} chars")
        print(f"     - Max length: {stats['max_length']} chars")
    
    print(f"\n📏 CONTENT LENGTH ANALYSIS:")
    print("-" * 30)
    print(f"   Total chunks: {len(content_lengths)}")
    print(f"   Average length: {np.mean(content_lengths):.1f} chars")
    print(f"   Median length: {np.median(content_lengths):.1f} chars")
    print(f"   Min length: {min(content_lengths)} chars")
    print(f"   Max length: {max(content_lengths)} chars")
    
    # Check for potential issues
    print(f"\n🔍 POTENTIAL ISSUES:")
    print("-" * 30)
    
    # Check if 1qRIetbuoH4 has unusually long content
    video_1q = video_stats.get('1qRIetbuoH4', {})
    if video_1q:
        avg_1q = video_1q['total_length'] / video_1q['count'] if video_1q['count'] > 0 else 0
        overall_avg = np.mean(content_lengths)
        
        if avg_1q > overall_avg * 1.5:
            print(f"   ⚠️ 1qRIetbuoH4 has unusually long content:")
            print(f"     - 1qRIetbuoH4 avg: {avg_1q:.1f} chars")
            print(f"     - Overall avg: {overall_avg:.1f} chars")
            print(f"     - Ratio: {avg_1q/overall_avg:.2f}x")
    
    # Check for empty or very short content
    short_chunks = [len for len in content_lengths if len < 50]
    if short_chunks:
        print(f"   ⚠️ Found {len(short_chunks)} chunks with <50 characters")
        print(f"     - This could cause search issues")
    
    # Check for duplicate content
    unique_texts = set()
    duplicate_count = 0
    for entry in metadata:
        if isinstance(entry, dict):
            text = entry.get('text', '').strip()
            if text in unique_texts:
                duplicate_count += 1
            else:
                unique_texts.add(text)
    
    if duplicate_count > 0:
        print(f"   ⚠️ Found {duplicate_count} duplicate text chunks")
    
    # Check embeddings file
    embeddings_file = latest_run / "embeddings.npy"
    if embeddings_file.exists():
        embeddings = np.load(embeddings_file)
        print(f"\n🧮 EMBEDDINGS ANALYSIS:")
        print("-" * 30)
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Embedding dimension: {embeddings.shape[1] if len(embeddings.shape) > 1 else 'N/A'}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)):
            print(f"   ⚠️ Found NaN values in embeddings")
        if np.any(np.isinf(embeddings)):
            print(f"   ⚠️ Found infinite values in embeddings")
        
        # Check embedding magnitudes
        magnitudes = np.linalg.norm(embeddings, axis=1)
        print(f"   Embedding magnitudes:")
        print(f"     - Min: {np.min(magnitudes):.4f}")
        print(f"     - Max: {np.max(magnitudes):.4f}")
        print(f"     - Mean: {np.mean(magnitudes):.4f}")
        print(f"     - Std: {np.std(magnitudes):.4f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    print(f"   1. Check if Streamlit is using the correct FAISS index")
    print(f"   2. Verify that search queries are being processed correctly")
    print(f"   3. Check if there's a caching issue in the Streamlit app")
    print(f"   4. Ensure the search algorithm is working properly")
    
    print(f"\n🌐 TO TEST:")
    print(f"   1. Try different types of queries in Streamlit")
    print(f"   2. Check if the issue persists after restarting Streamlit")
    print(f"   3. Look at the actual search results in the Streamlit logs")

if __name__ == "__main__":
    debug_search_issue()
