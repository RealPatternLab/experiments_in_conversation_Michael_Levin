#!/usr/bin/env python3
"""
Simple script to update the progress queue for step 7 completion
"""

from pipeline_progress_queue import get_progress_queue
from datetime import datetime

def main():
    progress_queue = get_progress_queue()
    
    # Update both videos to mark step 7 as completed
    videos = ["mOaSHpTOwu0", "FzFFeRVEdUM"]
    
    for video_id in videos:
        progress_queue.update_video_step_status(
            video_id,
            'step_07_consolidated_embedding',
            'completed',
            metadata={
                'timestamp': '20250817_120444_368',
                'content_hash': 'abe49116',
                'embedding_files': {
                    'text_index': 'text_index_20250817_120444_368.index',
                    'visual_index': 'visual_index_20250817_120444_368.index',
                    'combined_index': 'combined_index_20250817_120444_368.index',
                    'embeddings': 'chunks_embeddings_20250817_120444_368.npy',
                    'metadata': 'chunks_metadata_20250817_120444_368.pkl'
                },
                'status': 'already_processed',
                'completed_at': datetime.now().isoformat()
            }
        )
        print(f"✅ Updated {video_id} step_07_consolidated_embedding: completed")
    
    print("🎉 Progress queue updated for step 7!")

if __name__ == "__main__":
    main()
