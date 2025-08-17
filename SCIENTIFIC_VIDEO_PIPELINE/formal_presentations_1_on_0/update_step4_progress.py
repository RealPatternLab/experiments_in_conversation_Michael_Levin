#!/usr/bin/env python3
"""
Simple script to update the progress queue for step 4 completion
"""

from pipeline_progress_queue import get_progress_queue
from datetime import datetime

def main():
    progress_queue = get_progress_queue()
    
    # Update both videos to mark step 4 as completed
    videos = ["mOaSHpTOwu0", "FzFFeRVEdUM"]
    
    for video_id in videos:
        progress_queue.update_video_step_status(
            video_id,
            'step_04_semantic_chunking',
            'completed',
            metadata={
                'chunks_file': f"step_04_extract_chunks/{video_id}_chunks.json",
                'chunks_dir': f"step_04_extract_chunks/{video_id}_chunks/",
                'status': 'already_processed',
                'completed_at': datetime.now().isoformat()
            }
        )
        print(f"✅ Updated {video_id} step_04_semantic_chunking: completed")
    
    print("🎉 Progress queue updated for step 4!")

if __name__ == "__main__":
    main()
