#!/usr/bin/env python3
"""
Simple script to update the progress queue for step 6 completion
"""

from pipeline_progress_queue import get_progress_queue
from datetime import datetime

def main():
    progress_queue = get_progress_queue()
    
    # Update both videos to mark step 6 as completed
    videos = ["mOaSHpTOwu0", "FzFFeRVEdUM"]
    
    for video_id in videos:
        progress_queue.update_video_step_status(
            video_id,
            'step_06_frame_chunk_alignment',
            'completed',
            metadata={
                'alignments_file': f"{video_id}_alignments.json",
                'rag_ready_file': f"{video_id}_rag_ready_aligned_content.json",
                'summary_file': f"{video_id}_alignment_summary.json",
                'status': 'already_processed',
                'completed_at': datetime.now().isoformat()
            }
        )
        print(f"✅ Updated {video_id} step_06_frame_chunk_alignment: completed")
    
    print("🎉 Progress queue updated for step 6!")

if __name__ == "__main__":
    main()
