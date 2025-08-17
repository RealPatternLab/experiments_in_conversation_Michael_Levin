#!/usr/bin/env python3
"""
Simple script to update the progress queue for step 5 completion
"""

from pipeline_progress_queue import get_progress_queue
from datetime import datetime

def main():
    progress_queue = get_progress_queue()
    
    # Update both videos to mark step 5 as completed
    videos = ["mOaSHpTOwu0", "FzFFeRVEdUM"]
    
    for video_id in videos:
        progress_queue.update_video_step_status(
            video_id,
            'step_05_frame_extraction',
            'completed',
            metadata={
                'frames_summary_file': f"step_05_frames/{video_id}_frames_summary.json",
                'video_frames_dir': f"step_05_frames/{video_id}/",
                'status': 'already_processed',
                'completed_at': datetime.now().isoformat()
            }
        )
        print(f"✅ Updated {video_id} step_05_frame_extraction: completed")
    
    print("🎉 Progress queue updated for step 5!")

if __name__ == "__main__":
    main()
