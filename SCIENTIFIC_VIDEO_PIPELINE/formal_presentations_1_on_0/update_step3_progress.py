#!/usr/bin/env python3
"""
Simple script to update the progress queue for step 3 completion
"""

from pipeline_progress_queue import get_progress_queue
from datetime import datetime

def main():
    progress_queue = get_progress_queue()
    
    # Update both videos to mark step 3 as completed
    videos = ["mOaSHpTOwu0", "FzFFeRVEdUM"]
    
    for video_id in videos:
        progress_queue.update_video_step_status(
            video_id,
            'step_03_transcription',
            'completed',
            metadata={
                'transcript_file': f"step_03_transcription/{video_id}_transcript.json",
                'text_file': f"step_03_transcription/{video_id}_transcript.txt",
                'completed_at': datetime.now().isoformat(),
                'status': 'already_processed'
            }
        )
        print(f"✅ Updated {video_id} step_03_transcription: completed")
    
    print("🎉 Progress queue updated for step 3!")

if __name__ == "__main__":
    main()
