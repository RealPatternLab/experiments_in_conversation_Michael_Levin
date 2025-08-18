#!/usr/bin/env python3
"""
Flask webhook server for AssemblyAI transcription notifications.

This server runs alongside your Streamlit app and receives webhook calls from AssemblyAI
when transcription completes. It updates the webhook storage file that your Streamlit app reads.

Usage:
    python webhook_server.py

The server will listen on port 5000 and can receive webhooks at:
    http://localhost:5000/webhook
"""

from flask import Flask, request, jsonify
import json
import os
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Webhook storage file (same as in streamlit_app.py)
WEBHOOK_STORAGE_FILE = "SCIENTIFIC_VIDEO_PIPELINE/formal_presentations_1_on_0/logs/assemblyai_webhooks.json"

def load_webhook_storage():
    """Load webhook storage from file."""
    try:
        if os.path.exists(WEBHOOK_STORAGE_FILE):
            with open(WEBHOOK_STORAGE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load webhook storage: {e}")
    return {"pending_transcriptions": {}, "completed_transcriptions": {}}

def save_webhook_storage(data):
    """Save webhook storage to file."""
    try:
        with open(WEBHOOK_STORAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Webhook storage updated: {len(data.get('completed_transcriptions', {}))} completed, {len(data.get('pending_transcriptions', {}))} pending")
    except Exception as e:
        logger.error(f"Failed to save webhook storage: {e}")

def handle_assemblyai_webhook(transcript_id: str, status: str, data: dict = None):
    """Handle webhook from AssemblyAI when transcription completes."""
    webhook_data = load_webhook_storage()
    
    if status == "completed":
        # Move from pending to completed
        if transcript_id in webhook_data["pending_transcriptions"]:
            pending_info = webhook_data["pending_transcriptions"].pop(transcript_id)
            webhook_data["completed_transcriptions"][transcript_id] = {
                **pending_info,
                "completed_at": datetime.now().isoformat(),
                "status": "completed"
            }
            logger.info(f"Transcription {transcript_id} completed for video {pending_info.get('video_id', 'unknown')}")
        else:
            # Direct completion notification
            webhook_data["completed_transcriptions"][transcript_id] = {
                "completed_at": datetime.now().isoformat(),
                "status": "completed",
                "data": data
            }
            logger.info(f"Transcription {transcript_id} completed")
    
    elif status == "error":
        # Mark as failed
        if transcript_id in webhook_data["pending_transcriptions"]:
            pending_info = webhook_data["pending_transcriptions"].pop(transcript_id)
            webhook_data["completed_transcriptions"][transcript_id] = {
                **pending_info,
                "failed_at": datetime.now().isoformat(),
                "status": "error",
                "error": data.get("error", "Unknown error") if data else "Unknown error"
            }
            logger.error(f"Transcription {transcript_id} failed for video {pending_info.get('video_id', 'unknown')}")
    
    save_webhook_storage(webhook_data)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Receive webhook from AssemblyAI."""
    try:
        # Get the webhook data
        data = request.get_json()
        
        if not data:
            logger.error("No JSON data received in webhook")
            return jsonify({"error": "No JSON data"}), 400
        
        # Extract required fields
        transcript_id = data.get('transcript_id')
        status = data.get('status')
        
        if not transcript_id or not status:
            logger.error(f"Missing required fields in webhook: {data}")
            return jsonify({"error": "Missing transcript_id or status"}), 400
        
        # Log the webhook
        logger.info(f"Received webhook: transcript_id={transcript_id}, status={status}")
        
        # Handle the webhook
        handle_assemblyai_webhook(transcript_id, status, data)
        
        # Return success
        return jsonify({"status": "success", "message": f"Webhook processed for {transcript_id}"}), 200
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/webhooks', methods=['GET'])
def list_webhooks():
    """List all webhook data (for debugging)."""
    try:
        webhook_data = load_webhook_storage()
        return jsonify(webhook_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting AssemblyAI webhook server...")
    logger.info("Webhook endpoint: http://localhost:5001/webhook")
    logger.info("Health check: http://localhost:5001/health")
    logger.info("Webhook data: http://localhost:5001/webhooks")
    
    # Create webhook storage file if it doesn't exist
    if not os.path.exists(WEBHOOK_STORAGE_FILE):
        initial_data = {"pending_transcriptions": {}, "completed_transcriptions": {}}
        save_webhook_storage(initial_data)
        logger.info(f"Created initial webhook storage file: {WEBHOOK_STORAGE_FILE}")
    
    # Run the server
    app.run(host='0.0.0.0', port=5001, debug=True)
