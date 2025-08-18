# 🎯 AssemblyAI Webhook Setup Guide

This guide explains how to set up and use webhooks for AssemblyAI transcriptions, which solves the timeout issue for long videos.

## 🚀 What This Solves

**Before (Polling):**
- Pipeline waits for transcription to complete
- Long videos (67+ minutes) timeout
- Pipeline gets stuck on one video

**After (Webhooks):**
- Pipeline submits transcription and continues
- AssemblyAI calls your webhook when done
- No more timeouts - process multiple videos simultaneously

## 📋 Prerequisites

1. **AssemblyAI API Key** in your `.env` file
2. **Python packages** for the webhook server
3. **Streamlit app** deployed (for monitoring)

## 🔧 Setup Steps

### Step 1: Install Webhook Server Dependencies

```bash
# Install Flask and other requirements
uv pip install -r webhook_requirements.txt

# Or install individually:
uv pip install Flask==2.3.3 requests==2.31.0 python-dotenv==1.0.0
```

### Step 2: Configure Environment Variables

Add these to your `.env` file:

```bash
# AssemblyAI API Key (required)
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Webhook Configuration (optional - defaults provided)
WEBHOOK_URL=http://localhost:5000/webhook
WEBHOOK_AUTH_HEADER=X-Webhook-Secret
WEBHOOK_AUTH_VALUE=your-secret-value
```

### Step 3: Start the Webhook Server

```bash
# Start the webhook server
python webhook_server.py
```

**Expected Output:**
```
INFO - Starting AssemblyAI webhook server...
INFO - Webhook endpoint: http://localhost:5000/webhook
INFO - Health check: http://localhost:5000/health
INFO - Webhook data: http://localhost:5000/webhooks
```

### Step 4: Test the Webhook Server

Open a new terminal and test the endpoints:

```bash
# Health check
curl http://localhost:5000/health

# List webhooks (should be empty initially)
curl http://localhost:5000/webhooks
```

## 🎬 Using Webhooks in Your Pipeline

### Option A: Use the New Webhook-Based Transcription Step

```bash
# Instead of the old step_03_transcription.py
python step_03_transcription_webhook.py
```

**What happens:**
1. ✅ Extracts audio from videos
2. ✅ Submits transcription to AssemblyAI with webhook
3. ✅ Continues to next video immediately
4. ✅ No waiting, no timeouts

### Option B: Modify Your Existing Pipeline

Replace the old transcription step with the webhook version:

```bash
# Rename old file
mv step_03_transcription.py step_03_transcription_old.py

# Use new webhook version
mv step_03_transcription_webhook.py step_03_transcription.py
```

## 📊 Monitoring Progress

### Method 1: Streamlit App (Recommended)

1. **Open your Streamlit app**
2. **Navigate to "🎯 Webhook Status" page**
3. **Monitor pending and completed transcriptions**

### Method 2: Direct API Calls

```bash
# Check webhook status
curl http://localhost:5000/webhooks

# Health check
curl http://localhost:5000/health
```

### Method 3: Check Webhook Storage File

```bash
# View the JSON file directly
cat assemblyai_webhooks.json
```

## 🔄 Complete Workflow

### 1. Submit Transcriptions
```bash
python step_03_transcription_webhook.py
```

**Output:**
```
INFO - Found 2 videos to process
INFO - Processing video: N1KHs9hQ1V4 - Michael Levin Presentation
INFO - Audio extracted: step_03_transcription/N1KHs9hQ1V4_audio.wav
INFO - Audio uploaded: https://api.assemblyai.com/v2/upload/...
INFO - Transcription submitted with webhook: abc123-def456
INFO - Added abc123-def456 to pending transcriptions
INFO - Successfully submitted transcription for video: N1KHs9hQ1V4
```

### 2. Monitor Progress
- **Check Streamlit app** webhook status page
- **Watch webhook server logs** for incoming calls
- **Wait for AssemblyAI** to complete transcription

### 3. Retrieve Completed Transcripts
```bash
python retrieve_completed_transcripts.py
```

**Output:**
```
INFO - Found 1 completed transcriptions
INFO - Retrieving transcript abc123-def456 for video N1KHs9hQ1V4
INFO - Transcript saved: step_03_transcription/N1KHs9hQ1V4_transcript.json
INFO - Successfully processed transcript for N1KHs9hQ1V4
```

### 4. Continue Pipeline
```bash
# Now you can run step 04 with the completed transcripts
python step_04_extract_chunks.py
```

## 🌐 Production Deployment

### For Streamlit Cloud

Your Streamlit app already has webhook monitoring built-in. The webhook server needs to run separately.

### For Local Development

1. **Terminal 1:** Run webhook server
   ```bash
   python webhook_server.py
   ```

2. **Terminal 2:** Run your pipeline
   ```bash
   python step_03_transcription_webhook.py
   ```

3. **Terminal 3:** Monitor and retrieve
   ```bash
   python retrieve_completed_transcripts.py
   ```

## 🔍 Troubleshooting

### Webhook Server Won't Start

```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill process if needed
kill -9 <PID>

# Or use different port
export FLASK_RUN_PORT=5001
python webhook_server.py
```

### AssemblyAI Can't Reach Your Webhook

**Local Development:**
- Use ngrok to expose localhost:
  ```bash
  uv pip install pyngrok
  ngrok http 5000
  ```
- Update `WEBHOOK_URL` in `.env` with ngrok URL

**Production:**
- Deploy webhook server to cloud service
- Update `WEBHOOK_URL` with production URL

### Webhook Not Firing

1. **Check AssemblyAI logs** in your dashboard
2. **Verify webhook URL** is accessible
3. **Check authentication headers** match
4. **Test webhook endpoint** manually

## 📝 File Structure

```
your-project/
├── webhook_server.py              # Flask webhook server
├── webhook_requirements.txt       # Dependencies
├── assemblyai_webhooks.json      # Webhook storage (auto-created)
├── step_03_transcription_webhook.py  # New transcription step
├── retrieve_completed_transcripts.py  # Retrieve completed transcripts
├── streamlit_app.py              # Your app with webhook monitoring
└── .env                          # Environment variables
```

## 🎯 Benefits

- ✅ **No more timeouts** on long videos
- ✅ **Parallel processing** of multiple videos
- ✅ **Real-time monitoring** via Streamlit
- ✅ **Scalable architecture** for production
- ✅ **Automatic retry** on webhook failures
- ✅ **Comprehensive logging** for debugging

## 🚨 Important Notes

1. **Keep webhook server running** while processing videos
2. **Check webhook status** before running step 04
3. **Webhook server must be accessible** from AssemblyAI's servers
4. **Authentication headers** help secure your webhook endpoint
5. **Local development** requires ngrok or similar for external access

## 🆘 Need Help?

1. **Check webhook server logs** for errors
2. **Verify AssemblyAI API key** is valid
3. **Test webhook endpoint** manually
4. **Check Streamlit app** webhook status page
5. **Review this guide** for common issues

---

**🎉 Congratulations!** You now have a robust, scalable transcription system that won't timeout on long videos.
