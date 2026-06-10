# 🏥 Healthcare RAG AI Assistant - Setup & Usage Guide

## 📋 Quick Start

### 1. Get Your Hugging Face API Token
1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Give it a name: `healthcare-rag`
4. Select **"Read"** permission
5. Click **"Generate"**
6. Copy the token (looks like: `hf_xxxxxxxxxxxxxxxxxxxx`)

### 2. Configure Your .env File
Edit `.env` and replace the placeholder:
```env
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

### 3. Start the Backend API
```bash
# Terminal 1: Start FastAPI server
uvicorn app.main:app --reload
```
✅ Wait for: `Uvicorn running on http://127.0.0.1:8000`

### 4. Start the Streamlit UI
```bash
# Terminal 2: Start Streamlit interface
streamlit run app/ui.py
```
✅ Browser opens automatically at `http://localhost:8501`

### 5. Index Documents
1. In Streamlit sidebar → **"📥 Knowledge Ingestion"**
2. Click **"🔄 Index Documents"**
3. Wait for success message
4. You should see: `✅ Indexed X chunks from Y documents`

### 6. Ask Questions
Now in the main chat area, ask healthcare questions like:
- "What is hypertension?"
- "What are symptoms of diabetes?"
- "How is depression treated?"

---

## 🧪 Test the System Without Streamlit

Run this command to verify everything works:
```bash
python test_system.py
```

This will:
1. ✅ Check if API is running
2. ✅ Check knowledge base status
3. ✅ Ingest documents if needed
4. ✅ Test with sample questions

---

## 📊 API Endpoints Reference

### Check API Health
```bash
# PowerShell
$response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health"
$response.Content
```

### Get Knowledge Base Stats
```bash
# PowerShell
$response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/stats"
$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Ingest Documents
```bash
# PowerShell
$body = @{} | ConvertTo-Json
$response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/ingest" -Method Post `
  -Headers @{"Content-Type"="application/json"} -Body $body
$response.Content
```

### Ask a Question
```bash
# PowerShell
$body = @{ question = "What is hypertension?" } | ConvertTo-Json
$response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/ask" -Method Post `
  -Headers @{"Content-Type"="application/json"} -Body $body
$response.Content
```

---

## 📂 Project Structure

```
RAM/
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── config.py        # Configuration settings
│   ├── llm.py           # LLM integration (OpenAI/Hugging Face)
│   ├── embeddings.py    # Vector store (ChromaDB)
│   ├── rag.py           # Document loading & chunking
│   ├── ui.py            # Streamlit interface
│   └── __init__.py
├── data/
│   ├── document_1.txt   # Healthcare conditions (hypertension, diabetes, heart disease)
│   ├── document_2.txt   # Mental health & other conditions (depression, anxiety, asthma)
│   └── mplus_topics_2026-06-06.xml  # Optional: Real MedlinePlus data
├── vector_store/        # ChromaDB vector database (auto-created)
├── .env                 # Configuration (HUGGINGFACEHUB_API_TOKEN)
├── .env.example         # Configuration template
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── README.md            # Project overview
├── SETUP.md             # This file
└── test_system.py       # Test script
```

---

## 🔧 Troubleshooting

### Issue: "Backend is offline" in Streamlit
**Solution:**
- Make sure API is running: `uvicorn app.main:app --reload`
- Check API health: `http://127.0.0.1:8000/health`
- Verify network settings in Streamlit sidebar

### Issue: "I could not find information in the provided documents"
**Solution:**
- Documents aren't indexed yet
- Click **"🔄 Index Documents"** in Streamlit
- Wait for success message
- Try asking again

### Issue: HUGGINGFACEHUB_API_TOKEN validation error
**Solution:**
- Get a free token from: https://huggingface.co/settings/tokens
- Replace `your_hf_token_here` with actual token
- Restart the API: `uvicorn app.main:app --reload`

### Issue: "ModuleNotFoundError" or "No module named"
**Solution:**
```bash
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Issue: Streamlit exits with code 1
**Solution:**
```bash
# Run with debug logging
streamlit run app/ui.py --logger.level=debug

# Or check for missing dependencies
pip install -r requirements.txt

# And restart
```

---

## 📖 Example Questions to Ask

### Cardiovascular Health
- "What is hypertension?"
- "What are risk factors for heart disease?"
- "How is high cholesterol treated?"
- "What are symptoms of a heart attack?"

### Diabetes
- "What is type 2 diabetes?"
- "What are symptoms of diabetes?"
- "How do you manage diabetes with diet?"
- "What medications treat diabetes?"

### Mental Health
- "What is depression?"
- "How is anxiety treated?"
- "What are panic attack symptoms?"
- "What therapy types are available?"

### Respiratory Health
- "What is asthma?"
- "How do you manage asthma?"
- "What triggers asthma attacks?"
- "What are asthma medications?"

### Rheumatologic
- "What is arthritis?"
- "What types of arthritis exist?"
- "How is arthritis treated?"
- "What are arthritis risk factors?"

---

## 🚀 Production Deployment

### Using Docker
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access API at: http://localhost:8000
# Access UI at: http://localhost:8501
```

### Important for Production
1. **Use a real Hugging Face token** with proper permissions
2. **Set proper logging level**: `LOG_LEVEL=WARNING`
3. **Use environment-specific .env files**
4. **Enable HTTPS** for API endpoints
5. **Restrict API access** with authentication
6. **Use persistent storage** for vector database
7. **Monitor logs** for errors and usage

---

## 📚 Document Format Support

The system supports ingesting:
- **Text files** (`.txt`)
- **Markdown files** (`.md`)
- **PDF files** (`.pdf`)
- **MedlinePlus XML** (`.xml`)

Place documents in the `data/` folder and click **"🔄 Index Documents"**

---

## 🔒 Privacy & Safety

- **All answers are grounded** in the ingested documents
- **No external knowledge** is used
- **Chat history is local** to the browser
- **No data is sent to external servers** (except LLM API)
- **Documents stay private** in your vector database

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the README.md file
3. Check API logs: Look for error messages in the terminal running the API
4. Run `python test_system.py` to diagnose issues

---

## ✅ Verification Checklist

- [ ] Hugging Face token obtained and configured
- [ ] `.env` file updated with token
- [ ] API running: `uvicorn app.main:app --reload`
- [ ] API health check passes: `http://127.0.0.1:8000/health`
- [ ] Streamlit running: `streamlit run app/ui.py`
- [ ] Documents indexed: Shows > 0 chunks in stats
- [ ] Sample question returns valid answer
- [ ] Sources are displayed with answers
- [ ] Chat history works and persists in session

---

**Happy querying! 🎉**
