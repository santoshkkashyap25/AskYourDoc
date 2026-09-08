<div align="center">

# AskYourDoc

**Retrieval-Augmented Generation (RAG) platform for PDF documents.**  
Upload PDFs, ask questions, and receive grounded answers with exact page citations and live token streaming.

[**🌐 Live Application: https://askyourdoc-s0p9.onrender.com**](https://askyourdoc-s0p9.onrender.com)

<br/>

[![Live Demo](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=flat-square&logo=render)](https://askyourdoc-s0p9.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-blue?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Free%20Tier-orange?style=flat-square)](https://console.groq.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## Features

- **Real-Time Token Streaming**: Streams answers token-by-token using FastAPI Server-Sent Events (SSE) and Groq.
- **Page Citations**: Returns the exact PDF page number, excerpt, and similarity match score for every source chunk.
- **LangGraph Pipeline**: StateGraph workflow that checks cache, retrieves relevant chunks, formats context, and calls the LLM.
- **SQLite Caching**: Persistent disk cache for query answers and embeddings to avoid duplicate LLM calls and speed up repeated queries.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/santoshkkashyap25/AskYourDoc.git
cd AskYourDoc

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # On Windows
# source .venv/bin/activate       # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the example environment file and add your free [Groq API Key](https://console.groq.com/keys):

```bash
copy .env.example .env    # On Windows (or 'cp .env.example .env' on Linux/macOS)
```

Edit `.env`:
```env
GROQ_API_KEY=your_free_groq_api_key_here
```

### 3. Run Application

**Option A: Local Python**
```bash
python run.py
```

**Option B: Docker**
```bash
# Using Docker Compose (Recommended)
docker compose up --build

# Or standard Docker
docker build -t askyourdoc .
docker run -p 8000:8000 -v ${PWD}/data:/app/data --env-file .env askyourdoc
```

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## ⚙️ Configuration

All parameters can be customized in `.env`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `groq` | Generator backend (`groq`, `grok`, `openai`, `gemini`, `mock`) |
| `GROQ_API_KEY` | `""` | Free API key from [Groq Console](https://console.groq.com/keys) |
| `GROQ_MODEL_NAME` | `openai/gpt-oss-20b` | Model name (`llama-3.3-70b-versatile`, `qwen/qwen3.6-27b`) |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-Transformers embedding model |
| `CHUNK_SIZE` | `500` | Text chunk character length |
| `CHUNK_OVERLAP` | `50` | Character overlap between adjacent chunks |
| `RETRIEVER_TOP_K` | `4` | Number of context chunks retrieved per query |
| `MAX_DOCUMENTS` | `10` | Maximum number of simultaneous documents stored |
| `MAX_TOTAL_STORAGE_MB` | `25` | Maximum total cumulative PDF storage across all documents |
| `CACHE_TTL_SECONDS` | `86400` | SQLite cache expiration time (24 hours) |

---

## 🧪 Testing

```bash
# Run unit tests (splitters, SQLite cache, memory store, Groq adapter)
python -m unittest tests/test_rag.py

# Run end-to-end integration tests (PDF ingest, query, live stream)
python tests/test_e2e.py
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
