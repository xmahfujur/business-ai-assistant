# Business AI Assistant

A RAG-powered document Q&A assistant for business documents. Upload PDFs, CSVs, Word files, and more, then ask natural-language questions answered by Google Gemini with retrieved context.

## Features

- **Multi-format support** — PDF, TXT, CSV, DOCX, PNG, JPG
- **Persistent vector store** — ChromaDB index survives restarts (no full re-index on every launch)
- **Source citations** — See which document chunks informed each answer
- **Streamlit UI** — Chat interface with sidebar upload and index management
- **FastAPI backend** — REST endpoints for upload and chat

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure API key

```bash
copy .env.example .env
```

Add your [Google Gemini API key](https://aistudio.google.com/apikey) to `.env`:

```
GEMINI_API_KEY=your_key_here
```

### 3. Run the app

**Streamlit (recommended UI):**

```bash
streamlit run streamlit_app.py
```

**FastAPI backend:**

```bash
uvicorn main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## Usage

1. Upload documents via the sidebar (Streamlit) or `POST /uploads` (API)
2. Click **Save & Index Files** (Streamlit auto-indexes on API upload)
3. Ask questions in the chat

### API Examples

```bash
# Upload a document
curl -X POST http://127.0.0.1:8000/uploads -F "file=@report.pdf"

# Ask a question
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'
```

## Project Structure

```
├── streamlit_app.py      # Chat UI
├── main.py               # FastAPI REST API
├── src/
│   ├── config/config.py  # Environment settings
│   └── rag/
│       ├── loader.py     # Document loading
│       ├── chunker.py    # Text splitting
│       ├── vector_store.py  # ChromaDB indexing
│       └── engine.py     # Query engine (lazy load)
└── src/uploads/          # Uploaded documents
```

## License

GNU AGPL v3 — see [LICENSE](LICENSE).
