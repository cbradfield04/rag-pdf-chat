# RAG Doc Chat (OpenAI + FastAPI + pgvector)

A Retrieval-Augmented Generation (RAG) API that lets you upload PDFs, stores chunk embeddings in Postgres (pgvector), and answers questions with citations + snippets.

## Features
- Upload PDFs and auto-ingest text into chunk embeddings (`POST /upload_pdf`)
- Vector similarity search with pgvector (cosine distance)
- Ask questions with citation-grounded answers (`POST /ask`)
- Filter retrieval to a single document via `document_id`
- Document management:
  - list docs (`GET /documents`)
  - fetch a doc (`GET /documents/{id}`)
  - delete a doc (`DELETE /documents/{id}`)
- Swagger UI docs at `/docs`

## Architecture

```
Client (Swagger/UI)
      |
      v
FastAPI API
  - PDF upload (PyMuPDF)
  - text chunking
  - embeddings (OpenAI)
  - retrieval (pgvector)
  - answer generation (OpenAI)
      |
      v
Postgres + pgvector
  - documents (metadata)
  - chunks (text + embedding)
```

## Requirements
- Python 3.12+
- Docker Desktop (for Postgres + pgvector)
- OpenAI API key with billing enabled

## Setup

### 1) Clone and create a virtualenv
```bash
git clone <YOUR_REPO_URL>
cd rag-fastapi-openai

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3) Create `.env`
Create a `.env` file in the project root:

```txt
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+psycopg2://rag:ragpass@localhost:5432/ragdb
```

### 4) Start Postgres + pgvector
```bash
docker compose up -d
python init_db.py
```

### 5) Run the API
```bash
python -m uvicorn main:app --reload
```

Open:
- Home: http://127.0.0.1:8000/
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Quick Demo (Swagger)

### 1) Upload a PDF
Use `POST /upload_pdf`:
- Choose a `.pdf`
- Set title (optional)
- Execute

You should get a response like:
```json
{
  "ok": true,
  "document_id": 2,
  "chunks_indexed": 14,
  "filename": "example.pdf"
}
```

### 2) List documents
`GET /documents` to see all uploaded documents and their ids.

### 3) Ask a question (with document filter)
Use `POST /ask`:

```json
{
  "question": "Summarize this document in 3 bullets.",
  "top_k": 5,
  "document_id": 2
}
```

Response includes citations with snippets:
```json
{
  "answer": "... [123] ...",
  "citations": [
    {
      "chunk_id": 123,
      "source": "upload:example.pdf",
      "snippet": "FastAPI is a Python framework ..."
    }
  ]
}
```

## Endpoints

### Health
- `GET /health`  
Returns counts of documents and chunks.

### Documents
- `GET /documents`  
List documents with chunk counts.
- `GET /documents/{id}`  
Get metadata + chunk count for a single document.
- `DELETE /documents/{id}`  
Delete a document (cascades its chunks).

### Ingestion
- `POST /ingest`  
Manually ingest chunks (useful for debugging).
- `POST /upload_pdf`  
Upload a PDF and auto-ingest.

### QA
- `POST /ask`  
Ask a question using retrieved context. Supports optional `document_id` filter.

## Development Notes
- Embeddings model: `text-embedding-3-large` (3072 dims)
- Generation model: `gpt-4o-mini`
- Chunking defaults:
  - chunk_size: 1200 characters
  - overlap: 200 characters

## Troubleshooting

### `/health` returns 500
- Confirm Docker is running: `docker ps`
- Confirm `DATABASE_URL` is correct in `.env`
- Re-run: `python init_db.py`

### `insufficient_quota` (OpenAI 429)
- Enable billing / add credits on your OpenAI account
- Confirm the API key belongs to the funded project/org

### PDF upload returns “no extractable text”
- The PDF may be scanned/images only.
- Try a text-based PDF (syllabus, article, etc.)

## License
MIT 
