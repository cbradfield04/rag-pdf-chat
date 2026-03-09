import os
import re
from typing import List, Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key) if api_key else None

EMBED_MODEL = "text-embedding-3-large"
GEN_MODEL = "gpt-4o-mini"

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL not found in .env")

engine: Engine = create_engine(DB_URL, future=True)

app = FastAPI(
    title="RAG Doc Chat (OpenAI + FastAPI + pgvector)",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "none",
        "filter": True,
    },
)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>RAG Doc Chat API</h2>
    <ul>
      <li><a href="/docs">Swagger Docs</a></li>
      <li><a href="/health">Health</a></li>
      <li><a href="/documents">Documents</a></li>
    </ul>
    """


class IngestRequest(BaseModel):
    title: str = "Untitled"
    source: str
    chunks: List[str]


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    document_id: Optional[int] = None


def embed_texts(texts: List[str]) -> List[List[float]]:
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured.")
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in resp.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def snippet(text_value: str, limit: int = 220) -> str:
    clean = re.sub(r"\s+", " ", text_value).strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "…"


@app.get("/health")
def health():
    with engine.begin() as conn:
        docs = conn.execute(text("SELECT COUNT(*) FROM documents")).scalar_one()
        chunks = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()
    return {"ok": True, "documents": docs, "chunks": chunks}


@app.get("/documents")
def list_documents():
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    d.id,
                    d.title,
                    d.source,
                    d.created_at,
                    COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            )
        ).all()

    return [
        {
            "id": r[0],
            "title": r[1],
            "source": r[2],
            "created_at": str(r[3]),
            "chunk_count": int(r[4]),
        }
        for r in rows
    ]


@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    with engine.begin() as conn:
        doc = conn.execute(
            text(
                """
                SELECT id, title, source, created_at
                FROM documents
                WHERE id = :doc_id
                """
            ),
            {"doc_id": doc_id},
        ).first()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        chunk_count = conn.execute(
            text("SELECT COUNT(*) FROM chunks WHERE document_id = :doc_id"),
            {"doc_id": doc_id},
        ).scalar_one()

    return {
        "id": doc[0],
        "title": doc[1],
        "source": doc[2],
        "created_at": str(doc[3]),
        "chunk_count": int(chunk_count),
    }


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    with engine.begin() as conn:
        res = conn.execute(
            text("DELETE FROM documents WHERE id = :doc_id"),
            {"doc_id": doc_id},
        )
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found.")
    return {"ok": True, "deleted_document_id": doc_id}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not req.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided.")

    embeddings = embed_texts(req.chunks)

    with engine.begin() as conn:
        doc_id = conn.execute(
            text(
                "INSERT INTO documents (title, source) VALUES (:title, :source) RETURNING id"
            ),
            {"title": req.title, "source": req.source},
        ).scalar_one()

        for i, (chunk_text_, emb) in enumerate(zip(req.chunks, embeddings)):
            emb_str = "[" + ",".join(str(float(x)) for x in emb) + "]"
            conn.execute(
                text(
                    """
                    INSERT INTO chunks (document_id, chunk_index, text, embedding)
                    VALUES (:doc_id, :chunk_index, :text, CAST(:embedding AS vector))
                    """
                ),
                {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk_text_,
                    "embedding": emb_str,
                },
            )

    return {"ok": True, "document_id": doc_id, "chunks_indexed": len(req.chunks)}


@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    title: str = Form("Untitled PDF"),
    source: str = Form("upload"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    data = await file.read()

    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    pages_text = [page.get_text("text") for page in pdf]
    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise HTTPException(
            status_code=400,
            detail="PDF had no extractable text (it may be scanned/image-only).",
        )

    chunks = chunk_text(full_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")

    embeddings = embed_texts(chunks)

    with engine.begin() as conn:
        doc_id = conn.execute(
            text(
                "INSERT INTO documents (title, source) VALUES (:title, :source) RETURNING id"
            ),
            {"title": title, "source": f"{source}:{file.filename}"},
        ).scalar_one()

        for i, (chunk_text_, emb) in enumerate(zip(chunks, embeddings)):
            emb_str = "[" + ",".join(str(float(x)) for x in emb) + "]"
            conn.execute(
                text(
                    """
                    INSERT INTO chunks (document_id, chunk_index, text, embedding)
                    VALUES (:doc_id, :chunk_index, :text, CAST(:embedding AS vector))
                    """
                ),
                {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk_text_,
                    "embedding": emb_str,
                },
            )

    return {
        "ok": True,
        "document_id": doc_id,
        "chunks_indexed": len(chunks),
        "filename": file.filename,
    }


def retrieve(question: str, top_k: int, document_id: Optional[int] = None):
    q_emb = embed_texts([question])[0]
    q_emb_str = "[" + ",".join(str(float(x)) for x in q_emb) + "]"

    with engine.begin() as conn:
        if document_id is None:
            rows = conn.execute(
                text(
                    """
                    SELECT c.id, d.source, c.text
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    ORDER BY c.embedding <=> CAST(:qvec AS vector)
                    LIMIT :k
                    """
                ),
                {"qvec": q_emb_str, "k": top_k},
            ).all()
        else:
            rows = conn.execute(
                text(
                    """
                    SELECT c.id, d.source, c.text
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE d.id = :doc_id
                    ORDER BY c.embedding <=> CAST(:qvec AS vector)
                    LIMIT :k
                    """
                ),
                {"qvec": q_emb_str, "k": top_k, "doc_id": document_id},
            ).all()

    return [{"chunk_id": r[0], "source": r[1], "text": r[2]} for r in rows]


@app.post("/ask")
def ask(req: AskRequest):
    if req.top_k < 1 or req.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20.")

    hits = retrieve(req.question, req.top_k, req.document_id)
    if not hits:
        return {"answer": "I don't know. No chunks found.", "citations": []}

    context = "\n\n".join(
        [f"[{h['chunk_id']}] ({h['source']}) {h['text']}" for h in hits]
    )

    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context.\n"
        "Cite sources like [chunk_id] after the sentence they support.\n"
        "If the context is insufficient, say you don't know.\n\n"
        f"QUESTION:\n{req.question}\n\n"
        f"CONTEXT:\n{context}\n"
    )

    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured.")

    try:
        resp = client.responses.create(model=GEN_MODEL, input=prompt)
        answer_text = resp.output_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    citations = [
        {
            "chunk_id": h["chunk_id"],
            "source": h["source"],
            "snippet": snippet(h["text"]),
        }
        for h in hits
    ]

    return {"answer": answer_text, "citations": citations}