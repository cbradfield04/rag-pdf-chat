import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://rag:ragpass@localhost:5432/ragdb")

engine = create_engine(DB_URL, future=True)

SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id SERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  source TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
  id SERIAL PRIMARY KEY,
  document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR(3072) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
-- pgvector index (optional for small data; helpful later)
-- CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

with engine.begin() as conn:
    conn.execute(text(SQL))

print("DB initialized (documents, chunks, vector extension).")