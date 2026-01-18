from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json
import faiss
from contextlib import asynccontextmanager

from src.embedding.embeddings import EmbeddingModel
from src.vector_store.FAISS_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------- STARTUP --------
    embedding_model_name = "all-MiniLM-L6-v2"
    faiss_path = Path("faiss.index")
    metadata_path = Path("metadata.json")

    if not faiss_path.exists():
        raise RuntimeError("FAISS index not found")

    if not metadata_path.exists():
        raise RuntimeError("Metadata not found")

    app.state.embedder = EmbeddingModel(embedding_model_name)

    index = faiss.read_index(str(faiss_path))
    store = VectorStore(dim=index.d)
    store.index = index

    with open(metadata_path, "r", encoding="utf-8") as f:
        store.metadata = json.load(f)

    app.state.store = store

    yield

    # -------- SHUTDOWN --------
   
    pass


app = FastAPI(
    title="NLP RAG API",
    lifespan=lifespan
)


class Query(BaseModel):
    text: str
    k: int = 5


@app.post("/search")
def search(q: Query):
    vec = app.state.embedder.encode([q.text])
    results = app.state.store.search(vec, k=q.k)
    return {"results": results}
