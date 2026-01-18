from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
import faiss
from contextlib import asynccontextmanager

from src.embedding.embeddings import EmbeddingModel
from src.vector_store.FAISS_store import VectorStore
from src.services.rag_service import RAGService

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
    app.state.rag_service = RAGService(app.state.store, app.state.embedder)
    yield

    pass

app = FastAPI(title="Lecture-Saver 3000 API", lifespan=lifespan)

# --- Validation ---
class ChatRequest(BaseModel):
    question: str

    
@app.post("/ask")
async def ask(request: ChatRequest):
    response = await app.state.rag_service.answer_question(request.question)
    
    if response["status"] == "error":
        status_code = 503 if response["error_type"] == "LLMGenerationError" else 500
        raise HTTPException(status_code=status_code, detail=response["message"])
    
    return response