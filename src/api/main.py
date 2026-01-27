from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
import faiss
import logging
from contextlib import asynccontextmanager

from src.embedding.embeddings import EmbeddingModel
from src.vector_store.FAISS_store import VectorStore
from src.services.rag_service import RAGService

logger = logging.getLogger("rag-api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting RAG API...")

        faiss_path = Path("faiss.index")
        metadata_path = Path("metadata.json")

        if not faiss_path.exists() or not metadata_path.exists():
            logger.warning("FAISS or metadata not found. API will start in degraded mode.")
            app.state.rag_service = None
            yield
            return

        embedder = EmbeddingModel()

        index = faiss.read_index(str(faiss_path))
        store = VectorStore(dim=index.d)
        store.index = index

        with open(metadata_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)

        app.state.store = store
        app.state.rag_service = RAGService(store, embedder)

        logger.info("RAG API ready.")
        yield

    except Exception as e:
        logger.exception("Startup failed")
        raise e


app = FastAPI(
    title="Lecture-Saver 3000 API",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_loaded": app.state.rag_service is not None
    }


@app.post("/ask")
async def ask(request: ChatRequest):
    if app.state.rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized"
        )

    try:
        response = await app.state.rag_service.answer_question(request.question)
    except Exception as e:
        logger.exception("Unhandled RAG error")
        raise HTTPException(status_code=500, detail="Internal RAG failure")

    if response.get("status") == "error":
        status_code = 503 if response.get("error_type") == "LLMGenerationError" else 500
        raise HTTPException(status_code=status_code, detail=response.get("message"))

    return response
