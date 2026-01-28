from src.embedding.embeddings import EmbeddingModel
from src.generation.llm_client import LLMClient
import asyncio
import logging

logger = logging.getLogger("rag-service")

class RAGService:
    def __init__(self, store, embedder :EmbeddingModel):
        self.store = store
        self.embedder = embedder
        self.llm = LLMClient()

    async def answer_question(self, question: str, k: int = 5):
        
            if not question or len(question.strip()) == 0:
                 return {
                      "status": "error",
                      "message" : "Empty question"
                 }
            
            # Basic safety limit
            if len(question) > 2000:
                 return {
                      "status" : "error",
                      "message" : "Question too long"
                 }

            try:

                # --- Retrieval ---
                query_vector = self.embedder.encode([question])
                retrieved_docs = self.store.search(query_vector, k=k)

                if not retrieved_docs:
                
                    return {
                        "status": "success",
                        "answer": "Not found in the provided lecture material.",
                        "citations": {}
                    }

                # --- Generation ---
                context_chunks = [
                    f"[{i+1}] {doc['text']}"
                    for i, doc in enumerate(retrieved_docs)
                ]

                try:  
                    answer = await asyncio.wait_for(
                        self.llm.generate(question, context_chunks),
                        timeout=20
                    )
                except asyncio.TimeoutError:
                    logger.warning("LLM timout")
                    return {
                        "status" : "error",
                        "error_type": "LLMTimeout",
                        "message" : "LLM generation timed out"
                    }
            
                
                # --- Citations ---
                citations = {
                    str(i+1): doc["citation"]
                    for i, doc in enumerate(retrieved_docs)
                }


                # --- Logging ---
                
                return {
                    "status": "success",
                    "answer": answer,
                    "citations": citations
                }

            except Exception as e:
                
                logger.exception("RAG failure")
                return {
                    "status": "error",
                    "error_type": "RAGRuntimeError",
                    "message": "Internal RAG error"
                }
