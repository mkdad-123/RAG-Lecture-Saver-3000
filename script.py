from pathlib import Path
from src.ingestion.document_loader import load_documents
from src.preprocessing.chunker import  chunk_text
from src.embedding.embeddings import EmbeddingModel
from src.vector_store.FAISS_store import VectorStore
from src.mlops.tracking import *
import json

DATA_PATH = Path("C:/Users/Lenovo/Desktop/Homework_ The Lecture-Saver 3000.pdf")

start_experiment("nlp_rag_pipeline")

documents = load_documents(DATA_PATH)

chunk_size = 400
overlap = 50
all_chunks = []
all_metadatas = []

i = 0
for doc in documents :
    page_chunks = chunk_text(doc['text'], chunk_size=chunk_size, overlap=overlap)
    
    for chunk in page_chunks:
        all_chunks.append(chunk)
        all_metadatas.append({
            "source": doc["metadata"]["source"],
            "page": doc["metadata"]["page"],
            "text": chunk,
            "chunk_id": i
        })
        i+=1
    

embedder = EmbeddingModel()
vectors = embedder.encode(all_chunks)

store = VectorStore(dim=vectors.shape[1])


store.add(vectors, all_metadatas)

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(store.metadata, f, ensure_ascii=False, indent=2)

log_params({
    "chunk_size": chunk_size,
    "overlap": overlap,
    "embedding_model": "all-MiniLM-L6-v2"
})

log_metrics({
    "num_chunks": len(all_chunks),
    "embedding_dim": vectors.shape[1]
})

log_faiss_index(store.index, Path("faiss.index"))

log_artifact_metadatas("metadata.json")


end_run()
