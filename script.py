from pathlib import Path
from src.ingestion.document_loader import load_documents
from src.preprocessing.chunker_markdown import chunk_document
from src.embedding.embeddings import EmbeddingModel
from src.vector_store.FAISS_store import VectorStore
from src.mlops.tracking import *
import json

DATA_PATH = Path("C:/Users/Lenovo/Desktop/ISS_lectures/parctical-data-security-lec-1.pdf")

start_experiment("paraphrase-multilingual-mpnet-base-v2 exeriment")

documents = load_documents(DATA_PATH)

chunk_size = 400
overlap = 50
all_chunks = []
all_metadatas = []

i = 0
for doc in documents :
    page_chunks = chunk_document(doc["text"], chunk_size, overlap)
    
    for chunk in page_chunks:
        all_chunks.append(chunk)
        all_metadatas.append({
            "source": doc["metadata"]["source"],
            "page": doc["metadata"]["page"],
            "text": chunk,
            "chunk_id": i
        })
        i+=1
    
model_name = "paraphrase-multilingual-MiniLM-L12-v2"

embedder = EmbeddingModel()
vectors = embedder.encode(all_chunks)

store = VectorStore(dim=vectors.shape[1])


store.add(vectors, all_metadatas)

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(store.metadata, f, ensure_ascii=False, indent=2)

log_params_dict({
    "chunk_size": chunk_size,
    "overlap": overlap,
    "embedding_model": model_name
})

log_metrics_dict({
    "num_chunks": len(all_chunks),
    "embedding_dim": vectors.shape[1]
})

log_faiss_index(store.index, Path("faiss.index"))

log_artifact_metadatas("metadata.json")


end_run()


from groq import Groq

client = Groq(api_key="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")

print(client.models.list())
