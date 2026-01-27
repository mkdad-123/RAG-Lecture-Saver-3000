from pathlib import Path
import json
from typing import Union, List

from ingestion.document_loader import load_documents
from preprocessing.chunker_markdown import chunk_document
from embedding.embeddings import EmbeddingModel
from vector_store.FAISS_store import VectorStore
from mlops.tracking import *
import faiss

def run_ingestion(
    data_path: Path,
    chunk_size: int = 400,
    overlap: int = 50,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
):


    start_experiment("rag_ingestion")

    documents = load_documents(data_path)

    all_chunks = []
    all_metadatas = []

    chunk_id = 0
    for doc in documents:
        chunks = chunk_document(
            doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": doc["metadata"]["source"],
                "page": doc["metadata"]["page"],
                "text": chunk,
                "chunk_id": chunk_id
            })
            chunk_id += 1

    embedder = EmbeddingModel()
    vectors = embedder.encode(all_chunks)

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, all_metadatas)

    
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(store.metadata, f, ensure_ascii=False, indent=2)

    faiss.write_index(store.index, "faiss.index")

    log_params_dict({   
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": model_name
    })

    log_metrics_dict({
        "num_chunks": len(all_chunks),
        "embedding_dim": vectors.shape[1]
    })

    log_artifact_metadatas("metadata.json")
    end_run()

    return {
        "num_docs": len(documents),
        "num_chunks": len(all_chunks)
    }
