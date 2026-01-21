import faiss 
import numpy as np
from typing import List , Dict

class VectorStore:

    def __init__(self , dim : int):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata : List[Dict] = []

    def add(self , vectors: np.ndarray , metadatas:List[Dict]):
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_vector : np.ndarray , k :int = 5):
        scores, indices = self.index.search(query_vector , k)

        results = []

        for idx , score in zip(indices[0], scores[0]):
            if idx == -1 :
                continue
            meta = self.metadata[idx]

            results.append({
                "score": float(score),
                "text": meta["text"],           
                "citation": {
                    "source": meta["source"],
                    "page": meta["page"],
                }
            })
        return results

