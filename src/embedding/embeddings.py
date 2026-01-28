from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self , model_name:str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name)

    def encode(self , texts: List[str])-> np.ndarray:
        
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
