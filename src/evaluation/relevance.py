import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def context_relevance(query_embedding , chunk_embedding)-> float:

    
    sims = cosine_similarity(query_embedding , chunk_embedding)

    return float(np.mean(sims))
