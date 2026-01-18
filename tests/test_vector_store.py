import numpy as np
from vector_store.FAISS_store import VectorStore

def test_vector_store_search():
    store = VectorStore(dim=3)

    vectors = np.array([[1,0,0],[0,1,0]], dtype="float32")
    metas = [{"id": 1}, {"id": 2}]

    store.add(vectors, metas)

    query = np.array([[1,0,0]], dtype="float32")
    results = store.search(query, k=1)

    assert results[0]["metadata"]["id"] == 1
