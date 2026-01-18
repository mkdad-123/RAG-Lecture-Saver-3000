from embedding.embeddings import EmbeddingModel

def test_embedding_shape():
    model = EmbeddingModel()
    vectors = model.encode(["hello world"])
    assert vectors.shape[0] == 1
    assert vectors.shape[1] > 0
