
from preprocessing.chunker import chunk_text

def test_chunk_shape():
    text = "word" * 1000
    chunks = chunk_text(text)

    assert len(chunks) > 0 

def test_chunk_dim():
    text = "word" * 1000
    chunks = chunk_text(text)
    assert len(chunks[0].split()) <= 400 

