import numpy as np

from search.embedder import HashEmbedder, SentenceTransformerEmbedder


def test_dim_matches():
    e = HashEmbedder(dim=64)
    assert e.dim == 64


def test_encode_shape():
    e = HashEmbedder(dim=32)
    out = e.encode(["a", "b", "c"])
    assert out.shape == (3, 32)
    assert out.dtype == np.float32


def test_encode_normalised():
    e = HashEmbedder(dim=16)
    out = e.encode(["foo", "bar", "baz"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_deterministic():
    e = HashEmbedder(dim=16)
    a = e.encode(["hello"])
    b = e.encode(["hello"])
    np.testing.assert_array_equal(a, b)


def test_encode_different_inputs_yield_different_vectors():
    e = HashEmbedder(dim=16)
    a = e.encode(["foo"])
    b = e.encode(["bar"])
    assert not np.allclose(a, b)


def test_encode_empty_returns_empty_matrix():
    e = HashEmbedder(dim=16)
    out = e.encode([])
    assert out.shape == (0, 16)


class _FakeModel:
    def get_sentence_embedding_dimension(self):
        return 384


def test_sentence_transformer_dim_uses_real_api():
    # skip the real SentenceTransformer ctor, just check dim reads the right method
    e = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
    e._model = _FakeModel()
    assert e.dim == 384
