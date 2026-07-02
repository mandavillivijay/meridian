import numpy as np
import pytest

from meridian.embedder import Embedder, _instances


@pytest.fixture(autouse=True)
def clear_singleton_cache():
    """Each test gets a clean singleton cache to avoid cross-test state."""
    _instances.clear()
    yield
    _instances.clear()


class TestEmbedderSingleton:
    def test_same_model_name_returns_same_instance(self):
        a = Embedder()
        b = Embedder()
        assert a is b

    def test_different_model_names_return_different_instances(self):
        a = Embedder("all-MiniLM-L6-v2")
        b = Embedder("all-MiniLM-L12-v2")
        assert a is not b

    def test_model_name_property(self):
        e = Embedder("all-MiniLM-L6-v2")
        assert e.model_name == "all-MiniLM-L6-v2"


class TestEmbedderOutput:
    def test_single_string_returns_1d_array(self):
        e = Embedder()
        vec = e.embed("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1

    def test_list_of_strings_returns_2d_array(self):
        e = Embedder()
        vecs = e.embed(["hello", "world", "foo"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.ndim == 2
        assert vecs.shape[0] == 3

    def test_embedding_dim_consistent(self):
        e = Embedder()
        vec = e.embed("test")
        assert vec.shape[0] == e.embedding_dim

    def test_batch_embedding_dim_consistent(self):
        e = Embedder()
        vecs = e.embed(["a", "b"])
        assert vecs.shape[1] == e.embedding_dim

    def test_embeddings_are_l2_normalised_single(self):
        e = Embedder()
        vec = e.embed("normalisation check")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_embeddings_are_l2_normalised_batch(self):
        e = Embedder()
        vecs = e.embed(["first sentence", "second sentence"])
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_dtype_is_float32(self):
        e = Embedder()
        vec = e.embed("type check")
        assert vec.dtype == np.float32

    def test_identical_texts_produce_identical_embeddings(self):
        e = Embedder()
        v1 = e.embed("the quick brown fox")
        v2 = e.embed("the quick brown fox")
        assert np.allclose(v1, v2)

    def test_different_texts_produce_different_embeddings(self):
        e = Embedder()
        v1 = e.embed("the quick brown fox")
        v2 = e.embed("quantum electrodynamics")
        assert not np.allclose(v1, v2)

    def test_cosine_similarity_of_identical_texts_is_one(self):
        e = Embedder()
        v = e.embed("cosine similarity test")
        # vectors are already L2-normalised so dot product == cosine similarity
        sim = float(np.dot(v, v))
        assert abs(sim - 1.0) < 1e-5

    def test_similar_texts_have_higher_similarity_than_dissimilar(self):
        e = Embedder()
        base = e.embed("The capital of France is Paris.")
        similar = e.embed("Paris is the capital city of France.")
        dissimilar = e.embed("Stochastic gradient descent converges in expectation.")
        sim_high = float(np.dot(base, similar))
        sim_low = float(np.dot(base, dissimilar))
        assert sim_high > sim_low

    def test_empty_list_returns_empty_2d_array(self):
        e = Embedder()
        vecs = e.embed([])
        assert isinstance(vecs, np.ndarray)
        assert vecs.ndim == 2
        assert vecs.shape[0] == 0
