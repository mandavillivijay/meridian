"""
Local sentence-transformer embedder.

Runs entirely on-device — no API calls, no cloud dependency. This is a deliberate
design choice: unlike LLM-as-judge migration validators, MERIDIAN's scoring step
is free, fast, and deterministic.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Module-level cache: one loaded model per model name per process.
_instances: dict[str, "Embedder"] = {}


class Embedder:
    """Singleton-per-model-name wrapper around SentenceTransformer."""

    def __new__(cls, model_name: str = _DEFAULT_MODEL) -> "Embedder":
        if model_name not in _instances:
            instance = super().__new__(cls)
            instance._model_name = model_name
            instance._model = SentenceTransformer(model_name)
            _instances[model_name] = instance
        return _instances[model_name]

    def embed(self, text: Union[str, list[str]]) -> np.ndarray:
        """Return L2-normalised embeddings as a float32 numpy array.

        Single string → shape (dim,). List of strings → shape (n, dim).
        """
        single = isinstance(text, str)
        inputs = [text] if single else text
        if not inputs:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = self._model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True)
        return vectors[0] if single else vectors

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
