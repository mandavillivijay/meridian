"""
ModelAdapter Protocol — extension point for users who want to call a live model
during dataset construction rather than loading pre-populated JSON.

MERIDIAN itself does not depend on any LLM SDK. If you want to wire up a model,
implement this Protocol and pass it to DatasetSampler.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    def complete(self, prompt: str) -> str: ...
    def name(self) -> str: ...
