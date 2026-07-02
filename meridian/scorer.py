"""
Drift scoring via cosine similarity on sentence-transformer embeddings.

This is the core of MERIDIAN's approach. Rather than calling a second LLM to judge
whether outputs match (expensive, non-deterministic, requires API access), we embed
both outputs locally and compute cosine similarity — fast, free, and reproducible.
"""

from __future__ import annotations

import numpy as np

from meridian.embedder import Embedder
from meridian.models import DriftResult, PromptRecord, TierVerdict

_DEFAULT_EQUIVALENT_THRESHOLD = 0.92
_DEFAULT_REVIEW_THRESHOLD = 0.75


class DriftScorer:
    """Score PromptRecords against the three-tier equivalence gate.

    Thresholds are inclusive on the upper tier:
        similarity >= equivalent_threshold  →  EQUIVALENT
        review_threshold <= sim < equivalent_threshold  →  REVIEW
        similarity < review_threshold  →  DRIFTED
    """

    def __init__(
        self,
        equivalent_threshold: float = _DEFAULT_EQUIVALENT_THRESHOLD,
        review_threshold: float = _DEFAULT_REVIEW_THRESHOLD,
        embedder: Embedder | None = None,
    ) -> None:
        if not (0.0 <= review_threshold < equivalent_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 <= review_threshold < equivalent_threshold <= 1; "
                f"got review={review_threshold}, equivalent={equivalent_threshold}"
            )
        self.equivalent_threshold = equivalent_threshold
        self.review_threshold = review_threshold
        self._embedder = embedder or Embedder()

    def score(self, record: PromptRecord) -> DriftResult:
        """Score a single PromptRecord. Both outputs must be populated."""
        if record.old_output is None or record.new_output is None:
            raise ValueError(
                f"Both old_output and new_output must be set before scoring. "
                f"Got old_output={record.old_output!r}, new_output={record.new_output!r}"
            )
        v_old = self._embedder.embed(record.old_output)
        v_new = self._embedder.embed(record.new_output)
        similarity = float(np.dot(v_old, v_new))
        similarity = max(0.0, min(1.0, similarity))  # clamp for floating-point safety
        return DriftResult(
            prompt_record=record,
            similarity=similarity,
            verdict=self._verdict(similarity),
        )

    def score_all(self, records: list[PromptRecord]) -> list[DriftResult]:
        """Score a batch of PromptRecords. Embeddings are computed in two bulk calls."""
        if not records:
            return []
        for r in records:
            if r.old_output is None or r.new_output is None:
                raise ValueError(
                    f"All records must have both outputs populated before scoring. "
                    f"Record with prompt {r.prompt!r:.60} has missing output."
                )
        old_vecs = self._embedder.embed([r.old_output for r in records])
        new_vecs = self._embedder.embed([r.new_output for r in records])
        results = []
        for record, v_old, v_new in zip(records, old_vecs, new_vecs):
            similarity = float(np.dot(v_old, v_new))
            similarity = max(0.0, min(1.0, similarity))
            results.append(DriftResult(
                prompt_record=record,
                similarity=similarity,
                verdict=self._verdict(similarity),
            ))
        return results

    def _verdict(self, similarity: float) -> TierVerdict:
        if similarity >= self.equivalent_threshold:
            return TierVerdict.EQUIVALENT
        if similarity >= self.review_threshold:
            return TierVerdict.REVIEW
        return TierVerdict.DRIFTED
