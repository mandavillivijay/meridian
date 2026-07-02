"""
Scorer tests use a MockEmbedder that returns controlled unit vectors so that
cosine similarity values are exact and no model needs to be loaded.

Given L2-normalised vectors, cosine sim == dot product. To produce a target
similarity s between two vectors, we use:
    v1 = [1, 0]
    v2 = [s, sqrt(1 - s^2)]
Their dot product is exactly s.
"""

import numpy as np
import pytest

from meridian.models import IntentCategory, PromptRecord, TierVerdict
from meridian.scorer import DriftScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_pair(similarity: float) -> tuple[np.ndarray, np.ndarray]:
    """Return two 2-D unit vectors whose dot product equals `similarity`."""
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([similarity, np.sqrt(max(0.0, 1.0 - similarity ** 2))], dtype=np.float32)
    return v1, v2


class MockEmbedder:
    """Returns pre-queued vectors in FIFO order, one per embed() call."""

    def __init__(self, vectors: list[np.ndarray]) -> None:
        self._queue = list(vectors)
        self.model_name = "mock"
        self.embedding_dim = 2

    def embed(self, text):
        if isinstance(text, str):
            return self._queue.pop(0)
        # list input: pop one vector per item
        return np.stack([self._queue.pop(0) for _ in text])


def _record(old: str = "old", new: str = "new") -> PromptRecord:
    return PromptRecord(
        prompt="test prompt",
        intent=IntentCategory.factual,
        old_output=old,
        new_output=new,
    )


def _scorer_with_sim(similarity: float) -> tuple[DriftScorer, PromptRecord]:
    """Return a scorer wired to produce the given similarity for a single record."""
    v1, v2 = _unit_pair(similarity)
    embedder = MockEmbedder([v1, v2])
    scorer = DriftScorer(embedder=embedder)
    return scorer, _record()


# ---------------------------------------------------------------------------
# Threshold construction
# ---------------------------------------------------------------------------

class TestDriftScorerInit:
    def test_defaults_accepted(self):
        # uses real embedder singleton — just checks it doesn't raise
        s = DriftScorer(embedder=MockEmbedder([]))
        assert s.equivalent_threshold == 0.92
        assert s.review_threshold == 0.75

    def test_custom_thresholds(self):
        s = DriftScorer(equivalent_threshold=0.85, review_threshold=0.60, embedder=MockEmbedder([]))
        assert s.equivalent_threshold == 0.85
        assert s.review_threshold == 0.60

    def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            DriftScorer(equivalent_threshold=0.70, review_threshold=0.80, embedder=MockEmbedder([]))

    def test_equal_thresholds_raise(self):
        with pytest.raises(ValueError):
            DriftScorer(equivalent_threshold=0.80, review_threshold=0.80, embedder=MockEmbedder([]))

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError):
            DriftScorer(equivalent_threshold=0.92, review_threshold=-0.1, embedder=MockEmbedder([]))

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError):
            DriftScorer(equivalent_threshold=1.1, review_threshold=0.75, embedder=MockEmbedder([]))


# ---------------------------------------------------------------------------
# Verdict assignment (three-tier gate)
# ---------------------------------------------------------------------------

class TestVerdictTiers:
    def test_above_equivalent_threshold_is_equivalent(self):
        scorer, record = _scorer_with_sim(0.95)
        result = scorer.score(record)
        assert result.verdict == TierVerdict.EQUIVALENT

    def test_at_equivalent_threshold_is_equivalent(self):
        scorer, record = _scorer_with_sim(0.92)
        result = scorer.score(record)
        assert result.verdict == TierVerdict.EQUIVALENT

    def test_just_below_equivalent_threshold_is_review(self):
        scorer, record = _scorer_with_sim(0.919)
        result = scorer.score(record)
        assert result.verdict == TierVerdict.REVIEW

    def test_midpoint_review_band_is_review(self):
        scorer, record = _scorer_with_sim(0.83)
        result = scorer.score(record)
        assert result.verdict == TierVerdict.REVIEW

    def test_at_review_threshold_is_review(self):
        scorer, record = _scorer_with_sim(0.75)
        result = scorer.score(record)
        assert result.verdict == TierVerdict.REVIEW

    def test_just_below_review_threshold_is_drifted(self):
        scorer, record = _scorer_with_sim(0.749)
        result = scorer.score(record)
        assert result.verdict == TierVerdict.DRIFTED

    def test_zero_similarity_is_drifted(self):
        scorer, record = _scorer_with_sim(0.0)
        result = scorer.score(record)
        assert result.verdict == TierVerdict.DRIFTED

    def test_perfect_similarity_is_equivalent(self):
        scorer, record = _scorer_with_sim(1.0)
        result = scorer.score(record)
        assert result.verdict == TierVerdict.EQUIVALENT


# ---------------------------------------------------------------------------
# DriftResult structure
# ---------------------------------------------------------------------------

class TestDriftResultStructure:
    def test_similarity_stored_on_result(self):
        scorer, record = _scorer_with_sim(0.95)
        result = scorer.score(record)
        assert abs(result.similarity - 0.95) < 1e-5

    def test_prompt_record_preserved(self):
        scorer, record = _scorer_with_sim(0.80)
        result = scorer.score(record)
        assert result.prompt_record is record

    def test_missing_old_output_raises(self):
        record = PromptRecord(prompt="p", intent=IntentCategory.factual, new_output="n")
        scorer = DriftScorer(embedder=MockEmbedder([]))
        with pytest.raises(ValueError, match="old_output"):
            scorer.score(record)

    def test_missing_new_output_raises(self):
        record = PromptRecord(prompt="p", intent=IntentCategory.factual, old_output="o")
        scorer = DriftScorer(embedder=MockEmbedder([]))
        with pytest.raises(ValueError, match="new_output"):
            scorer.score(record)

    def test_both_outputs_missing_raises(self):
        record = PromptRecord(prompt="p", intent=IntentCategory.factual)
        scorer = DriftScorer(embedder=MockEmbedder([]))
        with pytest.raises(ValueError):
            scorer.score(record)


# ---------------------------------------------------------------------------
# score_all
# ---------------------------------------------------------------------------

class TestScoreAll:
    def _multi_scorer(self, similarities: list[float]) -> tuple[DriftScorer, list[PromptRecord]]:
        """score_all uses two bulk embed calls: one for all old outputs, one for all new."""
        old_vecs = [_unit_pair(s)[0] for s in similarities]
        new_vecs = [_unit_pair(s)[1] for s in similarities]
        # score_all calls embed([old1, old2, ...]) then embed([new1, new2, ...])
        embedder = MockEmbedder(old_vecs + new_vecs)
        scorer = DriftScorer(embedder=embedder)
        records = [_record(old=f"old{i}", new=f"new{i}") for i in range(len(similarities))]
        return scorer, records

    def test_empty_list_returns_empty(self):
        scorer = DriftScorer(embedder=MockEmbedder([]))
        assert scorer.score_all([]) == []

    def test_returns_one_result_per_record(self):
        scorer, records = self._multi_scorer([0.95, 0.80, 0.60])
        results = scorer.score_all(records)
        assert len(results) == 3

    def test_correct_verdicts_in_batch(self):
        scorer, records = self._multi_scorer([0.95, 0.80, 0.60])
        results = scorer.score_all(records)
        assert results[0].verdict == TierVerdict.EQUIVALENT
        assert results[1].verdict == TierVerdict.REVIEW
        assert results[2].verdict == TierVerdict.DRIFTED

    def test_records_preserved_in_batch(self):
        scorer, records = self._multi_scorer([0.95, 0.80])
        results = scorer.score_all(records)
        assert results[0].prompt_record is records[0]
        assert results[1].prompt_record is records[1]

    def test_missing_output_in_batch_raises(self):
        record = PromptRecord(prompt="p", intent=IntentCategory.factual, old_output="o")
        scorer = DriftScorer(embedder=MockEmbedder([]))
        with pytest.raises(ValueError):
            scorer.score_all([record])

    def test_custom_thresholds_respected_in_batch(self):
        # With thresholds 0.85 / 0.65, sim=0.80 should be REVIEW
        v1, v2 = _unit_pair(0.80)
        embedder = MockEmbedder([v1, v2])
        scorer = DriftScorer(equivalent_threshold=0.85, review_threshold=0.65, embedder=embedder)
        results = scorer.score_all([_record()])
        assert results[0].verdict == TierVerdict.REVIEW
