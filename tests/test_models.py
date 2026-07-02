import pytest
from pydantic import ValidationError

from meridian.models import (
    DriftResult,
    EquivalenceReport,
    IntentCategory,
    PromptRecord,
    TierVerdict,
)


# ---------------------------------------------------------------------------
# PromptRecord
# ---------------------------------------------------------------------------

class TestPromptRecord:
    def test_minimal_construction(self):
        r = PromptRecord(prompt="What is 2+2?", intent=IntentCategory.factual)
        assert r.prompt == "What is 2+2?"
        assert r.intent == IntentCategory.factual
        assert r.old_output is None
        assert r.new_output is None

    def test_full_construction(self):
        r = PromptRecord(
            prompt="Summarise this.",
            intent=IntentCategory.generative,
            old_output="old answer",
            new_output="new answer",
        )
        assert r.old_output == "old answer"
        assert r.new_output == "new answer"

    def test_intent_enum_values(self):
        for category in IntentCategory:
            r = PromptRecord(prompt="p", intent=category)
            assert r.intent == category

    def test_intent_accepts_string(self):
        r = PromptRecord(prompt="p", intent="classification")
        assert r.intent == IntentCategory.classification

    def test_missing_prompt_raises(self):
        with pytest.raises(ValidationError):
            PromptRecord(intent=IntentCategory.factual)

    def test_invalid_intent_raises(self):
        with pytest.raises(ValidationError):
            PromptRecord(prompt="p", intent="unsupported_intent")

    def test_outputs_are_mutable(self):
        r = PromptRecord(prompt="p", intent=IntentCategory.factual)
        r.old_output = "filled in later"
        assert r.old_output == "filled in later"


# ---------------------------------------------------------------------------
# DriftResult
# ---------------------------------------------------------------------------

class TestDriftResult:
    def _record(self) -> PromptRecord:
        return PromptRecord(
            prompt="Hello",
            intent=IntentCategory.factual,
            old_output="A",
            new_output="B",
        )

    def test_equivalent_verdict(self):
        dr = DriftResult(prompt_record=self._record(), similarity=0.95, verdict=TierVerdict.EQUIVALENT)
        assert dr.verdict == TierVerdict.EQUIVALENT
        assert dr.similarity == 0.95

    def test_review_verdict(self):
        dr = DriftResult(prompt_record=self._record(), similarity=0.80, verdict=TierVerdict.REVIEW)
        assert dr.verdict == TierVerdict.REVIEW

    def test_drifted_verdict(self):
        dr = DriftResult(prompt_record=self._record(), similarity=0.50, verdict=TierVerdict.DRIFTED)
        assert dr.verdict == TierVerdict.DRIFTED

    def test_similarity_boundary_zero(self):
        dr = DriftResult(prompt_record=self._record(), similarity=0.0, verdict=TierVerdict.DRIFTED)
        assert dr.similarity == 0.0

    def test_similarity_boundary_one(self):
        dr = DriftResult(prompt_record=self._record(), similarity=1.0, verdict=TierVerdict.EQUIVALENT)
        assert dr.similarity == 1.0

    def test_similarity_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            DriftResult(prompt_record=self._record(), similarity=1.1, verdict=TierVerdict.EQUIVALENT)

    def test_similarity_negative_raises(self):
        with pytest.raises(ValidationError):
            DriftResult(prompt_record=self._record(), similarity=-0.1, verdict=TierVerdict.DRIFTED)

    def test_is_immutable(self):
        dr = DriftResult(prompt_record=self._record(), similarity=0.9, verdict=TierVerdict.EQUIVALENT)
        with pytest.raises(ValidationError):
            dr.similarity = 0.5


# ---------------------------------------------------------------------------
# EquivalenceReport
# ---------------------------------------------------------------------------

class TestEquivalenceReport:
    def _valid_report(self, **overrides) -> dict:
        base = dict(
            total=100,
            equivalent_count=80,
            review_count=15,
            drifted_count=5,
            equivalent_pct=80.0,
            review_pct=15.0,
            drifted_pct=5.0,
            wilson_lower=0.71,
            wilson_upper=0.87,
            summary="80% equivalent, 15% review, 5% drifted",
        )
        base.update(overrides)
        return base

    def test_valid_construction(self):
        r = EquivalenceReport(**self._valid_report())
        assert r.total == 100
        assert r.equivalent_count == 80

    def test_counts_must_sum_to_total(self):
        with pytest.raises(ValidationError):
            EquivalenceReport(**self._valid_report(drifted_count=6))  # 80+15+6 = 101 ≠ 100

    def test_zero_total(self):
        r = EquivalenceReport(
            total=0,
            equivalent_count=0,
            review_count=0,
            drifted_count=0,
            equivalent_pct=0.0,
            review_pct=0.0,
            drifted_pct=0.0,
            wilson_lower=0.0,
            wilson_upper=0.0,
            summary="No results",
        )
        assert r.total == 0

    def test_pct_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            EquivalenceReport(**self._valid_report(equivalent_pct=101.0))

    def test_wilson_bounds_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            EquivalenceReport(**self._valid_report(wilson_upper=1.1))

    def test_summary_stored(self):
        r = EquivalenceReport(**self._valid_report(summary="All good"))
        assert r.summary == "All good"
