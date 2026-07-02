import json
import math
from pathlib import Path

import pytest

from meridian.models import DriftResult, IntentCategory, PromptRecord, TierVerdict
from meridian.reporter import Reporter, _wilson_interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(prompt: str = "p") -> PromptRecord:
    return PromptRecord(
        prompt=prompt,
        intent=IntentCategory.factual,
        old_output="old",
        new_output="new",
    )


def _result(verdict: TierVerdict, similarity: float = 0.9) -> DriftResult:
    return DriftResult(
        prompt_record=_record(),
        similarity=similarity,
        verdict=verdict,
    )


EQ = TierVerdict.EQUIVALENT
RV = TierVerdict.REVIEW
DR = TierVerdict.DRIFTED


# ---------------------------------------------------------------------------
# Wilson CI (pure function)
# ---------------------------------------------------------------------------

class TestWilsonInterval:
    def test_zero_total_returns_zeros(self):
        lo, hi = _wilson_interval(0, 0)
        assert lo == 0.0 and hi == 0.0

    def test_all_successes(self):
        lo, hi = _wilson_interval(100, 100)
        assert lo > 0.9
        assert math.isclose(hi, 1.0, abs_tol=1e-9)

    def test_no_successes(self):
        lo, hi = _wilson_interval(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_half_successes_symmetric(self):
        lo, hi = _wilson_interval(50, 100)
        assert abs((lo + hi) / 2 - 0.5) < 0.01  # centre near 0.5

    def test_bounds_are_in_unit_interval(self):
        for k, n in [(0, 1), (1, 1), (5, 10), (99, 100)]:
            lo, hi = _wilson_interval(k, n)
            assert 0.0 <= lo <= hi <= 1.0

    def test_wider_with_fewer_samples(self):
        lo10, hi10 = _wilson_interval(8, 10)
        lo100, hi100 = _wilson_interval(80, 100)
        assert (hi10 - lo10) > (hi100 - lo100)


# ---------------------------------------------------------------------------
# Reporter.build
# ---------------------------------------------------------------------------

class TestReporterBuild:
    def test_counts_all_tiers(self):
        results = [_result(EQ)] * 8 + [_result(RV)] * 1 + [_result(DR)] * 1
        r = Reporter().build(results)
        assert r.total == 10
        assert r.equivalent_count == 8
        assert r.review_count == 1
        assert r.drifted_count == 1

    def test_percentages(self):
        results = [_result(EQ)] * 3 + [_result(DR)] * 1
        r = Reporter().build(results)
        assert r.equivalent_pct == 75.0
        assert r.drifted_pct == 25.0
        assert r.review_pct == 0.0

    def test_empty_results(self):
        r = Reporter().build([])
        assert r.total == 0
        assert r.equivalent_count == 0
        assert r.wilson_lower == 0.0
        assert r.wilson_upper == 0.0

    def test_all_equivalent(self):
        results = [_result(EQ)] * 10
        r = Reporter().build(results)
        assert r.equivalent_pct == 100.0
        assert r.drifted_count == 0
        assert r.review_count == 0

    def test_all_drifted(self):
        results = [_result(DR)] * 5
        r = Reporter().build(results)
        assert r.drifted_pct == 100.0
        assert r.equivalent_count == 0

    def test_wilson_bounds_present(self):
        results = [_result(EQ)] * 80 + [_result(DR)] * 20
        r = Reporter().build(results)
        assert 0.0 < r.wilson_lower < r.wilson_upper <= 1.0

    def test_wilson_lower_less_than_upper(self):
        results = [_result(EQ)] * 50 + [_result(RV)] * 50
        r = Reporter().build(results)
        assert r.wilson_lower < r.wilson_upper

    def test_counts_sum_to_total(self):
        results = [_result(EQ)] * 7 + [_result(RV)] * 2 + [_result(DR)] * 1
        r = Reporter().build(results)
        assert r.equivalent_count + r.review_count + r.drifted_count == r.total


# ---------------------------------------------------------------------------
# Summary string
# ---------------------------------------------------------------------------

class TestSummaryString:
    def test_summary_mentions_equivalent_pct(self):
        results = [_result(EQ)] * 9 + [_result(DR)] * 1
        r = Reporter().build(results)
        assert "90.0%" in r.summary

    def test_summary_omits_zero_review(self):
        results = [_result(EQ)] * 5 + [_result(DR)] * 5
        r = Reporter().build(results)
        assert "review" not in r.summary.lower()

    def test_summary_omits_zero_drifted(self):
        results = [_result(EQ)] * 5 + [_result(RV)] * 5
        r = Reporter().build(results)
        assert "regression" not in r.summary.lower()

    def test_empty_summary_message(self):
        r = Reporter().build([])
        assert "No results" in r.summary

    def test_summary_ends_with_period(self):
        results = [_result(EQ)] * 3 + [_result(RV)] * 1 + [_result(DR)] * 1
        r = Reporter().build(results)
        assert r.summary.endswith(".")


# ---------------------------------------------------------------------------
# Reporter.write (file I/O)
# ---------------------------------------------------------------------------

class TestReporterWrite:
    def test_json_file_created(self, tmp_path):
        results = [_result(EQ)] * 4 + [_result(DR)] * 1
        reporter = Reporter(reports_dir=tmp_path)
        report = reporter.build(results)
        json_path = reporter.write(report, stem="test_run")
        assert json_path.exists()
        assert json_path.suffix == ".json"

    def test_markdown_file_created(self, tmp_path):
        reporter = Reporter(reports_dir=tmp_path)
        report = reporter.build([_result(EQ)] * 3)
        reporter.write(report, stem="test_run")
        md_path = tmp_path / "test_run.md"
        assert md_path.exists()

    def test_json_is_valid_and_contains_total(self, tmp_path):
        reporter = Reporter(reports_dir=tmp_path)
        report = reporter.build([_result(EQ)] * 5 + [_result(DR)] * 2)
        json_path = reporter.write(report, stem="test_run")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["total"] == 7
        assert data["equivalent_count"] == 5

    def test_markdown_contains_tier_table(self, tmp_path):
        reporter = Reporter(reports_dir=tmp_path)
        report = reporter.build([_result(EQ)] * 3 + [_result(RV)] * 1)
        reporter.write(report, stem="test_run")
        md = (tmp_path / "test_run.md").read_text(encoding="utf-8")
        assert "EQUIVALENT" in md
        assert "REVIEW" in md
        assert "DRIFTED" in md

    def test_reports_dir_created_if_missing(self, tmp_path):
        subdir = tmp_path / "nested" / "reports"
        reporter = Reporter(reports_dir=subdir)
        report = reporter.build([_result(EQ)])
        reporter.write(report, stem="x")
        assert subdir.exists()

    def test_auto_stem_generates_filename(self, tmp_path):
        reporter = Reporter(reports_dir=tmp_path)
        report = reporter.build([_result(EQ)])
        json_path = reporter.write(report)
        assert json_path.stem.startswith("meridian_")
