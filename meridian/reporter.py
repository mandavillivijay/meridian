"""
Aggregate drift results into a dataset-level verdict and write reports.

Produces both a machine-readable JSON file and a human-readable markdown summary.
The Wilson score confidence interval quantifies uncertainty on the EQUIVALENT
proportion — important when golden datasets are small (as is typical in enterprise
migration validation).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from meridian.models import DriftResult, EquivalenceReport, TierVerdict

_DEFAULT_REPORTS_DIR = Path("reports")
_WILSON_Z = 1.96  # 95 % confidence


def _wilson_interval(k: int, n: int, z: float = _WILSON_Z) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z / (1 + z2 / n)) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total * 100, 2)


def _build_summary(eq: int, rv: int, dr: int, total: int) -> str:
    if total == 0:
        return "No results to summarise."
    eq_pct = _pct(eq, total)
    rv_pct = _pct(rv, total)
    dr_pct = _pct(dr, total)
    parts = [f"{eq_pct}% of outputs are semantically equivalent"]
    if rv > 0:
        parts.append(f"{rv_pct}% show minor drift requiring human review")
    if dr > 0:
        parts.append(f"{dr_pct}% show significant drift (regression flagged)")
    return ", ".join(parts) + "."


class Reporter:
    def __init__(self, reports_dir: Path | str = _DEFAULT_REPORTS_DIR) -> None:
        self.reports_dir = Path(reports_dir)

    def build(self, results: list[DriftResult]) -> EquivalenceReport:
        """Aggregate a list of DriftResults into an EquivalenceReport."""
        total = len(results)
        eq = sum(1 for r in results if r.verdict == TierVerdict.EQUIVALENT)
        rv = sum(1 for r in results if r.verdict == TierVerdict.REVIEW)
        dr = sum(1 for r in results if r.verdict == TierVerdict.DRIFTED)
        lower, upper = _wilson_interval(eq, total)
        return EquivalenceReport(
            total=total,
            equivalent_count=eq,
            review_count=rv,
            drifted_count=dr,
            equivalent_pct=_pct(eq, total),
            review_pct=_pct(rv, total),
            drifted_pct=_pct(dr, total),
            wilson_lower=round(lower, 4),
            wilson_upper=round(upper, 4),
            summary=_build_summary(eq, rv, dr, total),
        )

    def write(
        self,
        report: EquivalenceReport,
        stem: Optional[str] = None,
    ) -> Path:
        """Write JSON + markdown reports to reports_dir. Returns the JSON path."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = stem or f"meridian_{timestamp}"

        json_path = self.reports_dir / f"{name}.json"
        md_path = self.reports_dir / f"{name}.md"

        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(_markdown(report), encoding="utf-8")

        return json_path


def _markdown(r: EquivalenceReport) -> str:
    return f"""# MERIDIAN Equivalence Report

## Verdict

{r.summary}

## Tier Breakdown

| Tier | Count | Percentage |
|------|------:|----------:|
| EQUIVALENT | {r.equivalent_count} | {r.equivalent_pct}% |
| REVIEW | {r.review_count} | {r.review_pct}% |
| DRIFTED | {r.drifted_count} | {r.drifted_pct}% |
| **Total** | **{r.total}** | |

## Confidence Interval

Wilson 95% CI on EQUIVALENT proportion: [{r.wilson_lower:.4f}, {r.wilson_upper:.4f}]

*MERIDIAN uses local sentence-transformer cosine similarity rather than LLM-as-judge
evaluation — scoring is free, fast, and fully reproducible.*
"""
