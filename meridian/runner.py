"""
End-to-end pipeline: load dataset → score → report.

Single entry point for the most common workflow: hand MERIDIAN a pre-populated
JSON dataset and get back a verdict report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from meridian.embedder import Embedder
from meridian.models import EquivalenceReport
from meridian.reporter import Reporter
from meridian.sampler import load, stratified_sample
from meridian.scorer import DriftScorer


def run(
    dataset_path: Path | str,
    *,
    sample_n: Optional[int] = None,
    seed: Optional[int] = None,
    equivalent_threshold: float = 0.92,
    review_threshold: float = 0.75,
    embedder_model: str = "all-MiniLM-L6-v2",
    reports_dir: Path | str = "reports",
    report_stem: Optional[str] = None,
    write_report: bool = True,
) -> EquivalenceReport:
    """Run the full MERIDIAN equivalence pipeline.

    Parameters
    ----------
    dataset_path:
        Path to a pre-populated JSON dataset (see ``datasets/example.json``).
    sample_n:
        If set, draw a stratified sample of this size before scoring.
    seed:
        Random seed for reproducible sampling.
    equivalent_threshold:
        Cosine similarity >= this value → EQUIVALENT. Default 0.92.
    review_threshold:
        Cosine similarity >= this value → REVIEW. Default 0.75.
    embedder_model:
        Sentence-transformer model name. Runs locally, no API key required.
    reports_dir:
        Directory where JSON and markdown reports are written.
    report_stem:
        Filename stem for the report (auto-timestamped if omitted).
    write_report:
        Set to False to skip writing report files (useful in tests / notebooks).

    Returns
    -------
    EquivalenceReport
        Dataset-level verdict with tier counts, percentages, and Wilson CI.
    """
    records = load(dataset_path)

    if sample_n is not None:
        records = stratified_sample(records, sample_n, seed=seed)

    embedder = Embedder(embedder_model)
    scorer = DriftScorer(
        equivalent_threshold=equivalent_threshold,
        review_threshold=review_threshold,
        embedder=embedder,
    )
    results = scorer.score_all(records)

    reporter = Reporter(reports_dir=reports_dir)
    report = reporter.build(results)

    if write_report:
        reporter.write(report, stem=report_stem)

    return report
