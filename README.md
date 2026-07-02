# MERIDIAN

**Model Equivalence and Regression via Intent Drift In AI Networks**

A lightweight Python library for validating LLM model equivalence when a vendor deprecates a model and you need to migrate to a replacement.

---

## The Problem

When OpenAI deprecates `gpt-4-0613` or Anthropic retires `claude-2`, enterprise teams have no established, reusable methodology to validate that the replacement produces semantically equivalent outputs for their specific workload. Traditional software testing checks exact outputs — useless for non-deterministic LLM responses. Existing benchmarks (MMLU, HELM) measure absolute capability, not relative equivalence between two specific models on your use case.

## How MERIDIAN Is Different

Recent work ([arXiv:2604.27082](https://arxiv.org/abs/2604.27082), [arXiv:2507.05573](https://arxiv.org/abs/2507.05573), [arXiv:2604.27789](https://arxiv.org/abs/2604.27789)) describes migration validation processes using LLM-as-judge evaluation or human review. MERIDIAN takes a different approach:

| | Existing approaches | MERIDIAN |
|---|---|---|
| **Scoring method** | LLM-as-judge or human eval | Sentence-transformer cosine similarity |
| **Cloud dependency** | Requires API calls to score | Runs entirely locally |
| **Cost** | Per-token API cost to evaluate | Free after model download |
| **Reproducibility** | Non-deterministic (LLM judge) | Deterministic |
| **Framing** | Evaluation problem | Regression testing problem |
| **Format** | Research process descriptions | Reusable open-source library |

**Core insight:** embed old and new model outputs using a sentence-transformer, compute cosine similarity, and flag pairs below a drift threshold. Same technique as [canvas-heal](https://pypi.org/project/canvas-heal/) (UI locator healing), different problem surface.

## Three-Tier Gate

```
Cosine Similarity
─────────────────────────────────────────────────────────
0.0 ──────────── 0.75 ──────────── 0.92 ──────────── 1.0
     DRIFTED          REVIEW              EQUIVALENT
     (flag)         (human eye)          (auto-pass)
```

Thresholds are configurable. Defaults (0.92 / 0.75) are starting points — calibrate them against a small human-labeled set for your domain. See the accompanying paper for a calibration procedure derived from the deepseek-chat (V3) → deepseek-reasoner (R1) empirical study.

## Installation

```bash
pip install meridian-regression
```

Or from source:

```bash
git clone https://github.com/mandavillivijay/meridian
cd meridian
pip install -e ".[dev]"
```

## Quickstart

### 1. Build your golden dataset

Create a JSON file with outputs from both models for each prompt:

```json
[
  {
    "prompt": "What is the capital of France?",
    "intent": "factual",
    "old_output": "The capital of France is Paris.",
    "new_output": "Paris is the capital city of France."
  }
]
```

Intent categories: `factual`, `generative`, `classification`, `structured_output`.

Run your old model and new model on the same prompts, save the outputs. MERIDIAN doesn't call any APIs — you bring the outputs.

### 2. Run the pipeline

```python
from meridian.runner import run

report = run("datasets/my_golden_set.json")
print(report.summary)
# "94.0% of outputs are semantically equivalent, 4.0% show minor drift
#  requiring human review, 2.0% show significant drift (regression flagged)."
```

### 3. Use the report

```python
print(f"Equivalent: {report.equivalent_pct}%")
print(f"Wilson 95% CI: [{report.wilson_lower:.3f}, {report.wilson_upper:.3f}]")
```

JSON and markdown reports are written to `reports/` automatically.

## Advanced Usage

```python
from meridian.runner import run

report = run(
    "datasets/my_golden_set.json",
    sample_n=50,              # stratified sample of 50 prompts
    seed=42,                  # reproducible sampling
    equivalent_threshold=0.90,
    review_threshold=0.70,
    report_stem="sonnet_migration_v2",
)
```

### Using modules directly

```python
from meridian.sampler import load, stratified_sample
from meridian.scorer import DriftScorer
from meridian.reporter import Reporter

records = load("datasets/my_golden_set.json")
records = stratified_sample(records, n=50, seed=42)

scorer = DriftScorer()
results = scorer.score_all(records)

reporter = Reporter()
report = reporter.build(results)
reporter.write(report, stem="my_run")
```

### Bringing your own adapter

If you want to populate outputs programmatically rather than from a JSON file, implement the `ModelAdapter` protocol:

```python
from meridian.adapters.base import ModelAdapter

class MyAdapter:
    def complete(self, prompt: str) -> str:
        # call your model here
        ...
    def name(self) -> str:
        return "my-model-v2"
```

## Project Structure

```
meridian/
├── meridian/
│   ├── models.py       # Pydantic data models
│   ├── embedder.py     # Sentence-transformer wrapper (singleton)
│   ├── scorer.py       # Three-tier drift gate
│   ├── reporter.py     # Aggregate verdict + JSON/markdown output
│   ├── sampler.py      # Dataset loading + stratified sampling
│   ├── runner.py       # End-to-end pipeline entry point
│   └── adapters/
│       └── base.py     # ModelAdapter Protocol (extension point)
├── datasets/           # Example golden datasets
├── reports/            # Generated reports
└── tests/              # pytest suite (106 tests)
```

## Running Tests

```bash
pytest
```

## Author

Vijay Mandavilli — Quality Engineering Lead, Cognida AI, Hyderabad, India

## License

MIT
