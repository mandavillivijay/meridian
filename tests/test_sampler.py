import json
from pathlib import Path

import pytest

from meridian.models import IntentCategory, PromptRecord
from meridian.sampler import load, stratified_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dataset(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(records), encoding="utf-8")
    return p


def _record_dict(
    prompt: str = "p",
    intent: str = "factual",
    old: str = "old",
    new: str = "new",
) -> dict:
    return {"prompt": prompt, "intent": intent, "old_output": old, "new_output": new}


def _make_records(counts: dict[str, int]) -> list[PromptRecord]:
    """Create PromptRecord list with given per-intent counts."""
    records = []
    for intent, n in counts.items():
        for i in range(n):
            records.append(PromptRecord(
                prompt=f"{intent}_{i}",
                intent=IntentCategory(intent),
                old_output="old",
                new_output="new",
            ))
    return records


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_loads_valid_dataset(self, tmp_path):
        path = _write_dataset(tmp_path, [_record_dict()])
        records = load(path)
        assert len(records) == 1
        assert isinstance(records[0], PromptRecord)

    def test_loads_all_four_intents(self, tmp_path):
        data = [
            _record_dict(intent="factual"),
            _record_dict(intent="generative"),
            _record_dict(intent="classification"),
            _record_dict(intent="structured_output"),
        ]
        records = load(_write_dataset(tmp_path, data))
        intents = {r.intent for r in records}
        assert intents == set(IntentCategory)

    def test_loads_example_dataset(self):
        path = Path("datasets/example.json")
        records = load(path)
        assert len(records) == 8
        intents = {r.intent for r in records}
        assert intents == set(IntentCategory)

    def test_non_array_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array"):
            load(p)

    def test_invalid_intent_raises(self, tmp_path):
        path = _write_dataset(tmp_path, [_record_dict(intent="unknown")])
        with pytest.raises(ValueError, match="Record 0"):
            load(path)

    def test_missing_old_output_raises(self, tmp_path):
        data = [{"prompt": "p", "intent": "factual", "new_output": "n"}]
        path = _write_dataset(tmp_path, data)
        with pytest.raises(ValueError, match="old_output"):
            load(path)

    def test_missing_new_output_raises(self, tmp_path):
        data = [{"prompt": "p", "intent": "factual", "old_output": "o"}]
        path = _write_dataset(tmp_path, data)
        with pytest.raises(ValueError, match="new_output"):
            load(path)

    def test_missing_prompt_raises(self, tmp_path):
        data = [{"intent": "factual", "old_output": "o", "new_output": "n"}]
        path = _write_dataset(tmp_path, data)
        with pytest.raises(ValueError, match="Record 0"):
            load(path)

    def test_empty_array_returns_empty_list(self, tmp_path):
        path = _write_dataset(tmp_path, [])
        assert load(path) == []

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# stratified_sample()
# ---------------------------------------------------------------------------

class TestStratifiedSample:
    def test_empty_input_returns_empty(self):
        assert stratified_sample([], 5) == []

    def test_n_gte_total_returns_all(self):
        records = _make_records({"factual": 3, "generative": 2})
        result = stratified_sample(records, 10)
        assert len(result) == 5

    def test_n_equals_total_returns_all(self):
        records = _make_records({"factual": 3, "generative": 3})
        result = stratified_sample(records, 6)
        assert len(result) == 6

    def test_sample_size_respected(self):
        records = _make_records({"factual": 10, "generative": 10, "classification": 10})
        result = stratified_sample(records, 6, seed=42)
        assert len(result) == 6

    def test_all_present_categories_represented(self):
        records = _make_records({
            "factual": 5,
            "generative": 5,
            "classification": 5,
            "structured_output": 5,
        })
        result = stratified_sample(records, 4, seed=0)
        intents = {r.intent for r in result}
        assert len(intents) == 4

    def test_single_category_still_works(self):
        records = _make_records({"factual": 10})
        result = stratified_sample(records, 3, seed=0)
        assert len(result) == 3
        assert all(r.intent == IntentCategory.factual for r in result)

    def test_result_is_subset_of_input(self):
        records = _make_records({"factual": 5, "generative": 5})
        result = stratified_sample(records, 4, seed=1)
        prompt_set = {r.prompt for r in records}
        for r in result:
            assert r.prompt in prompt_set

    def test_seed_produces_deterministic_output(self):
        records = _make_records({"factual": 10, "generative": 10})
        r1 = [r.prompt for r in stratified_sample(records, 5, seed=99)]
        r2 = [r.prompt for r in stratified_sample(records, 5, seed=99)]
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        records = _make_records({"factual": 10, "generative": 10})
        r1 = [r.prompt for r in stratified_sample(records, 5, seed=1)]
        r2 = [r.prompt for r in stratified_sample(records, 5, seed=2)]
        assert r1 != r2

    def test_no_duplicates_in_sample(self):
        records = _make_records({"factual": 5, "generative": 5})
        result = stratified_sample(records, 6, seed=7)
        prompts = [r.prompt for r in result]
        assert len(prompts) == len(set(prompts))
