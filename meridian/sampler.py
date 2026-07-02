"""
Golden dataset loading and stratified sampling.

In MERIDIAN's default workflow both model outputs are pre-populated in a JSON
file — teams run each model on their own infrastructure and hand this library
the results. The sampler validates that structure and, optionally, draws a
stratified subset that maintains intent-category balance.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from meridian.models import IntentCategory, PromptRecord


def load(path: Path | str) -> list[PromptRecord]:
    """Load and validate a pre-populated dataset JSON file.

    Expected format: a JSON array of objects with keys
    ``prompt``, ``intent``, ``old_output``, ``new_output``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array; got {type(data).__name__}")
    records = []
    for i, item in enumerate(data):
        try:
            record = PromptRecord(**item)
        except Exception as exc:
            raise ValueError(f"Record {i} is invalid: {exc}") from exc
        if record.old_output is None or record.new_output is None:
            raise ValueError(
                f"Record {i} (prompt={record.prompt!r:.40}) is missing "
                f"old_output or new_output — dataset must be pre-populated."
            )
        records.append(record)
    return records


def stratified_sample(
    records: list[PromptRecord],
    n: int,
    seed: Optional[int] = None,
) -> list[PromptRecord]:
    """Return a stratified sample of size n drawn proportionally across intent categories.

    If n >= len(records) all records are returned (shuffled).
    Each category that is present gets at least one record, provided n allows it.
    """
    if not records:
        return []
    if n >= len(records):
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        return shuffled

    buckets: dict[IntentCategory, list[PromptRecord]] = defaultdict(list)
    for r in records:
        buckets[r.intent].append(r)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Proportional allocation with a floor of 1 per present category.
    total = len(records)
    k = len(buckets)
    allocations: dict[IntentCategory, int] = {}
    remainder = n

    for intent, bucket in buckets.items():
        alloc = max(1, math.floor(n * len(bucket) / total))
        alloc = min(alloc, len(bucket))
        allocations[intent] = alloc
        remainder -= alloc

    # Distribute leftover slots to the largest buckets first.
    if remainder > 0:
        by_size = sorted(buckets.keys(), key=lambda c: len(buckets[c]), reverse=True)
        for intent in by_size:
            if remainder == 0:
                break
            cap = len(buckets[intent]) - allocations[intent]
            give = min(remainder, cap)
            allocations[intent] += give
            remainder -= give

    sampled: list[PromptRecord] = []
    for intent, count in allocations.items():
        sampled.extend(buckets[intent][:count])

    rng.shuffle(sampled)
    return sampled
