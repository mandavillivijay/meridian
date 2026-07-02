"""
Core Pydantic data models for MERIDIAN.

MERIDIAN treats model migration as a regression testing problem: given a deprecated
model and its replacement, does the replacement produce semantically equivalent outputs
on your specific workload? These models represent the data flowing through that pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class IntentCategory(str, Enum):
    factual = "factual"
    generative = "generative"
    classification = "classification"
    structured_output = "structured_output"


class TierVerdict(str, Enum):
    EQUIVALENT = "EQUIVALENT"
    REVIEW = "REVIEW"
    DRIFTED = "DRIFTED"


class PromptRecord(BaseModel):
    """A single prompt with outputs from both the deprecated and replacement model."""

    prompt: str
    intent: IntentCategory
    old_output: Optional[str] = None
    new_output: Optional[str] = None

    model_config = {"frozen": False}


class DriftResult(BaseModel):
    """Per-prompt equivalence result from the three-tier gate."""

    prompt_record: PromptRecord
    similarity: float = Field(ge=0.0, le=1.0)
    verdict: TierVerdict

    model_config = {"frozen": True}


class EquivalenceReport(BaseModel):
    """Dataset-level aggregate verdict with confidence interval on the EQUIVALENT proportion."""

    total: int = Field(ge=0)
    equivalent_count: int = Field(ge=0)
    review_count: int = Field(ge=0)
    drifted_count: int = Field(ge=0)

    equivalent_pct: float = Field(ge=0.0, le=100.0)
    review_pct: float = Field(ge=0.0, le=100.0)
    drifted_pct: float = Field(ge=0.0, le=100.0)

    # Wilson score confidence interval on the EQUIVALENT proportion
    wilson_lower: float = Field(ge=0.0, le=1.0)
    wilson_upper: float = Field(ge=0.0, le=1.0)

    summary: str

    @model_validator(mode="after")
    def counts_sum_to_total(self) -> "EquivalenceReport":
        if self.equivalent_count + self.review_count + self.drifted_count != self.total:
            raise ValueError("equivalent_count + review_count + drifted_count must equal total")
        return self
