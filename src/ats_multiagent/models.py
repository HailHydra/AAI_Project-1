from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Resume:
    candidate_name: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobDescription:
    title: str
    text: str
    required_skills: list[str] = field(default_factory=list)


@dataclass
class FeatureResult:
    name: str
    value: Any
    rationale: str


@dataclass
class ScreeningReport:
    candidate_name: str
    fit_score: float
    strengths: list[str]
    concerns: list[str]
    final_recommendation: str
    feature_breakdown: list[FeatureResult]
    iteration_notes: list[str]
