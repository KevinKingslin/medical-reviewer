from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class CaseInput(BaseModel):
    """Normalized case input for the reviewer assistant."""

    case_id: str = Field(default="manual_case")
    narrative: str = Field(..., min_length=1)
    suspect_drug: str | None = None
    concomitant_drugs: list[str] = Field(default_factory=list)
    patient_age: str | None = None
    patient_sex: str | None = None
    report_outcomes: list[str] = Field(default_factory=list)
    reactions: list[str] = Field(default_factory=list)
    labs: dict[str, Any] = Field(default_factory=dict)
    source: str = "manual"
    gold: dict[str, Any] | None = None


class EvidenceSpan(BaseModel):
    text: str
    source: str = "case_narrative"


class SeriousnessAssessment(BaseModel):
    is_serious: bool
    criteria: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class MeddraSuggestion(BaseModel):
    verbatim: str
    pt: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    reason: str = ""


class LabFinding(BaseModel):
    lab_name: str
    value: str
    interpretation: Literal["high", "low", "normal", "unknown"]
    supports_event: bool = False
    reason: str = ""


class LabelingStatus(BaseModel):
    status: Literal["listed", "unlisted", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    reason: str = ""


class CausalitySupport(BaseModel):
    category: Literal["certain", "probable", "possible", "unlikely", "unassessable"]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_factors: list[str] = Field(default_factory=list)
    weakening_factors: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class ReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    case_summary: str
    seriousness: SeriousnessAssessment
    meddra_suggestions: list[MeddraSuggestion]
    lab_findings: list[LabFinding] = Field(default_factory=list)
    labelling_status: LabelingStatus
    causality_support: CausalitySupport
    reviewer_follow_up_questions: list[str] = Field(default_factory=list)
    safety_notice: str = "Reviewer-assist output only. Human medical review is required."

    @field_validator("meddra_suggestions")
    @classmethod
    def at_least_one_meddra(cls, value: list[MeddraSuggestion]) -> list[MeddraSuggestion]:
        if not value:
            raise ValueError("At least one MedDRA suggestion is required")
        return value


class SFTExample(BaseModel):
    messages: list[dict[str, str]]
    case_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
