from __future__ import annotations

from typing import Any
from .schemas import LabFinding

# Simplified demo thresholds. Replace with unit-aware clinical reference ranges for production.
THRESHOLDS = {
    "alt": (0, 40),
    "ast": (0, 40),
    "bilirubin": (0, 1.2),
    "creatinine": (0.4, 1.3),
    "ck": (0, 200),
    "creatine kinase": (0, 200),
    "platelets": (150000, 450000),
    "wbc": (4000, 11000),
    "potassium": (3.5, 5.2),
    "sodium": (135, 145),
    "amylase": (0, 110),
    "lipase": (0, 160),
}

EVENT_LAB_HINTS = {
    "rhabdomyolysis": ["ck", "creatine kinase"],
    "hepatitis": ["alt", "ast", "bilirubin"],
    "liver injury": ["alt", "ast", "bilirubin"],
    "pancreatitis": ["amylase", "lipase"],
    "thrombocytopenia": ["platelets"],
    "renal failure": ["creatinine"],
    "acute kidney injury": ["creatinine"],
}


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = "".join(ch for ch in value if ch.isdigit() or ch in ".-")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def assess_labs(labs: dict[str, Any], reactions: list[str]) -> list[LabFinding]:
    findings: list[LabFinding] = []
    reaction_text = " ".join(reactions).lower()
    for raw_name, raw_value in labs.items():
        name = raw_name.strip().lower()
        value = _to_float(raw_value)
        if value is None or name not in THRESHOLDS:
            findings.append(
                LabFinding(
                    lab_name=raw_name,
                    value=str(raw_value),
                    interpretation="unknown",
                    supports_event=False,
                    reason="No demo threshold available or value could not be parsed.",
                )
            )
            continue
        low, high = THRESHOLDS[name]
        if value < low:
            interpretation = "low"
        elif value > high:
            interpretation = "high"
        else:
            interpretation = "normal"

        supports = False
        for event, labs_for_event in EVENT_LAB_HINTS.items():
            if event in reaction_text and name in labs_for_event and interpretation != "normal":
                supports = True
                break

        findings.append(
            LabFinding(
                lab_name=raw_name,
                value=str(raw_value),
                interpretation=interpretation,
                supports_event=supports,
                reason=f"Demo threshold range for {raw_name}: {low}-{high}.",
            )
        )
    return findings
