from __future__ import annotations

import re

from .schemas import EvidenceSpan, SeriousnessAssessment, CaseInput
from .utils import normalize_text

SERIOUSNESS_KEYWORDS = {
    "death": ["death", "died", "fatal", "expired"],
    "life_threatening": ["life-threatening", "life threatening", "anaphylaxis", "shock", "icu", "intensive care"],
    "hospitalization": ["hospitalized", "hospitalised", "admitted", "admission", "inpatient", "hospitalization", "hospitalisation"],
    "disability": ["disability", "disabled", "permanent damage", "persistent incapacity"],
    "congenital_anomaly": ["birth defect", "congenital", "foetal", "fetal", "teratogenic"],
    "required_intervention": ["required intervention", "surgery", "intubated", "ventilated", "dialysis"],
    "important_medical_event": ["rhabdomyolysis", "seizure", "stevens-johnson", "toxic epidermal", "pancreatitis", "liver failure"],
}

OUTCOME_TO_CRITERION = {
    "death": "death",
    "lt": "life_threatening",
    "life-threatening": "life_threatening",
    "hospitalization": "hospitalization",
    "hospitalisation": "hospitalization",
    "disability": "disability",
    "congenital anomaly": "congenital_anomaly",
    "other serious": "important_medical_event",
}


def _negated(text: str, keyword: str) -> bool:
    # Handles simple phrases such as "no hospitalization" and list-style negation
    # such as "no hospitalization, emergency treatment, or life-threatening features".
    for match in re.finditer(re.escape(keyword), text):
        window_start = max(0, match.start() - 90)
        window = text[window_start:match.start()]
        # reset at sentence boundaries so an earlier unrelated "no" does not suppress later evidence
        window = re.split(r"[.;]", window)[-1]
        if re.search(r"\b(no|not|without|denies|denied)\b", window):
            return True
    return False


def assess_seriousness(case: CaseInput) -> SeriousnessAssessment:
    text = normalize_text(" ".join([case.narrative, " ".join(case.report_outcomes), " ".join(case.reactions)]))
    criteria: list[str] = []
    evidence: list[EvidenceSpan] = []

    for outcome in case.report_outcomes:
        key = normalize_text(outcome)
        for phrase, criterion in OUTCOME_TO_CRITERION.items():
            if phrase in key and criterion not in criteria:
                criteria.append(criterion)
                evidence.append(EvidenceSpan(text=f"Reported outcome: {outcome}", source="case_outcome"))

    for criterion, keywords in SERIOUSNESS_KEYWORDS.items():
        matched_keywords = [k for k in keywords if k in text and not _negated(text, k)]
        if matched_keywords and criterion not in criteria:
            criteria.append(criterion)
            matched = matched_keywords[0]
            evidence.append(EvidenceSpan(text=f"Narrative/reaction contains seriousness clue: {matched}", source="case_narrative"))

    is_serious = bool(criteria)
    confidence = 0.9 if criteria else 0.7
    if not evidence:
        evidence.append(EvidenceSpan(text="No explicit seriousness criterion was identified in the provided case fields.", source="case_review"))
    return SeriousnessAssessment(is_serious=is_serious, criteria=criteria, confidence=confidence, evidence=evidence)
