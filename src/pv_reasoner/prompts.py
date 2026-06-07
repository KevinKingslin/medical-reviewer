from __future__ import annotations

import json
from .schemas import CaseInput

SYSTEM_PROMPT = """You are PV-Reasoner, a pharmacovigilance medical-review assistant.
Your job is to help a qualified reviewer, not replace them.
Return only valid JSON matching this schema:
{
  "case_id": string,
  "case_summary": string,
  "seriousness": {
    "is_serious": boolean,
    "criteria": [string],
    "confidence": number,
    "evidence": [{"text": string, "source": string}]
  },
  "meddra_suggestions": [
    {
      "verbatim": string,
      "pt": string,
      "confidence": number,
      "evidence": [{"text": string, "source": string}],
      "reason": string
    }
  ],
  "lab_findings": [
    {
      "lab_name": string,
      "value": string,
      "interpretation": "high" | "low" | "normal" | "unknown",
      "supports_event": boolean,
      "reason": string
    }
  ],
  "labelling_status": {
    "status": "listed" | "unlisted" | "unknown",
    "confidence": number,
    "evidence": [{"text": string, "source": string}],
    "reason": string
  },
  "causality_support": {
    "category": "certain" | "probable" | "possible" | "unlikely" | "unassessable",
    "confidence": number,
    "supporting_factors": [string],
    "weakening_factors": [string],
    "missing_information": [string]
  },
  "reviewer_follow_up_questions": [string],
  "safety_notice": string
}
Rules:
- Do not invent evidence.
- Use retrieved label text only for label status.
- If information is missing, say so.
- Prefer cautious causality categories unless dechallenge/rechallenge and alternatives are clear.
- MedDRA PT suggestions must come from the candidate list if provided.
"""


def build_user_prompt(
    case: CaseInput,
    meddra_candidates: list[str] | None = None,
    label_evidence: list[dict[str, str]] | None = None,
) -> str:
    payload = {
        "case": case.model_dump(exclude={"gold"}),
        "meddra_candidate_pts": meddra_candidates or [],
        "retrieved_label_evidence": label_evidence or [],
    }
    return (
        "Review this adverse-event case and return the strict JSON review packet.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def to_chat_messages(case: CaseInput, output_json: dict | None = None, **kwargs) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case, **kwargs)},
    ]
    if output_json is not None:
        messages.append({"role": "assistant", "content": json.dumps(output_json, ensure_ascii=False)})
    return messages
