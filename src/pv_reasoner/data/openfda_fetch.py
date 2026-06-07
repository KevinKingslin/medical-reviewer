from __future__ import annotations

import argparse
import os
from typing import Any

import requests

from pv_reasoner.schemas import CaseInput
from pv_reasoner.utils import write_jsonl

OPENFDA_EVENT_ENDPOINT = "https://api.fda.gov/drug/event.json"

SERIOUS_FLAGS = {
    "serious",
    "seriousnessdeath",
    "seriousnesslifethreatening",
    "seriousnesshospitalization",
    "seriousnessdisabling",
    "seriousnesscongenitalanomali",
    "seriousnessother",
}


def _first(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return str(value[0])
    if value is None:
        return None
    return str(value)


def normalize_openfda_event(row: dict[str, Any], idx: int) -> CaseInput:
    patient = row.get("patient", {}) or {}
    reactions = [r.get("reactionmeddrapt", "") for r in patient.get("reaction", []) if r.get("reactionmeddrapt")]
    drugs = patient.get("drug", []) or []
    suspect_drugs = []
    concomitant_drugs = []
    for drug in drugs:
        name = _first((drug.get("openfda") or {}).get("brand_name")) or drug.get("medicinalproduct")
        if not name:
            continue
        role = str(drug.get("drugcharacterization", ""))
        if role == "1":
            suspect_drugs.append(name)
        else:
            concomitant_drugs.append(name)

    outcomes = []
    for flag, label in [
        ("seriousnessdeath", "death"),
        ("seriousnesslifethreatening", "life-threatening"),
        ("seriousnesshospitalization", "hospitalization"),
        ("seriousnessdisabling", "disability"),
        ("seriousnesscongenitalanomali", "congenital anomaly"),
        ("seriousnessother", "other serious"),
    ]:
        if str(row.get(flag, "")) == "1":
            outcomes.append(label)

    sex = {"1": "male", "2": "female"}.get(str(patient.get("patientsex", "")), None)
    age = patient.get("patientonsetage")
    age_unit = patient.get("patientonsetageunit")
    age_text = f"{age} unit:{age_unit}" if age else None
    drug = suspect_drugs[0] if suspect_drugs else (drugs[0].get("medicinalproduct") if drugs else None)
    reaction_text = ", ".join(reactions[:5]) or "adverse event"
    narrative = (
        f"FAERS structured case. Patient {age_text or 'age unknown'}, sex {sex or 'unknown'}, "
        f"reported reaction(s): {reaction_text}. Suspect drug: {drug or 'unknown'}. "
        f"Seriousness outcomes reported: {', '.join(outcomes) if outcomes else 'none reported'}. "
        "This generated narrative is built from structured FAERS fields and may not contain full clinical detail."
    )
    return CaseInput(
        case_id=str(row.get("safetyreportid", f"OPENFDA-{idx}")),
        narrative=narrative,
        suspect_drug=drug,
        concomitant_drugs=concomitant_drugs[:10],
        patient_age=age_text,
        patient_sex=sex,
        report_outcomes=outcomes,
        reactions=reactions[:5],
        source="openFDA_FAERS",
        gold={
            "serious": str(row.get("serious", "")) == "1" or bool(outcomes),
            "primary_pt": reactions[0] if reactions else None,
        },
    )


def fetch_events(limit: int, search: str | None = None, api_key: str | None = None) -> list[CaseInput]:
    params: dict[str, Any] = {"limit": min(limit, 100)}
    if search:
        params["search"] = search
    if api_key:
        params["api_key"] = api_key
    response = requests.get(OPENFDA_EVENT_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    results = response.json().get("results", [])
    return [normalize_openfda_event(row, idx) for idx, row in enumerate(results)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch openFDA drug event examples and normalize them.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--search", default=None, help="Optional openFDA search string, e.g. patient.drug.medicinalproduct:atorvastatin")
    parser.add_argument("--out", default="data/raw/openfda_events.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = fetch_events(args.limit, args.search, os.getenv("OPENFDA_API_KEY"))
    write_jsonl(args.out, [case.model_dump() for case in cases])
    print(f"Wrote {len(cases)} normalized cases to {args.out}")


if __name__ == "__main__":
    main()
