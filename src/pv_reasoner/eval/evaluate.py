from __future__ import annotations

import argparse
from statistics import mean

from pv_reasoner.inference.reviewer import make_reviewer
from pv_reasoner.schemas import CaseInput
from pv_reasoner.utils import read_jsonl, write_jsonl


def evaluate(cases_path: str, label_path: str, predictions_out: str | None = None, base_model: str | None = None, adapter: str | None = None) -> dict:
    cases = [CaseInput.model_validate(row) for row in read_jsonl(cases_path)]
    reviewer = make_reviewer(label_path, base_model=base_model, adapter=adapter)
    predictions = []
    seriousness_hits = []
    label_hits = []
    top3_hits = []
    evidence_coverage = []
    json_validity = []

    for case in cases:
        try:
            packet = reviewer.review(case)
            pred = packet.model_dump()
            json_validity.append(1)
        except Exception as exc:
            pred = {"case_id": case.case_id, "error": str(exc)}
            json_validity.append(0)
            predictions.append(pred)
            continue

        gold = case.gold or {}
        if "serious" in gold:
            seriousness_hits.append(int(packet.seriousness.is_serious == gold["serious"]))
        if gold.get("label_status"):
            label_hits.append(int(packet.labelling_status.status == gold["label_status"]))
        if gold.get("primary_pt"):
            pts = [s.pt.lower() for s in packet.meddra_suggestions[:3]]
            top3_hits.append(int(str(gold["primary_pt"]).lower() in pts))

        has_evidence = bool(packet.seriousness.evidence) and bool(packet.meddra_suggestions[0].evidence)
        evidence_coverage.append(int(has_evidence))
        predictions.append(pred)

    metrics = {
        "n_cases": len(cases),
        "json_validity": mean(json_validity) if json_validity else 0,
        "seriousness_accuracy": mean(seriousness_hits) if seriousness_hits else None,
        "label_status_accuracy": mean(label_hits) if label_hits else None,
        "meddra_top3_hit_rate": mean(top3_hits) if top3_hits else None,
        "evidence_coverage": mean(evidence_coverage) if evidence_coverage else None,
    }
    if predictions_out:
        write_jsonl(predictions_out, predictions)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PV-Reasoner predictions.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--labels", default="data/labels/sample_label_sections.jsonl")
    parser.add_argument("--predictions_out", default=None)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--adapter", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.cases, args.labels, args.predictions_out, args.base_model, args.adapter)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
