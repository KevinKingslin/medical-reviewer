from __future__ import annotations

import argparse
from pathlib import Path

from pv_reasoner.inference.reviewer import BaselineReviewer
from pv_reasoner.prompts import to_chat_messages
from pv_reasoner.retrieval.label_store import LabelStore
from pv_reasoner.retrieval.meddra import MeddraCandidateRetriever
from pv_reasoner.schemas import CaseInput, SFTExample
from pv_reasoner.utils import read_jsonl, write_jsonl


def build_examples(cases_path: str, label_path: str, out_path: str, max_examples: int | None = None) -> int:
    cases = [CaseInput.model_validate(row) for row in read_jsonl(cases_path)]
    if max_examples:
        cases = cases[:max_examples]
    label_store = LabelStore.from_jsonl(label_path)
    reviewer = BaselineReviewer(label_store, MeddraCandidateRetriever())
    rows = []
    for case in cases:
        packet = reviewer.review(case)
        candidates = reviewer.meddra.candidates(case.narrative, case.reactions, top_k=10)
        label_sections = label_store.search(case.suspect_drug, " ".join(candidates + case.reactions), top_k=4)
        messages = to_chat_messages(
            case,
            output_json=packet.model_dump(),
            meddra_candidates=candidates,
            label_evidence=[s.to_evidence() for s in label_sections],
        )
        rows.append(
            SFTExample(
                messages=messages,
                case_id=case.case_id,
                metadata={"source": case.source, "generated_by": "baseline_reviewer"},
            ).model_dump()
        )
    write_jsonl(out_path, rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supervised fine-tuning JSONL examples.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--labels", default="data/labels/sample_label_sections.jsonl")
    parser.add_argument("--out", default="data/processed/sft_train.jsonl")
    parser.add_argument("--max_examples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_examples(args.cases, args.labels, args.out, args.max_examples)
    print(f"Wrote {count} SFT examples to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
