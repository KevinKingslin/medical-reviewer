from __future__ import annotations

import argparse
import json
from pathlib import Path

from pv_reasoner.inference.reviewer import make_reviewer
from pv_reasoner.schemas import CaseInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PV-Reasoner on a single case JSON file.")
    parser.add_argument("--case_json", required=True)
    parser.add_argument("--labels", default="data/labels/sample_label_sections.jsonl")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--adapter", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = CaseInput.model_validate(json.loads(Path(args.case_json).read_text(encoding="utf-8")))
    reviewer = make_reviewer(args.labels, base_model=args.base_model, adapter=args.adapter)
    packet = reviewer.review(case)
    print(json.dumps(packet.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
