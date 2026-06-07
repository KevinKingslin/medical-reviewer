from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_reasoner.data.synthetic import demo_cases
from pv_reasoner.utils import write_jsonl

LABELS = [
    {
        "drug": "Atorvastatin",
        "section": "Warnings and Precautions / Adverse Reactions",
        "text": "Atorvastatin labeling includes myopathy and rhabdomyolysis. Patients should report unexplained muscle pain, tenderness, or weakness, particularly if accompanied by malaise or fever.",
        "adverse_reactions": ["Myalgia", "Myopathy", "Rhabdomyolysis", "Liver enzyme increased"],
        "source": "demo_label",
    },
    {
        "drug": "Amoxicillin",
        "section": "Adverse Reactions",
        "text": "The most common adverse reactions associated with amoxicillin include nausea, diarrhea, and rash. Serious hypersensitivity reactions including anaphylaxis have been reported.",
        "adverse_reactions": ["Rash", "Diarrhoea", "Nausea", "Anaphylactic reaction", "Urticaria"],
        "source": "demo_label",
    },
    {
        "drug": "Metformin",
        "section": "Warnings and Precautions",
        "text": "Metformin labeling warns about lactic acidosis and use in renal impairment. Acute kidney injury, dehydration, sepsis, and hypoxic states may increase risk and require interruption of metformin.",
        "adverse_reactions": ["Lactic acidosis", "Renal impairment", "Acute kidney injury", "Vomiting", "Diarrhoea"],
        "source": "demo_label",
    },
    {
        "drug": "Ibuprofen",
        "section": "Warnings and Precautions",
        "text": "Ibuprofen and other NSAIDs can cause gastrointestinal bleeding, ulceration, perforation, renal toxicity, hypersensitivity reactions, and serious skin reactions.",
        "adverse_reactions": ["Gastrointestinal haemorrhage", "Renal failure", "Rash", "Stevens-Johnson syndrome"],
        "source": "demo_label",
    },
    {
        "drug": "Aspirin",
        "section": "Warnings and Precautions / Adverse Reactions",
        "text": "Aspirin labeling includes bleeding risk, gastrointestinal bleeding, dyspepsia, nausea, vomiting, bronchospasm, urticaria, and hypersensitivity reactions.",
        "adverse_reactions": ["Gastrointestinal haemorrhage", "Bleeding", "Dyspepsia", "Nausea", "Urticaria"],
        "source": "demo_label",
    },
]

SEED_MEDDRA_TERMS = [
    "Rash",
    "Urticaria",
    "Anaphylactic reaction",
    "Nausea",
    "Vomiting",
    "Diarrhoea",
    "Headache",
    "Dizziness",
    "Rhabdomyolysis",
    "Myalgia",
    "Myopathy",
    "Hepatitis",
    "Drug-induced liver injury",
    "Pancreatitis",
    "Acute kidney injury",
    "Renal impairment",
    "Renal failure",
    "Thrombocytopenia",
    "Stevens-Johnson syndrome",
    "Toxic epidermal necrolysis",
    "Hypoglycaemia",
    "Hyperkalaemia",
    "Seizure",
    "Dyspnoea",
    "Chest pain",
    "Angioedema",
    "Gastrointestinal haemorrhage",
    "Bleeding",
    "Lactic acidosis",
]


def main() -> None:
    (ROOT / "data" / "demo").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "labels").mkdir(parents=True, exist_ok=True)
    cases = demo_cases()
    write_jsonl(ROOT / "data" / "demo" / "demo_cases.jsonl", [case.model_dump() for case in cases])
    (ROOT / "data" / "demo" / "example_case.json").write_text(json.dumps(cases[0].model_dump(), indent=2), encoding="utf-8")
    write_jsonl(ROOT / "data" / "labels" / "sample_label_sections.jsonl", LABELS)
    (ROOT / "data" / "labels" / "seed_meddra_terms.txt").write_text("\n".join(SEED_MEDDRA_TERMS) + "\n", encoding="utf-8")
    print("Demo cases, label sections, and seed terms generated.")


if __name__ == "__main__":
    main()
