from __future__ import annotations

from pathlib import Path
from rapidfuzz import fuzz, process

DEFAULT_TERMS = [
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
    "Hepatitis",
    "Drug-induced liver injury",
    "Pancreatitis",
    "Acute kidney injury",
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
    "Renal impairment",
]


class MeddraCandidateRetriever:
    """Tiny demo retriever. Replace terms with licensed MedDRA PT/LLT list in production."""

    def __init__(self, terms: list[str] | None = None):
        self.terms = sorted(set(terms or DEFAULT_TERMS))

    @classmethod
    def from_file(cls, path: str | Path | None) -> "MeddraCandidateRetriever":
        if path is None or not Path(path).exists():
            return cls()
        terms = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(terms)

    def candidates(self, text: str, reactions: list[str] | None = None, top_k: int = 8) -> list[str]:
        queries = [text] + (reactions or [])
        scores: dict[str, float] = {}
        for query in queries:
            if not query:
                continue
            matches = process.extract(query, self.terms, scorer=fuzz.token_set_ratio, limit=top_k)
            for term, score, _ in matches:
                scores[term] = max(scores.get(term, 0), float(score))
            lowered = query.lower()
            for term in self.terms:
                if term.lower() in lowered:
                    scores[term] = max(scores.get(term, 0), 100.0)
        return [term for term, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]

    def best_match(self, phrase: str) -> tuple[str, float]:
        match = process.extractOne(phrase, self.terms, scorer=fuzz.token_set_ratio)
        if not match:
            return phrase, 0.0
        term, score, _ = match
        return term, float(score) / 100.0
