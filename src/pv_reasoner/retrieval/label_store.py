from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pv_reasoner.utils import read_jsonl, normalize_text


@dataclass
class LabelSection:
    drug: str
    section: str
    text: str
    adverse_reactions: list[str]
    source: str = "sample_label"

    @classmethod
    def from_dict(cls, row: dict) -> "LabelSection":
        return cls(
            drug=row.get("drug", ""),
            section=row.get("section", ""),
            text=row.get("text", ""),
            adverse_reactions=row.get("adverse_reactions", []) or [],
            source=row.get("source", "sample_label"),
        )

    def to_evidence(self) -> dict[str, str]:
        return {
            "drug": self.drug,
            "section": self.section,
            "text": self.text,
            "source": self.source,
        }


class LabelStore:
    """Simple TF-IDF product-label retriever for hackathon use."""

    def __init__(self, sections: Iterable[LabelSection]):
        self.sections = list(sections)
        corpus = [self._doc_text(s) for s in self.sections]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LabelStore":
        rows = read_jsonl(path)
        return cls(LabelSection.from_dict(row) for row in rows)

    @staticmethod
    def _doc_text(section: LabelSection) -> str:
        return " ".join([section.drug, section.section, section.text, " ".join(section.adverse_reactions)])

    def search(self, drug: str | None, query: str, top_k: int = 3) -> list[LabelSection]:
        if not self.sections:
            return []
        drug_norm = normalize_text(drug)
        candidates = self.sections
        if drug_norm:
            drug_filtered = [s for s in self.sections if drug_norm in normalize_text(s.drug) or normalize_text(s.drug) in drug_norm]
            if drug_filtered:
                candidates = drug_filtered

        if not query.strip():
            return candidates[:top_k]

        if candidates != self.sections:
            # Build a tiny temporary matrix over drug-specific docs for more precise ranking.
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            docs = [self._doc_text(s) for s in candidates]
            matrix = vectorizer.fit_transform(docs)
            scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [s for s, _ in ranked[:top_k]]

        if self.matrix is None:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(zip(self.sections, scores), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:top_k]]

    def label_status(self, drug: str | None, event_pt: str, label_evidence: list[LabelSection] | None = None) -> tuple[str, float, list[LabelSection], str]:
        evidence = label_evidence if label_evidence is not None else self.search(drug, event_pt, top_k=3)
        if not evidence:
            return "unknown", 0.4, [], "No label evidence available."

        best = 0.0
        best_term = None
        for section in evidence:
            text = normalize_text(section.text)
            pt_norm = normalize_text(event_pt)
            if pt_norm and pt_norm in text:
                negated_direct = any(phrase in text for phrase in [f"{pt_norm} is not", f"{pt_norm} not", f"not {pt_norm}", f"no {pt_norm}"])
                if not negated_direct:
                    return "listed", 0.92, evidence, f"The reported event '{event_pt}' appears directly in retrieved label text."
            for adr in section.adverse_reactions:
                score = fuzz.token_set_ratio(event_pt, adr) / 100.0
                if score > best:
                    best = score
                    best_term = adr

        if best >= 0.82:
            return "listed", best, evidence, f"The event is similar to labelled ADR '{best_term}'."
        if best >= 0.65:
            return "unknown", best, evidence, f"The event partially matches labelled ADR '{best_term}', but reviewer confirmation is needed."
        return "unlisted", 0.7, evidence, "No close match to retrieved labelled adverse reactions."
