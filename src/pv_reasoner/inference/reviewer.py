from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pv_reasoner.lab_rules import assess_labs
from pv_reasoner.prompts import build_user_prompt, SYSTEM_PROMPT
from pv_reasoner.retrieval.label_store import LabelStore
from pv_reasoner.retrieval.meddra import MeddraCandidateRetriever
from pv_reasoner.schemas import (
    CaseInput,
    CausalitySupport,
    EvidenceSpan,
    LabelingStatus,
    MeddraSuggestion,
    ReviewPacket,
)
from pv_reasoner.seriousness import assess_seriousness
from pv_reasoner.utils import extract_json_object, normalize_text


@dataclass
class ReviewerConfig:
    base_model: str | None = None
    adapter: str | None = None
    max_new_tokens: int = 900
    temperature: float = 0.1
    device_map: str = "auto"


class BaselineReviewer:
    """Deterministic reviewer used before/without fine-tuning."""

    def __init__(self, label_store: LabelStore, meddra: MeddraCandidateRetriever | None = None):
        self.label_store = label_store
        self.meddra = meddra or MeddraCandidateRetriever()

    def review(self, case: CaseInput) -> ReviewPacket:
        seriousness = assess_seriousness(case)
        lab_findings = assess_labs(case.labs, case.reactions)
        candidates = self.meddra.candidates(case.narrative, case.reactions, top_k=5)

        suggestions: list[MeddraSuggestion] = []
        event_phrases = case.reactions or candidates[:1] or ["Adverse event"]
        for phrase in event_phrases[:3]:
            pt, score = self.meddra.best_match(phrase)
            if score < 0.55 and candidates:
                pt = candidates[0]
                score = 0.55
            suggestions.append(
                MeddraSuggestion(
                    verbatim=phrase,
                    pt=pt,
                    confidence=min(max(score, 0.45), 0.95),
                    evidence=[EvidenceSpan(text=phrase, source="case_reactions" if case.reactions else "case_narrative")],
                    reason="Closest available demo MedDRA Preferred Term candidate.",
                )
            )

        primary_pt = suggestions[0].pt
        label_sections = self.label_store.search(case.suspect_drug, primary_pt + " " + case.narrative, top_k=3)
        status, conf, evidence_sections, reason = self.label_store.label_status(case.suspect_drug, primary_pt, label_sections)
        label_status = LabelingStatus(
            status=status, 
            confidence=conf,
            evidence=[EvidenceSpan(text=s.text[:600], source=f"label:{s.drug}:{s.section}") for s in evidence_sections],
            reason=reason,
        )

        causality = self._causality(case, seriousness.is_serious, label_status.status, lab_findings)
        summary = self._summary(case, primary_pt, seriousness.is_serious)
        return ReviewPacket(
            case_id=case.case_id,
            case_summary=summary,
            seriousness=seriousness,
            meddra_suggestions=suggestions,
            lab_findings=lab_findings,
            labelling_status=label_status,
            causality_support=causality,
            reviewer_follow_up_questions=self._follow_up_questions(case, causality.missing_information),
        )

    def _summary(self, case: CaseInput, primary_pt: str, is_serious: bool) -> str:
        drug = case.suspect_drug or "the suspect drug"
        seriousness = "serious" if is_serious else "non-serious based on available information"
        return f"Reported event '{primary_pt}' after exposure to {drug}; case is {seriousness}."

    def _causality(self, case: CaseInput, serious: bool, label_status: str, labs: list[Any]) -> CausalitySupport:
        text = normalize_text(case.narrative)
        supporting: list[str] = []
        weakening: list[str] = []
        missing: list[str] = []

        if any(word in text for word in ["after", "following", "started", "initiated", "days after", "weeks after"]):
            supporting.append("Temporal association is described between drug exposure and event onset.")
        else:
            missing.append("Exact start date and event onset date")

        if any(word in text for word in ["improved after stopping", "resolved after stopping", "dechallenge positive", "withdrawn and recovered"]):
            supporting.append("Positive dechallenge is described.")
        else:
            missing.append("Outcome after suspect drug withdrawal/dechallenge")

        if any(word in text for word in ["rechallenged", "rechallenge positive", "recurred after restart"]):
            supporting.append("Positive rechallenge is described.")
        else:
            missing.append("Rechallenge information")

        if any(word in text for word in ["infection", "alcohol", "gallstone", "trauma", "comorbidity", "other medication"]):
            weakening.append("Possible alternative cause or confounder is mentioned.")
        else:
            missing.append("Alternative causes and relevant medical history")

        if label_status == "listed":
            supporting.append("The event appears consistent with retrieved product-label evidence.")

        if any(l.supports_event for l in labs):
            supporting.append("Abnormal lab finding supports the reported event.")

        if "Positive rechallenge is described." in supporting and "Positive dechallenge is described." in supporting:
            category = "probable"
            confidence = 0.75
        elif supporting:
            category = "possible"
            confidence = 0.62
        else:
            category = "unassessable"
            confidence = 0.5

        if weakening and category == "probable":
            category = "possible"
            confidence = 0.6

        return CausalitySupport(
            category=category,
            confidence=confidence,
            supporting_factors=supporting,
            weakening_factors=weakening,
            missing_information=list(dict.fromkeys(missing)),
        )

    def _follow_up_questions(self, case: CaseInput, missing: list[str]) -> list[str]:
        questions = []
        mapping = {
            "Exact start date and event onset date": "What were the exact suspect-drug start date and adverse-event onset date?",
            "Outcome after suspect drug withdrawal/dechallenge": "Was the suspect drug stopped, and did the event improve afterward?",
            "Rechallenge information": "Was the suspect drug restarted, and did the event recur?",
            "Alternative causes and relevant medical history": "Were alternative causes, medical history, and concomitant medications assessed?",
        }
        for item in missing:
            if item in mapping:
                questions.append(mapping[item])
        if not case.labs:
            questions.append("Are relevant laboratory values available to support or rule out the diagnosis?")
        return list(dict.fromkeys(questions))[:6]


class LLMReviewer:
    """Fine-tuned reviewer wrapper. Falls back to strict validation after generation."""

    def __init__(self, label_store: LabelStore, config: ReviewerConfig, meddra: MeddraCandidateRetriever | None = None):
        self.label_store = label_store
        self.meddra = meddra or MeddraCandidateRetriever()
        self.config = config
        if not config.base_model:
            raise ValueError("base_model is required for LLMReviewer")
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            device_map=self.config.device_map,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        if self.config.adapter:
            if PeftModel is None:
                raise ImportError("peft is required to load LoRA adapters")
            self.model = PeftModel.from_pretrained(self.model, self.config.adapter)
        self.model.eval()

    def review(self, case: CaseInput) -> ReviewPacket:
        candidates = self.meddra.candidates(case.narrative, case.reactions, top_k=10)
        label_sections = self.label_store.search(case.suspect_drug, " ".join(candidates + case.reactions), top_k=4)
        label_evidence = [section.to_evidence() for section in label_sections]
        user_prompt = build_user_prompt(case, meddra_candidates=candidates, label_evidence=label_evidence)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = SYSTEM_PROMPT + "\n\n" + user_prompt + "\n\nJSON:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        data = extract_json_object(generated)
        return ReviewPacket.model_validate(data)


def make_reviewer(label_path: str, base_model: str | None = None, adapter: str | None = None) -> BaselineReviewer | LLMReviewer:
    label_store = LabelStore.from_jsonl(label_path)
    if base_model:
        return LLMReviewer(label_store, ReviewerConfig(base_model=base_model, adapter=adapter))
    return BaselineReviewer(label_store)
