from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from pv_reasoner.inference.reviewer import make_reviewer
from pv_reasoner.schemas import CaseInput

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS = ROOT / "data" / "labels" / "sample_label_sections.jsonl"

st.set_page_config(page_title="PV-Reasoner", page_icon="🧪", layout="wide")
st.title("PV-Reasoner: Medical Review Assistant")
st.caption("Reviewer-assist prototype. Human medical review is required.")

with st.sidebar:
    st.header("Model")
    use_llm = st.checkbox("Use fine-tuned/base LLM", value=False)
    base_model = st.text_input("Base model", value=os.getenv("PV_REASONER_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    adapter = st.text_input("LoRA adapter path", value=os.getenv("PV_REASONER_ADAPTER_DIR", ""))
    label_path = st.text_input("Label JSONL", value=str(DEFAULT_LABELS))
    st.info("Leave LLM unchecked for the fast deterministic demo baseline.")

examples_path = ROOT / "data" / "demo" / "demo_cases.jsonl"
examples = []
if examples_path.exists():
    examples = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines() if line.strip()]

example_names = [f"{row['case_id']} — {row.get('suspect_drug', 'unknown')}" for row in examples]
selected = st.selectbox("Load demo case", ["Manual"] + example_names)

if selected != "Manual" and examples:
    row = examples[example_names.index(selected)]
else:
    row = {
        "case_id": "MANUAL-001",
        "suspect_drug": "Atorvastatin",
        "patient_age": "64 years",
        "patient_sex": "female",
        "reactions": ["Rhabdomyolysis"],
        "report_outcomes": ["hospitalization"],
        "labs": {"CK": "6200", "creatinine": "1.6"},
        "narrative": "A 64-year-old female started atorvastatin two weeks ago. She developed severe muscle pain and CK 6200. She was admitted to hospital for suspected rhabdomyolysis.",
    }

left, right = st.columns([0.42, 0.58])

with left:
    st.subheader("Case input")
    case_id = st.text_input("Case ID", value=row.get("case_id", "MANUAL-001"))
    suspect_drug = st.text_input("Suspect drug", value=row.get("suspect_drug") or "")
    patient_age = st.text_input("Patient age", value=row.get("patient_age") or "")
    patient_sex = st.selectbox("Patient sex", ["", "male", "female", "unknown"], index=["", "male", "female", "unknown"].index(row.get("patient_sex") or "") if (row.get("patient_sex") or "") in ["", "male", "female", "unknown"] else 0)
    reactions_text = st.text_input("Reported reactions, comma-separated", value=", ".join(row.get("reactions", [])))
    outcomes_text = st.text_input("Outcomes, comma-separated", value=", ".join(row.get("report_outcomes", [])))
    labs_text = st.text_area("Labs as JSON", value=json.dumps(row.get("labs", {}), indent=2), height=120)
    narrative = st.text_area("Narrative", value=row.get("narrative", ""), height=240)

    run = st.button("Generate review packet", type="primary")

with right:
    st.subheader("Review output")
    if run:
        try:
            labs = json.loads(labs_text) if labs_text.strip() else {}
            case = CaseInput(
                case_id=case_id,
                narrative=narrative,
                suspect_drug=suspect_drug or None,
                patient_age=patient_age or None,
                patient_sex=patient_sex or None,
                reactions=[x.strip() for x in reactions_text.split(",") if x.strip()],
                report_outcomes=[x.strip() for x in outcomes_text.split(",") if x.strip()],
                labs=labs,
                source="streamlit",
            )
            reviewer = make_reviewer(label_path, base_model=base_model if use_llm else None, adapter=adapter or None)
            packet = reviewer.review(case)
            data = packet.model_dump()
            st.success("Review packet generated")

            c1, c2, c3 = st.columns(3)
            c1.metric("Serious", "Yes" if packet.seriousness.is_serious else "No")
            c2.metric("Primary PT", packet.meddra_suggestions[0].pt)
            c3.metric("Label status", packet.labelling_status.status)

            st.markdown("**Case summary**")
            st.write(packet.case_summary)

            st.markdown("**Seriousness**")
            st.json(packet.seriousness.model_dump())

            st.markdown("**MedDRA suggestions**")
            st.dataframe([s.model_dump() | {"evidence": "; ".join(e.text for e in s.evidence)} for s in packet.meddra_suggestions], use_container_width=True)

            st.markdown("**Labelling status**")
            st.json(packet.labelling_status.model_dump())

            st.markdown("**Causality support**")
            st.json(packet.causality_support.model_dump())

            st.markdown("**Follow-up questions**")
            for q in packet.reviewer_follow_up_questions:
                st.write(f"- {q}")

            with st.expander("Raw JSON"):
                st.code(json.dumps(data, indent=2), language="json")
        except Exception as exc:
            st.error(f"Could not generate review: {exc}")
    else:
        st.write("Enter or load a case, then generate a review packet.")
