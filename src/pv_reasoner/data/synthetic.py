from __future__ import annotations

from pv_reasoner.schemas import CaseInput


def demo_cases() -> list[CaseInput]:
    return [
        CaseInput(
            case_id="DEMO-001",
            suspect_drug="Atorvastatin",
            patient_age="64 years",
            patient_sex="female",
            reactions=["Rhabdomyolysis"],
            report_outcomes=["hospitalization"],
            labs={"CK": "6200", "creatinine": "1.6"},
            narrative=(
                "A 64-year-old female started atorvastatin two weeks ago. She developed severe muscle pain "
                "and weakness, and labs showed CK 6200. She was admitted to hospital for suspected rhabdomyolysis. "
                "The report does not state whether atorvastatin was stopped."
            ),
            source="demo",
            gold={"serious": True, "primary_pt": "Rhabdomyolysis", "label_status": "listed"},
        ),
        CaseInput(
            case_id="DEMO-002",
            suspect_drug="Amoxicillin",
            patient_age="28 years",
            patient_sex="male",
            reactions=["Rash"],
            report_outcomes=[],
            labs={},
            narrative=(
                "A 28-year-old male took amoxicillin for sinus infection. Two days after starting treatment, "
                "he developed a mild generalized rash. No hospitalization, emergency treatment, or life-threatening "
                "features were reported. The rash resolved after stopping the medicine."
            ),
            source="demo",
            gold={"serious": False, "primary_pt": "Rash", "label_status": "listed"},
        ),
        CaseInput(
            case_id="DEMO-003",
            suspect_drug="Metformin",
            patient_age="72 years",
            patient_sex="female",
            reactions=["Acute kidney injury"],
            report_outcomes=["hospitalization"],
            labs={"creatinine": "3.4", "potassium": "5.8"},
            narrative=(
                "A 72-year-old female on metformin was admitted with dehydration, vomiting, acute kidney injury, "
                "creatinine 3.4, and potassium 5.8. The patient also had gastroenteritis, which may be an alternative "
                "cause. Timing of metformin start and dechallenge outcome were not provided."
            ),
            source="demo",
            gold={"serious": True, "primary_pt": "Acute kidney injury", "label_status": "listed"},
        ),
        CaseInput(
            case_id="DEMO-004",
            suspect_drug="Ibuprofen",
            patient_age="45 years",
            patient_sex="male",
            reactions=["Pancreatitis"],
            report_outcomes=["hospitalization"],
            labs={"lipase": "900", "amylase": "350"},
            narrative=(
                "A 45-year-old male used ibuprofen for back pain and five days later developed severe abdominal pain. "
                "Lipase was 900 and amylase was 350. He was hospitalized with pancreatitis. Gallstone and alcohol history "
                "were not documented."
            ),
            source="demo",
            gold={"serious": True, "primary_pt": "Pancreatitis", "label_status": "unlisted"},
        ),
        CaseInput(
            case_id="DEMO-005",
            suspect_drug="Aspirin",
            patient_age="58 years",
            patient_sex="female",
            reactions=["Gastrointestinal haemorrhage"],
            report_outcomes=["hospitalization"],
            labs={"platelets": "230000"},
            narrative=(
                "A 58-year-old female taking aspirin developed black stools and dizziness. She was admitted to hospital "
                "for gastrointestinal bleeding and received supportive treatment. Outcome after stopping aspirin is unknown."
            ),
            source="demo",
            gold={"serious": True, "primary_pt": "Gastrointestinal haemorrhage", "label_status": "listed"},
        ),
    ]
