# Hackathon Submission Notes

## Problem
Medical reviewers receive adverse-event case reports and must assess seriousness, code the event with standardized terminology, check whether the event is known in the product label, and support causality assessment.

## Solution
PV-Reasoner converts an adverse-event case into an auditable JSON review packet:

1. seriousness assessment with evidence
2. MedDRA Preferred Term suggestions
3. label listed/unlisted/unknown status with retrieved label evidence
4. lab abnormality interpretation
5. cautious causality support
6. missing follow-up questions

## What makes it novel
The model does not simply output a final classification. It creates an evidence-grounded review packet and keeps the reviewer in control. The app shows why the model made each suggestion, making the system inspectable.

## Fine-tuning angle
The project fine-tunes an instruction model to generate strict medical-review JSON. The baseline reviewer produces weak-supervision examples from structured case fields and label retrieval; real reviewer corrections can then be used for better SFT or DPO after the hackathon.

## Demo flow
1. Open the Streamlit app.
2. Select DEMO-001 for a serious rhabdomyolysis case.
3. Generate review packet.
4. Show seriousness criterion: hospitalization.
5. Show MedDRA suggestion: Rhabdomyolysis.
6. Show label status: listed.
7. Show causality: possible, not certain, because dechallenge/rechallenge are missing.
8. Show follow-up questions.

## What is complete
- Runnable Streamlit app
- Demo data generation
- SFT JSONL dataset builder
- QLoRA training script
- Baseline reviewer
- Inference wrapper
- Evaluation script
- Tests

## What to improve after hackathon
- Replace demo MedDRA terms with a licensed MedDRA dictionary.
- Connect to DailyMed/openFDA label retrieval at runtime.
- Add human reviewer edit capture.
- Fine-tune on more FAERS/openFDA cases and ADR label datasets.
- Add unit-aware lab ranges.
