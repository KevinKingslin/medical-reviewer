# PV-Reasoner: Evidence-Grounded Medical Review Assistant

PV-Reasoner is a hackathon-ready pharmacovigilance reviewer copilot. It reads an adverse event case report and produces a structured, auditable review packet:

- seriousness assessment
- MedDRA Preferred Term suggestions
- product-label listed/unlisted/unknown status
- simple lab abnormality support
- causality support and weakening factors
- missing follow-up questions

The system is designed for a fine-tuning track. It includes a runnable rule/retrieval baseline, a dataset builder, a QLoRA supervised fine-tuning script, an inference wrapper, evaluation code, and a Streamlit demo UI.

Important: this is a reviewer-assist prototype, not a medical decision system. A qualified medical reviewer must verify every suggestion.

## Architecture

```text
Case narrative / structured fields
        |
        v
Case normalization + lab parsing
        |
        +--> seriousness rules
        +--> MedDRA candidate retrieval/fuzzy matching
        +--> product-label retrieval
        |
        v
Fine-tuned reviewer LLM or baseline reviewer
        |
        v
Strict JSON validation + audit trail
        |
        v
Streamlit reviewer dashboard
```

## Fast start

```bash
cd pv_reasoner
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_demo_data.py
streamlit run src/pv_reasoner/app/streamlit_app.py
```

## Generate supervised fine-tuning data

This creates JSONL examples from bundled demo cases and optional openFDA examples.

```bash
python scripts/make_demo_data.py
python -m pv_reasoner.data.build_sft_dataset \
  --cases data/demo/demo_cases.jsonl \
  --labels data/labels/sample_label_sections.jsonl \
  --out data/processed/sft_train.jsonl
```

Optional openFDA download:

```bash
python -m pv_reasoner.data.openfda_fetch \
  --limit 100 \
  --out data/raw/openfda_events.jsonl
```

You can set `OPENFDA_API_KEY` in `.env`, but small pulls work without a key.

## Fine-tune with QLoRA

Use a compact instruct model for the hackathon. Qwen2.5-1.5B-Instruct or Phi-3-mini are good quick starts. A 7B/8B model needs a proper GPU.

```bash
python -m pv_reasoner.training.finetune_qlora \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/sft_train.jsonl \
  --output_dir outputs/pv-reasoner-qwen15b \
  --max_steps 200 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
```

## Run inference from a fine-tuned adapter

```bash
python -m pv_reasoner.inference.run_case \
  --case_json data/demo/example_case.json \
  --labels data/labels/sample_label_sections.jsonl \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/pv-reasoner-qwen15b
```

Without `--base_model` and `--adapter`, it uses the deterministic baseline reviewer.

## Evaluate

```bash
python -m pv_reasoner.eval.evaluate \
  --cases data/demo/demo_cases.jsonl \
  --labels data/labels/sample_label_sections.jsonl \
  --predictions_out outputs/predictions.jsonl
```

Metrics include seriousness accuracy, label-status accuracy, top-3 MedDRA hit rate, JSON validity, and evidence coverage.

## Data sources to use in the full hackathon version

- openFDA / FAERS drug event API: seriousness-like fields and reaction terms.
- DailyMed/openFDA drug label API: current product label sections.
- SPL-ADR-200db and TAC 2017 ADR: ADR extraction and label-linked MedDRA supervision.
- ADE-Corpus-V2 or CADEC: extra adverse-event extraction examples.

## MedDRA note

MedDRA is proprietary/licensed. This repo includes only a tiny seed list of common demonstration terms. For a real system, replace `data/labels/seed_meddra_terms.txt` with a properly licensed MedDRA dictionary and do not redistribute it.

## Project layout

```text
pv_reasoner/
  src/pv_reasoner/
    app/streamlit_app.py
    data/build_sft_dataset.py
    data/openfda_fetch.py
    data/synthetic.py
    eval/evaluate.py
    inference/reviewer.py
    inference/run_case.py
    retrieval/label_store.py
    retrieval/meddra.py
    training/finetune_qlora.py
    schemas.py
    prompts.py
    lab_rules.py
    seriousness.py
    utils.py
  scripts/make_demo_data.py
  data/demo/
  data/labels/
  tests/
```
