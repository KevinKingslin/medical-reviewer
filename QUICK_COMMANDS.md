# Quick Commands

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

# 2. Generate bundled data
python scripts/make_demo_data.py

# 3. Run app
streamlit run src/pv_reasoner/app/streamlit_app.py

# 4. Build SFT data
python -m pv_reasoner.data.build_sft_dataset \
  --cases data/demo/demo_cases.jsonl \
  --labels data/labels/sample_label_sections.jsonl \
  --out data/processed/sft_train.jsonl

# 5. Train LoRA adapter
python -m pv_reasoner.training.finetune_qlora \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/sft_train.jsonl \
  --output_dir outputs/pv-reasoner-qwen15b \
  --max_steps 200 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8

# 6. Run single-case inference
python -m pv_reasoner.inference.run_case \
  --case_json data/demo/example_case.json \
  --labels data/labels/sample_label_sections.jsonl

# 7. Evaluate baseline
python -m pv_reasoner.eval.evaluate \
  --cases data/demo/demo_cases.jsonl \
  --labels data/labels/sample_label_sections.jsonl \
  --predictions_out outputs/predictions.jsonl

# 8. Optional openFDA pull
python -m pv_reasoner.data.openfda_fetch \
  --limit 100 \
  --out data/raw/openfda_events.jsonl
```
