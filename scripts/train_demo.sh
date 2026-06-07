#!/usr/bin/env bash
set -euo pipefail
python scripts/make_demo_data.py
python -m pv_reasoner.data.build_sft_dataset \
  --cases data/demo/demo_cases.jsonl \
  --labels data/labels/sample_label_sections.jsonl \
  --out data/processed/sft_train.jsonl
python -m pv_reasoner.training.finetune_qlora \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/sft_train.jsonl \
  --output_dir outputs/pv-reasoner-qwen15b \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
