from __future__ import annotations

import argparse
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA supervised fine-tuning for PV-Reasoner.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--output_dir", default="outputs/pv-reasoner")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true", help="Enable bitsandbytes 4-bit loading. Requires CUDA Linux.")
    return parser.parse_args()


def format_messages(example: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    messages = example["messages"]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    dataset = load_dataset("json", data_files=data_files)

    def _format(batch: dict[str, Any]) -> list[str]:
        return [format_messages({"messages": messages}, tokenizer) for messages in batch["messages"]]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps if args.num_train_epochs is None else -1,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs is not None else 1,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps" if args.eval_file else "no",
        eval_steps=100 if args.eval_file else None,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        formatting_func=_format,
        max_seq_length=args.max_seq_length,
        args=training_args,
        packing=False,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
