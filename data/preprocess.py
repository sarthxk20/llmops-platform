"""
Data preprocessing pipeline for LLMOps fine-tuning.
Domain: Customer support (Bitext customer support dataset via HuggingFace).
Outputs tokenised Arrow files to data/processed/.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def format_example(example: Dict) -> Dict:
    """Convert a raw dataset row into instruction-tuning format."""
    instruction = example.get("instruction", "")
    input_text  = example.get("input", "")
    output      = example.get("output", example.get("response", ""))

    # Bitext customer-support specific mapping
    if "utterance" in example and "response" in example:
        instruction = "Respond to the following customer support query."
        input_text  = example["utterance"]
        output      = example["response"]

    return {
        "text": PROMPT_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output,
        )
    }


def tokenize_fn(examples, tokenizer, max_length: int):
    tokenised = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    labels = [
        [(t if t != tokenizer.pad_token_id else -100) for t in ids]
        for ids in tokenised["input_ids"]
    ]
    tokenised["labels"] = labels
    return tokenised


def load_and_split(dataset_name: str, dataset_config: str, split_ratio: float):
    log.info(f"Loading dataset: {dataset_name} ({dataset_config})")
    raw = load_dataset(dataset_name, dataset_config, trust_remote_code=True)

    data = raw["train"] if "train" in raw else list(raw.values())[0]
    data = data.shuffle(seed=42)

    n       = len(data)
    n_train = int(n * split_ratio)
    n_val   = int(n * (1 - split_ratio) / 2)

    train = data.select(range(n_train))
    val   = data.select(range(n_train, n_train + n_val))
    test  = data.select(range(n_train + n_val, n))

    log.info(f"Splits — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return DatasetDict({"train": train, "validation": val, "test": test})


def validate_dataset(dataset: DatasetDict) -> None:
    for split, ds in dataset.items():
        assert len(ds) > 0, f"Split '{split}' is empty!"
        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert "attention_mask" in sample
    log.info("Dataset validation passed.")


def save_stats(dataset: DatasetDict, out_dir: Path, tokenizer) -> dict:
    stats = {}
    for split, ds in dataset.items():
        lengths = [
            sum(1 for t in row["input_ids"] if t != tokenizer.pad_token_id)
            for row in ds
        ]
        stats[split] = {
            "n_examples": len(ds),
            "mean_tokens": round(sum(lengths) / len(lengths), 1),
            "max_tokens":  max(lengths),
            "min_tokens":  min(lengths),
        }
    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved → {stats_path}")
    return stats


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits    = load_and_split(args.dataset_name, args.dataset_config, args.train_ratio)
    formatted = splits.map(format_example, remove_columns=splits["train"].column_names)

    log.info(f"Tokenising (max_length={args.max_length})...")
    tokenised = formatted.map(
        lambda ex: tokenize_fn(ex, tokenizer, args.max_length),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["text"],
    )

    validate_dataset(tokenised)
    tokenised.save_to_disk(str(out_dir))
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    save_stats(tokenised, out_dir, tokenizer)
    log.info("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",     default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset_name",   default="bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    parser.add_argument("--dataset_config", default="default")
    parser.add_argument("--output_dir",     default="data/processed")
    parser.add_argument("--max_length",     type=int,   default=512)
    parser.add_argument("--train_ratio",    type=float, default=0.85)
    parser.add_argument("--batch_size",     type=int,   default=1000)
    args = parser.parse_args()
    main(args)
