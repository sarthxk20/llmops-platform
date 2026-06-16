"""
QLoRA fine-tuning pipeline with full MLflow experiment tracking.

Usage:
    python training/fine_tune.py --config training/config.yaml

What this does:
  1. Loads config + preprocessed dataset from disk
  2. Quantises base model in 4-bit (QLoRA)
  3. Attaches LoRA adapters via PEFT
  4. Trains with HuggingFace Trainer
  5. Logs all params, metrics, and artifacts to MLflow
  6. Registers best checkpoint in MLflow Model Registry
  7. Runs ROUGE evaluation on test split and logs scores
"""

import json
import logging
import argparse
from pathlib import Path

import yaml
import torch
import mlflow
import mlflow.pytorch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from evaluate import load as load_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_quantised_model(cfg: dict):
    """Load base model with 4-bit QLoRA quantisation."""
    q = cfg["quantisation"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_config,
        device_map=cfg["model"]["device_map"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        trust_remote_code=True,
    )
    model.config.use_cache = False                 # required for gradient checkpointing
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    log.info(f"Base model loaded: {cfg['model']['name']}")
    return model


def attach_lora(model, cfg: dict):
    """Wrap model with LoRA adapters."""
    lc = cfg["lora"]
    lora_config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        target_modules=lc["target_modules"],
        lora_dropout=lc["lora_dropout"],
        bias=lc["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    log.info(
        f"LoRA adapters attached — "
        f"trainable: {trainable:,} / {total:,} params "
        f"({100 * trainable / total:.2f}%)"
    )
    return model, lora_config


# ── MLflow callbacks ───────────────────────────────────────────────────────────

class MLflowMetricsCallback:
    """HuggingFace Trainer callback that streams metrics to an active MLflow run."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=step)


# ── ROUGE evaluation ───────────────────────────────────────────────────────────

def evaluate_rouge(model, tokenizer, test_dataset, cfg: dict) -> dict:
    """Run generation-based ROUGE eval on a subset of the test split."""
    rouge = load_metric("rouge")
    ec    = cfg["evaluation"]
    n     = min(ec["num_eval_samples"], len(test_dataset))
    subset = test_dataset.select(range(n))

    model.eval()
    predictions, references = [], []

    log.info(f"Running ROUGE eval on {n} test examples...")
    for row in subset:
        # Find where response starts and trim
        full_text = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
        prompt    = full_text.split("### Response:")[0] + "### Response:\n"
        response  = full_text.split("### Response:")[-1].strip()

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=ec["max_new_tokens"],
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions.append(generated.strip())
        references.append(response)

    scores = rouge.compute(predictions=predictions, references=references)
    log.info(f"ROUGE scores: {scores}")
    return {k: round(float(v), 4) for k, v in scores.items()}


# ── Training ───────────────────────────────────────────────────────────────────

def build_training_args(cfg: dict) -> TrainingArguments:
    tc = cfg["training"]
    return TrainingArguments(
        output_dir=tc["output_dir"],
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        per_device_eval_batch_size=tc["per_device_eval_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        max_grad_norm=tc["max_grad_norm"],
        fp16=tc["fp16"],
        evaluation_strategy=tc["evaluation_strategy"],
        eval_steps=tc["eval_steps"],
        save_strategy=tc["save_strategy"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        load_best_model_at_end=tc["load_best_model_at_end"],
        metric_for_best_model=tc["metric_for_best_model"],
        greater_is_better=tc["greater_is_better"],
        logging_steps=tc["logging_steps"],
        report_to=tc["report_to"],
        seed=tc["seed"],
        dataloader_num_workers=tc["dataloader_num_workers"],
        gradient_checkpointing=True,
    )


def main(args):
    cfg = load_config(args.config)

    # ── MLflow setup ──────────────────────────────────────────────────────────
    mlf = cfg["mlflow"]
    mlflow.set_tracking_uri(mlf["tracking_uri"])
    mlflow.set_experiment(mlf["experiment_name"])

    with mlflow.start_run(tags=mlf.get("run_tags", {})) as run:
        log.info(f"MLflow run ID: {run.info.run_id}")

        # ── Log all config as params ──────────────────────────────────────────
        flat_params = {
            "model.name":             cfg["model"]["name"],
            "lora.r":                 cfg["lora"]["r"],
            "lora.alpha":             cfg["lora"]["lora_alpha"],
            "lora.dropout":           cfg["lora"]["lora_dropout"],
            "lora.target_modules":    ",".join(cfg["lora"]["target_modules"]),
            "quant.4bit":             cfg["quantisation"]["load_in_4bit"],
            "quant.type":             cfg["quantisation"]["bnb_4bit_quant_type"],
            "train.epochs":           cfg["training"]["num_train_epochs"],
            "train.lr":               cfg["training"]["learning_rate"],
            "train.batch_size":       cfg["training"]["per_device_train_batch_size"],
            "train.grad_accum":       cfg["training"]["gradient_accumulation_steps"],
            "train.warmup_ratio":     cfg["training"]["warmup_ratio"],
            "train.scheduler":        cfg["training"]["lr_scheduler_type"],
            "data.max_seq_length":    cfg["data"]["max_seq_length"],
        }
        mlflow.log_params(flat_params)

        # ── Load dataset ──────────────────────────────────────────────────────
        log.info(f"Loading dataset from {cfg['data']['processed_dir']}")
        dataset = load_from_disk(cfg["data"]["processed_dir"])

        # Log dataset stats
        stats_path = Path(cfg["data"]["processed_dir"]) / "dataset_stats.json"
        if stats_path.exists():
            mlflow.log_artifact(str(stats_path), "data")
            with open(stats_path) as f:
                stats = json.load(f)
            for split, s in stats.items():
                mlflow.log_metric(f"data.{split}.n_examples", s["n_examples"])
                mlflow.log_metric(f"data.{split}.mean_tokens", s["mean_tokens"])

        # ── Load tokenizer ────────────────────────────────────────────────────
        tok_dir = Path(cfg["data"]["processed_dir"]) / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(
            str(tok_dir) if tok_dir.exists() else cfg["model"]["name"],
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Load + prepare model ──────────────────────────────────────────────
        model = load_quantised_model(cfg)
        model, lora_config = attach_lora(model, cfg)

        trainable, total = model.get_nb_trainable_parameters()
        mlflow.log_metric("model.trainable_params", trainable)
        mlflow.log_metric("model.total_params", total)
        mlflow.log_metric("model.trainable_pct", round(100 * trainable / total, 4))

        # ── Data collator ─────────────────────────────────────────────────────
        collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, pad_to_multiple_of=8
        )

        # ── Trainer ───────────────────────────────────────────────────────────
        training_args = build_training_args(cfg)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        log.info("Starting training...")
        train_result = trainer.train()

        # ── Log final training metrics ────────────────────────────────────────
        mlflow.log_metric("train.runtime_s",       train_result.metrics["train_runtime"])
        mlflow.log_metric("train.samples_per_sec", train_result.metrics["train_samples_per_second"])
        mlflow.log_metric("train.steps_per_sec",   train_result.metrics["train_steps_per_second"])
        mlflow.log_metric("train.final_loss",      train_result.metrics.get("train_loss", 0))

        # ── Save adapter ──────────────────────────────────────────────────────
        adapter_dir = Path(cfg["training"]["output_dir"]) / "best_adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        log.info(f"Adapter saved → {adapter_dir}")

        # Log adapter as MLflow artifact
        mlflow.log_artifacts(str(adapter_dir), "adapter")
        mlflow.log_artifact(args.config,        "config")

        # ── ROUGE evaluation ──────────────────────────────────────────────────
        rouge_scores = evaluate_rouge(model, tokenizer, dataset["test"], cfg)
        for metric, score in rouge_scores.items():
            mlflow.log_metric(f"eval.{metric}", score)

        # ── Register in MLflow Model Registry ─────────────────────────────────
        model_uri  = f"runs:/{run.info.run_id}/adapter"
        model_name = f"llmops-{cfg['model']['name'].split('/')[-1]}-lora"

        mlflow.register_model(model_uri=model_uri, name=model_name)
        log.info(f"Model registered in MLflow registry as '{model_name}'")

        # ── Final summary ─────────────────────────────────────────────────────
        summary = {
            "run_id":          run.info.run_id,
            "model_name":      model_name,
            "train_loss":      train_result.metrics.get("train_loss", "n/a"),
            "rouge_scores":    rouge_scores,
            "trainable_params": trainable,
        }
        summary_path = Path(cfg["training"]["output_dir"]) / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(str(summary_path), "summary")

        log.info("Training pipeline complete.")
        log.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()
    main(args)
