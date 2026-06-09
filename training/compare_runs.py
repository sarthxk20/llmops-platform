"""
Compare MLflow runs and promote the best model to 'Production' stage.

Usage:
    python training/compare_runs.py --experiment "llmops-customer-support-finetuning"
"""

import argparse
import json
import logging

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def get_best_run(experiment_name: str, metric: str = "eval.rougeL", higher_is_better: bool = True):
    client = MlflowClient()
    exp    = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{metric} {'DESC' if higher_is_better else 'ASC'}"],
        max_results=20,
    )

    if not runs:
        log.warning("No finished runs found.")
        return None

    log.info(f"\n{'='*60}")
    log.info(f"{'Run ID':<32} {'Loss':>8} {'ROUGE-L':>8} {'R1':>8}")
    log.info(f"{'─'*60}")
    for r in runs:
        m = r.data.metrics
        log.info(
            f"{r.info.run_id:<32} "
            f"{m.get('train.final_loss', float('nan')):>8.4f} "
            f"{m.get('eval.rougeL', float('nan')):>8.4f} "
            f"{m.get('eval.rouge1', float('nan')):>8.4f}"
        )
    log.info(f"{'='*60}\n")

    best = runs[0]
    log.info(f"Best run: {best.info.run_id} — {metric}={best.data.metrics.get(metric):.4f}")
    return best


def promote_best_model(best_run, model_name: str):
    """Transition the best model version to Production stage."""
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    # Find the version from this run
    target = next((v for v in versions if v.run_id == best_run.info.run_id), None)

    if target is None:
        log.warning(f"No model version found for run {best_run.info.run_id}.")
        return

    # Archive all current Production versions
    for v in versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Archived"
            )
            log.info(f"Archived version {v.version}")

    # Promote new best to Production
    client.transition_model_version_stage(
        name=model_name,
        version=target.version,
        stage="Production",
        archive_existing_versions=False,
    )
    log.info(f"Promoted version {target.version} → Production")

    # Tag with promotion reason
    client.set_model_version_tag(
        name=model_name, version=target.version,
        key="promoted_by", value="compare_runs.py"
    )
    client.set_model_version_tag(
        name=model_name, version=target.version,
        key="best_metric", value="eval.rougeL"
    )


def export_leaderboard(runs, output_path: str = "models/leaderboard.json"):
    rows = []
    for r in runs:
        rows.append({
            "run_id":      r.info.run_id,
            "lora_r":      r.data.params.get("lora.r"),
            "lr":          r.data.params.get("train.lr"),
            "train_loss":  r.data.metrics.get("train.final_loss"),
            "rouge1":      r.data.metrics.get("eval.rouge1"),
            "rouge2":      r.data.metrics.get("eval.rouge2"),
            "rougeL":      r.data.metrics.get("eval.rougeL"),
        })
    import os; os.makedirs("models", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)
    log.info(f"Leaderboard saved → {output_path}")


def main(args):
    mlflow.set_tracking_uri(args.tracking_uri)
    best = get_best_run(args.experiment, args.metric)
    if best is None:
        return

    if args.promote:
        promote_best_model(best, args.model_name)

    client = MlflowClient()
    exp    = client.get_experiment_by_name(args.experiment)
    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
        max_results=50,
    )
    export_leaderboard(all_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",   default="llmops-customer-support-finetuning")
    parser.add_argument("--tracking_uri", default="http://localhost:5000")
    parser.add_argument("--metric",       default="eval.rougeL")
    parser.add_argument("--model_name",   default="llmops-TinyLlama-1.1B-Chat-v1.0-lora")
    parser.add_argument("--promote",      action="store_true",
                        help="Promote best model to Production stage")
    args = parser.parse_args()
    main(args)
