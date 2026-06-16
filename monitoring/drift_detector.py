"""
Input/output drift detection for the LLM serving layer.

Compares live request distributions against a baseline captured
during evaluation. Emits alerts when drift exceeds thresholds.

Run as a cron job or standalone service:
    python monitoring/drift_detector.py --log_dir logs/ --baseline_path monitoring/baseline.json
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import ks_2samp

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ── Feature extractors ─────────────────────────────────────────────────────────

def extract_token_lengths(texts: List[str]) -> np.ndarray:
    return np.array([len(t.split()) for t in texts], dtype=float)


def extract_char_lengths(texts: List[str]) -> np.ndarray:
    return np.array([len(t) for t in texts], dtype=float)


def extract_vocab_distribution(texts: List[str], top_k: int = 200) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(text.lower().split())
    return Counter(dict(counts.most_common(top_k)))


# ── Statistical tests ──────────────────────────────────────────────────────────

def ks_drift(baseline: np.ndarray, live: np.ndarray, alpha: float = 0.05) -> Dict:
    """Kolmogorov-Smirnov test for continuous features (lengths etc.)."""
    stat, pval = ks_2samp(baseline, live)
    return {
        "test":      "ks",
        "statistic": round(float(stat), 4),
        "p_value":   round(float(pval), 4),
        "drifted":   pval < alpha,
        "baseline_mean": round(float(baseline.mean()), 2),
        "live_mean":     round(float(live.mean()), 2),
    }


def psi_score(baseline_counts: Counter, live_counts: Counter) -> Dict:
    """Population Stability Index for vocabulary distribution drift."""
    all_tokens = set(baseline_counts) | set(live_counts)
    n_base = max(sum(baseline_counts.values()), 1)
    n_live = max(sum(live_counts.values()), 1)

    psi = 0.0
    for tok in all_tokens:
        p_base = baseline_counts.get(tok, 0.5) / n_base
        p_live = live_counts.get(tok, 0.5) / n_live
        p_base = max(p_base, 1e-6)
        p_live = max(p_live, 1e-6)
        psi   += (p_live - p_base) * np.log(p_live / p_base)

    return {
        "test":    "psi",
        "score":   round(float(psi), 4),
        "drifted": psi > 0.2,          # PSI > 0.2 = significant drift
        "level":   "low" if psi < 0.1 else ("medium" if psi < 0.2 else "high"),
    }


# ── Baseline capture ───────────────────────────────────────────────────────────

def capture_baseline(texts: List[str], output_path: str) -> None:
    baseline = {
        "captured_at":     datetime.utcnow().isoformat(),
        "n_samples":       len(texts),
        "token_lengths":   extract_token_lengths(texts).tolist(),
        "char_lengths":    extract_char_lengths(texts).tolist(),
        "vocab_top200":    dict(extract_vocab_distribution(texts)),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(baseline, f, indent=2)
    log.info(f"Baseline captured ({len(texts)} samples) → {output_path}")


# ── Log parsing ────────────────────────────────────────────────────────────────

def parse_structured_logs(log_dir: str, hours: int = 24) -> Tuple[List[str], List[str]]:
    """
    Parse JSON request logs from the last N hours.
    Returns (input_instructions, output_responses).
    """
    cutoff    = datetime.utcnow() - timedelta(hours=hours)
    log_dir_p = Path(log_dir)
    inputs, outputs = [], []

    for log_file in sorted(log_dir_p.glob("requests_*.jsonl")):
        with open(log_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    ts    = datetime.fromisoformat(entry.get("timestamp", "2000-01-01"))
                    if ts < cutoff:
                        continue
                    if entry.get("instruction"):
                        inputs.append(entry["instruction"])
                    if entry.get("response"):
                        outputs.append(entry["response"])
                except (json.JSONDecodeError, ValueError):
                    continue

    log.info(f"Parsed {len(inputs)} requests from last {hours}h")
    return inputs, outputs


# ── Main drift report ──────────────────────────────────────────────────────────

def run_drift_report(baseline_path: str, log_dir: str, hours: int = 24) -> Dict:
    with open(baseline_path) as f:
        baseline = json.load(f)

    live_inputs, live_outputs = parse_structured_logs(log_dir, hours)

    if len(live_inputs) < 30:
        log.warning(f"Only {len(live_inputs)} live samples — results may be unreliable.")

    b_tok_len = np.array(baseline["token_lengths"], dtype=float)
    b_chr_len = np.array(baseline["char_lengths"],  dtype=float)
    b_vocab   = Counter(baseline["vocab_top200"])

    l_tok_len = extract_token_lengths(live_inputs)
    l_chr_len = extract_char_lengths(live_inputs)
    l_vocab   = extract_vocab_distribution(live_inputs)

    report = {
        "generated_at":  datetime.utcnow().isoformat(),
        "baseline_n":    baseline["n_samples"],
        "live_n":        len(live_inputs),
        "window_hours":  hours,
        "checks": {
            "token_length_drift": ks_drift(b_tok_len, l_tok_len),
            "char_length_drift":  ks_drift(b_chr_len, l_chr_len),
            "vocab_psi":          psi_score(b_vocab, l_vocab),
        },
    }

    any_drift = any(c.get("drifted") for c in report["checks"].values())
    report["overall_drift_detected"] = any_drift

    if any_drift:
        log.warning("DRIFT DETECTED — review report and consider re-evaluation.")
    else:
        log.info("No significant drift detected.")

    return report


def main(args):
    if args.capture_baseline:
        # Read sample texts from a newline-delimited file
        texts = Path(args.baseline_texts).read_text().splitlines()
        capture_baseline(texts, args.baseline_path)
        return

    report = run_drift_report(args.baseline_path, args.log_dir, args.hours)
    out = Path(args.output or "monitoring/drift_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Drift report → {out}")

    if args.fail_on_drift and report["overall_drift_detected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path",   default="monitoring/baseline.json")
    parser.add_argument("--log_dir",         default="logs/")
    parser.add_argument("--hours",           type=int, default=24)
    parser.add_argument("--output",          default=None)
    parser.add_argument("--capture_baseline", action="store_true")
    parser.add_argument("--baseline_texts",  default="monitoring/baseline_texts.txt")
    parser.add_argument("--fail_on_drift",   action="store_true")
    args = parser.parse_args()
    main(args)
