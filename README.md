# LLMOps Serving Platform

[![CI](https://github.com/sarthxk20/llmops-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/sarthxk20/llmops-platform/actions/workflows/ci.yml)
[![CD](https://github.com/sarthxk20/llmops-platform/actions/workflows/cd.yml/badge.svg)](https://github.com/sarthxk20/llmops-platform/actions/workflows/cd.yml)
[![Docker Image](https://img.shields.io/badge/ghcr.io-llmops--platform-blue?logo=docker)](https://github.com/sarthxk20/llmops-platform/pkgs/container/llmops-platform)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

Production-grade MLOps system: fine-tunes a small LLM (TinyLlama-1.1B) with QLoRA, augments inference with RAG, serves via FastAPI, tracks experiments in MLflow, and deploys on Kubernetes with GitHub Actions CI/CD.

**1.2% of parameters trained → 138% relative improvement in ROUGE-1**, with a fully automated test → build → push pipeline.

---

## Architecture

```
GitHub Actions CI/CD
    ↓ (test → build → push → kubectl apply)
┌──────────────────────────────────────────────────┐
│  Data Pipeline       │  Fine-tuning (QLoRA)      │
│  HuggingFace →       │  PEFT + TRL + MLflow      │
│  tokenise → Arrow    │  Model Registry           │
├──────────────────────────────────────────────────┤
│  FastAPI Server      │  RAG Pipeline             │
│  /generate /embed    │  FAISS + sentence-        │
│  /health  /ready     │  transformers             │
├──────────────────────────────────────────────────┤
│  Kubernetes          │  Monitoring               │
│  Deployment + HPA    │  Prometheus + drift       │
│  ConfigMap + PVC     │  detection                │
└──────────────────────────────────────────────────┘
```

---

## CI/CD Pipeline

| Stage | Tool | Status |
|-------|------|--------|
| Lint | ruff | ✅ Passing |
| Unit tests | pytest (27 tests) | ✅ Passing |
| Build | Docker multi-stage build | ✅ Automated |
| Push | GitHub Actions → GHCR | ✅ Automated |
| Deploy | kubectl → minikube | Manual (see below) |

Every push to `main` runs lint + the full test suite (`ci.yml`). The image build and push to [`ghcr.io/sarthxk20/llmops-platform`](https://github.com/sarthxk20/llmops-platform/pkgs/container/llmops-platform) is triggered on demand via `cd.yml`. Kubernetes deployment targets a local minikube cluster and is run manually — see [Deploy to Kubernetes](#deploy-to-kubernetes-minikube) below.

---

## Results

| Metric | Base model | Fine-tuned (QLoRA) | Improvement |
|--------|-----------|-------------------|-------------|
| ROUGE-1 | 0.21 | **0.50** | +138% |
| ROUGE-2 | 0.09 | **0.27** | +200% |
| ROUGE-L | 0.18 | **0.36** | +100% |
| Avg response latency | 820ms | 740ms | −10% |
| Trainable parameters | 1.1B (100%) | **13M (1.2%)** | 98.8% fewer |

*Evaluated on 500-sample validation set from Bitext customer-support dataset. Training: 3 epochs, 375 steps, loss 0.99 → 0.56. MLflow run [`ef3500f6`](https://dagshub.com/sarthxk20/llmops-platform.mlflow) tracked on DagsHub. Model registered as `llmops-TinyLlama-1.1B-Chat-v1.0-lora`.*

---

## Quick start

```bash
# 1. Start MLflow server
docker-compose -f docker-compose.mlflow.yml up -d

# 2. Preprocess data
python data/preprocess.py

# 3. Fine-tune (requires GPU)
python training/fine_tune.py --config training/config.yaml

# 4. Compare runs and promote best model
python training/compare_runs.py --promote

# 5. Build FAISS index
mkdir -p data/docs && cp your_docs/*.txt data/docs/
python serving/build_index.py

# 6. Run the API server
uvicorn serving.app:app --host 0.0.0.0 --port 8000

# 7. Test it
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction": "How do I cancel my subscription?", "use_rag": true}'
```

---

## Deploy to Kubernetes (minikube)

```bash
# Start minikube
minikube start --memory=6g --cpus=4

# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml     # edit with real values first
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Expose service
minikube service llmops-service -n llmops --url
```

---

## Project structure

```
llmops-platform/
├── data/
│   ├── preprocess.py          # tokenise + split dataset
│   └── processed/             # Arrow files (gitignored)
├── training/
│   ├── config.yaml            # all hyperparams
│   ├── fine_tune.py           # QLoRA training + MLflow logging
│   └── compare_runs.py        # leaderboard + model promotion
├── serving/
│   ├── app.py                 # FastAPI: /generate /embed /health /ready
│   ├── model_loader.py        # base model + LoRA adapter loading
│   ├── rag_pipeline.py        # FAISS retrieval + IndexBuilder
│   └── build_index.py         # CLI to build FAISS index from docs
├── monitoring/
│   └── drift_detector.py      # KS test + PSI vocab drift detection
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml        # 2 replicas, rolling update, probes
│   ├── service.yaml
│   ├── hpa.yaml               # autoscale 1→5 pods on CPU/memory
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── pvc.yaml
├── tests/
│   ├── test_preprocessing.py
│   └── test_serving.py
├── .github/workflows/
│   ├── ci.yml                 # pytest + ruff on every push
│   └── cd.yml                 # Docker build → push to GHCR
├── models/
│   └── checkpoints/
│       └── best_adapter/      # QLoRA adapter weights (adapter_model.safetensors + config)
├── Dockerfile                 # multi-stage build
├── docker-compose.mlflow.yml  # local MLflow server
└── requirements.txt
```

---

## Stack

`PyTorch` · `Hugging Face Transformers` · `PEFT` · `TRL` · `MLflow` · `DagsHub` · `FastAPI` · `FAISS` · `sentence-transformers` · `Docker` · `Kubernetes` · `GitHub Actions` · `Prometheus`

---

## Author

Sarthak — [github.com/sarthxk20](https://github.com/sarthxk20) · [sarthakportfolio.streamlit.app](https://sarthakportfolio.streamlit.app)
