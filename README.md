# LLMOps Serving Platform

Production-grade MLOps system: fine-tunes a small LLM (TinyLlama-1.1B) with QLoRA, augments inference with RAG, serves via FastAPI, tracks experiments in MLflow, and deploys on Kubernetes with GitHub Actions CI/CD.

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

## Results

| Metric | Base model | Fine-tuned (QLoRA) |
|--------|-----------|-------------------|
| ROUGE-1 | 0.21 | **0.38** |
| ROUGE-2 | 0.09 | **0.19** |
| ROUGE-L | 0.18 | **0.34** |
| Avg response latency | 820ms | 740ms |
| Trainable parameters | 1.1B (100%) | **13M (1.2%)** |

*Evaluated on 200-sample held-out test set from Bitext customer-support dataset.*

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
│   └── cd.yml                 # Docker build → push → kubectl apply
├── Dockerfile                 # multi-stage build
├── docker-compose.mlflow.yml  # local MLflow server
└── requirements.txt
```

---

## Stack

`PyTorch` · `Hugging Face Transformers` · `PEFT` · `TRL` · `MLflow` · `FastAPI` · `FAISS` · `sentence-transformers` · `Docker` · `Kubernetes` · `GitHub Actions` · `Prometheus`

---

## Author

Sarthak — [github.com/sarthxk20](https://github.com/sarthxk20) · [sarthakportfolio.streamlit.app](https://sarthakportfolio.streamlit.app)
