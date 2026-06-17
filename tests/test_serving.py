"""
Tests for FastAPI endpoints and serving utilities.
Run: pytest tests/test_serving.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Model loader unit tests ────────────────────────────────────────────────────

class TestModelLoader:
    def test_model_name_includes_lora(self):
        from serving.model_loader import ModelLoader
        ml = ModelLoader("org/TinyLlama-1.1B", adapter_path="some/adapter")
        assert "lora" in ml.model_name

    def test_model_name_no_lora(self):
        from serving.model_loader import ModelLoader
        ml = ModelLoader("org/TinyLlama-1.1B")
        assert "lora" not in ml.model_name

    def test_not_loaded_initially(self):
        from serving.model_loader import ModelLoader
        ml = ModelLoader("org/TinyLlama-1.1B")
        assert ml.is_loaded is False


# ── RAG pipeline unit tests ────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_retrieve_raises_if_not_loaded(self):
        from serving.rag_pipeline import RAGPipeline
        rag = RAGPipeline(index_path="/tmp/nonexistent")
        with pytest.raises(RuntimeError, match="not loaded"):
            rag.retrieve("test query")

    def test_chunk_text_overlap(self):
        from serving.rag_pipeline import IndexBuilder
        builder = IndexBuilder(chunk_size=5, overlap=2)
        words  = "a b c d e f g h i j k".split()
        text   = " ".join(words)
        chunks = builder.chunk_text(text)
        assert len(chunks) >= 2
        first  = chunks[0].split()
        second = chunks[1].split()
        assert first[-2:] == second[:2]

    def test_chunk_filters_short_chunks(self):
        from serving.rag_pipeline import IndexBuilder
        builder = IndexBuilder(chunk_size=5, overlap=1)
        text   = "hello world foo bar baz x"
        chunks = builder.chunk_text(text)
        # With filter > 0, all non-empty chunks are kept
        for c in chunks:
            assert len(c.strip()) > 0

    def test_add_documents_raises_if_not_loaded(self):
        from serving.rag_pipeline import RAGPipeline
        rag = RAGPipeline(index_path="/tmp/nonexistent")
        with pytest.raises(RuntimeError):
            rag.add_documents(["some doc"])

    def test_index_builder_raises_on_empty_dir(self, tmp_path):
        from serving.rag_pipeline import IndexBuilder
        builder = IndexBuilder()
        with pytest.raises(ValueError, match="No .txt files"):
            builder.build(str(tmp_path), str(tmp_path / "index"))


# ── FastAPI endpoint tests ─────────────────────────────────────────────────────

class TestAPIEndpoints:
    """Test endpoints with mocked model and RAG."""

   @pytest.fixture
    def client(self):
    import serving.app as app_module
    from unittest.mock import patch

    mock_loader = MagicMock()
    mock_loader.is_loaded = True
    mock_loader.model_name = "TinyLlama+lora"
    mock_loader.device = "cpu"
    mock_loader.generate.return_value = ("I can help with that!", 12)
    mock_loader.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = ["Relevant policy: subscriptions can be cancelled anytime."]

    with patch("serving.app.ModelLoader", return_value=mock_loader), \
         patch("serving.app.RAGPipeline", return_value=mock_rag), \
         patch("os.path.exists", return_value=False):
        app_module._READY = False
        with TestClient(app_module.app) as c:
            app_module.MODEL_LOADER = mock_loader
            app_module.RAG = mock_rag
            app_module._READY = True
            yield c

    app_module.MODEL_LOADER = None
    app_module.RAG = None
    app_module._READY = False

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["rag_enabled"]  is True

    def test_ready_returns_200_when_ready(self, client):
        r = client.get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True

    def test_generate_returns_response(self, client):
        r = client.post("/generate", json={
            "instruction": "How do I cancel my subscription?",
            "use_rag": True,
        })
        assert r.status_code == 200
        body = r.json()
        assert "response" in body
        assert body["response"] == "I can help with that!"
        assert body["tokens_generated"] == 12
        assert "latency_ms" in body
        assert len(body["retrieved_docs"]) > 0

    def test_generate_without_rag(self, client):
        r = client.post("/generate", json={
            "instruction": "Hello",
            "use_rag": False,
        })
        assert r.status_code == 200
        assert r.json()["response"] == "I can help with that!"

    def test_generate_validates_empty_instruction(self, client):
        r = client.post("/generate", json={"instruction": ""})
        assert r.status_code == 422

    def test_embed_returns_embeddings(self, client):
        r = client.post("/embed", json={"texts": ["Hello world"]})
        assert r.status_code == 200
        body = r.json()
        assert "embeddings" in body
        assert len(body["embeddings"]) == 1
        assert isinstance(body["embeddings"][0], list)

    def test_embed_empty_list_fails(self, client):
        r = client.post("/embed", json={"texts": []})
        assert r.status_code == 422

    def test_ready_returns_503_when_not_ready(self, client):
        import serving.app as app_module
        app_module._READY = False
        r = client.get("/ready")
        assert r.status_code == 503
        app_module._READY = False
