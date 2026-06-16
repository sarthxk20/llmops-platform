"""
RAG pipeline: FAISS-backed retrieval + context injection.

Build the index:
    python serving/build_index.py --docs_dir data/docs --index_path data/faiss_index

Use at inference:
    rag = RAGPipeline(index_path="data/faiss_index")
    rag.load()
    docs = rag.retrieve("How do I cancel my subscription?", k=3)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 22M params, fast


class RAGPipeline:
    def __init__(
        self,
        index_path:   str,
        embed_model:  str = EMBED_MODEL,
        device:       str = "cpu",
    ):
        self.index_path  = Path(index_path)
        self.embed_model = embed_model
        self.device      = device

        self.encoder: Optional[SentenceTransformer] = None
        self.index:   Optional[faiss.Index]         = None
        self.docs:    List[str]                     = []
        self.metadata: List[dict]                   = []

    def load(self) -> None:
        """Load FAISS index + document store + embedding model."""
        log.info(f"Loading embedding model: {self.embed_model}")
        self.encoder = SentenceTransformer(self.embed_model, device=self.device)

        index_file = self.index_path / "index.faiss"
        docs_file  = self.index_path / "documents.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")
        if not docs_file.exists():
            raise FileNotFoundError(f"Document store not found: {docs_file}")

        self.index = faiss.read_index(str(index_file))
        with open(docs_file) as f:
            store         = json.load(f)
            self.docs     = store["documents"]
            self.metadata = store.get("metadata", [{}] * len(self.docs))

        log.info(f"RAG index loaded — {self.index.ntotal} vectors, {len(self.docs)} docs")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Embed query and return top-k most relevant document chunks."""
        if self.encoder is None or self.index is None:
            raise RuntimeError("RAGPipeline not loaded. Call .load() first.")

        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        distances, indices = self.index.search(query_vec, k)
        retrieved = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            log.debug(f"Retrieved doc {idx} (distance={dist:.4f}): {self.docs[idx][:80]}...")
            retrieved.append(self.docs[idx])

        return retrieved

    def add_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        """Add new documents to the in-memory index (does not persist)."""
        if self.encoder is None:
            raise RuntimeError("RAGPipeline not loaded.")
        vecs = self.encoder.encode(documents, normalize_embeddings=True, show_progress_bar=True)
        vecs = np.array(vecs, dtype=np.float32)
        self.index.add(vecs)
        self.docs.extend(documents)
        self.metadata.extend(metadata or [{}] * len(documents))
        log.info(f"Added {len(documents)} docs — index now has {self.index.ntotal} vectors")


class IndexBuilder:
    """Build a FAISS index from a directory of .txt files."""

    def __init__(self, embed_model: str = EMBED_MODEL, chunk_size: int = 300, overlap: int = 50):
        self.embed_model = embed_model
        self.chunk_size  = chunk_size
        self.overlap     = overlap

    def chunk_text(self, text: str) -> List[str]:
        words  = text.split()
        chunks = []
        start  = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.overlap
        return [c for c in chunks if len(c.strip()) > 0]

    def build(self, docs_dir: str, index_path: str) -> None:
        out = Path(index_path)
        out.mkdir(parents=True, exist_ok=True)

        encoder = SentenceTransformer(self.embed_model)

        all_chunks, all_meta = [], []
        for f in Path(docs_dir).glob("**/*.txt"):
            text   = f.read_text(encoding="utf-8", errors="ignore")
            chunks = self.chunk_text(text)
            all_chunks.extend(chunks)
            all_meta.extend([{"source": str(f)}] * len(chunks))
            log.info(f"Chunked {f.name} → {len(chunks)} chunks")

        if not all_chunks:
            raise ValueError(f"No .txt files found in {docs_dir}")

        log.info(f"Encoding {len(all_chunks)} chunks...")
        vecs = encoder.encode(all_chunks, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=64)
        vecs = np.array(vecs, dtype=np.float32)

        dim   = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)   # inner-product = cosine on L2-normed vecs
        index.add(vecs)
        faiss.write_index(index, str(out / "index.faiss"))

        with open(out / "documents.json", "w") as f:
            json.dump({"documents": all_chunks, "metadata": all_meta}, f)

        log.info(f"Index built — {index.ntotal} vectors → {out}")
