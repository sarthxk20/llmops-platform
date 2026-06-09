"""
CLI to build the FAISS index from a docs directory.

Usage:
    python serving/build_index.py --docs_dir data/docs --index_path data/faiss_index
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from serving.rag_pipeline import IndexBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir",   default="data/docs")
    parser.add_argument("--index_path", default="data/faiss_index")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--overlap",    type=int, default=50)
    args = parser.parse_args()

    builder = IndexBuilder(chunk_size=args.chunk_size, overlap=args.overlap)
    builder.build(args.docs_dir, args.index_path)
