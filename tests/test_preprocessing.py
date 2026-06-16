"""
Unit tests for data preprocessing pipeline.
Run: pytest tests/test_preprocessing.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocess import format_example, tokenize_fn, validate_dataset, save_stats


class TestFormatExample:
    def test_standard_instruction_format(self):
        example = {
            "instruction": "Summarise this.",
            "input": "Long text here.",
            "output": "Summary.",
        }
        result = format_example(example)
        assert "### Instruction:" in result["text"]
        assert "### Input:" in result["text"]
        assert "### Response:" in result["text"]
        assert "Summarise this." in result["text"]
        assert "Summary." in result["text"]

    def test_bitext_format_mapping(self):
        example = {
            "utterance": "Where is my order?",
            "response": "Let me check your order status.",
        }
        result = format_example(example)
        assert "Where is my order?" in result["text"]
        assert "Let me check" in result["text"]
        assert "### Response:" in result["text"]

    def test_missing_optional_fields(self):
        example = {"instruction": "Tell me a joke.", "output": "Why did the..."}
        result = format_example(example)
        assert "Tell me a joke." in result["text"]

    def test_returns_text_key(self):
        example = {"instruction": "Hi", "input": "", "output": "Hello!"}
        result = format_example(example)
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0


class TestTokenizeFn:
    def test_labels_mask_padding(self):
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.return_value = {
            "input_ids":      [[1, 2, 3, 0, 0]],
            "attention_mask": [[1, 1, 1, 0, 0]],
        }
        result = tokenize_fn({"text": ["Hello world test pad pad"]}, tok, max_length=5)
        assert result["labels"][0][3] == -100
        assert result["labels"][0][4] == -100
        assert result["labels"][0][0] == 1
        assert result["labels"][0][1] == 2

    def test_output_has_labels_key(self):
        tok = MagicMock()
        tok.pad_token_id = 0
        tok.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1,1,1]]}
        result = tokenize_fn({"text": ["abc"]}, tok, max_length=8)
        assert "labels" in result


class TestValidateDataset:
    def _make_dataset(self, n=5):
        from datasets import Dataset
        data = {
            "input_ids":      [[1, 2, 3, 0] for _ in range(n)],
            "attention_mask": [[1, 1, 1, 0] for _ in range(n)],
            "labels":         [[1, 2, 3, -100] for _ in range(n)],
        }
        return {"train": Dataset.from_dict(data), "validation": Dataset.from_dict(data)}

    def test_passes_valid_dataset(self):
        ds = self._make_dataset()
        validate_dataset(ds)

    def test_fails_missing_input_ids(self):
        from datasets import Dataset
        bad = {"train": Dataset.from_dict({"labels": [[1,2,3]], "attention_mask": [[1,1,1]]})}
        with pytest.raises((AssertionError, KeyError)):
            validate_dataset(bad)

    def test_fails_empty_split(self):
        from datasets import Dataset
        bad = {"train": Dataset.from_dict({"input_ids": [], "labels": [], "attention_mask": []})}
        with pytest.raises(AssertionError):
            validate_dataset(bad)


class TestSaveStats:
    def test_creates_json_file(self, tmp_path):
        from datasets import Dataset
        tok = MagicMock()
        tok.pad_token_id = 0
        ds = {"train": Dataset.from_dict({
            "input_ids":      [[1, 2, 3, 0], [1, 0, 0, 0]],
            "attention_mask": [[1,1,1,0], [1,0,0,0]],
            "labels":         [[1,2,3,-100], [1,-100,-100,-100]],
        })}
        save_stats(ds, tmp_path, tok)
        stats_file = tmp_path / "dataset_stats.json"
        assert stats_file.exists()
        with open(stats_file) as f:
            stats = json.load(f)
        assert "train" in stats
        assert "n_examples" in stats["train"]
        assert stats["train"]["n_examples"] == 2

    def test_correct_token_counts(self, tmp_path):
        from datasets import Dataset
        tok = MagicMock()
        tok.pad_token_id = 0
        ds = {"train": Dataset.from_dict({
            "input_ids":      [[1,2,3,0], [1,2,0,0]],
            "attention_mask": [[1,1,1,0], [1,1,0,0]],
            "labels":         [[1,2,3,-100], [1,2,-100,-100]],
        })}
        stats = save_stats(ds, tmp_path, tok)
        assert stats["train"]["mean_tokens"] == 2.5
        assert stats["train"]["max_tokens"]  == 3
        assert stats["train"]["min_tokens"]  == 2
