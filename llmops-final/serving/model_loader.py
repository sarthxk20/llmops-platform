"""
Model loading utilities — handles base model + LoRA adapter loading,
generation, and embedding extraction.
"""

import logging
from typing import List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)


class ModelLoader:
    """Thread-safe wrapper around model + tokenizer for inference."""

    def __init__(self, base_model: str, adapter_path: Optional[str] = None):
        self.base_model   = base_model
        self.adapter_path = adapter_path
        self.model        = None
        self.tokenizer    = None
        self.is_loaded    = False
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name   = f"{base_model.split('/')[-1]}"
        if adapter_path:
            self.model_name += "+lora"

    def load(self) -> None:
        """Load tokenizer and model (with optional LoRA adapter)."""
        log.info(f"Loading tokenizer from: {self.adapter_path or self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path or self.base_model,
            use_fast=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info(f"Loading base model: {self.base_model} on {self.device}")

        # Use 4-bit quantisation if GPU available; otherwise full precision CPU
        if self.device.type == "cuda":
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        # Attach LoRA adapter if provided
        if self.adapter_path:
            log.info(f"Loading LoRA adapter from: {self.adapter_path}")
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()   # merge for faster inference
            log.info("LoRA adapter merged.")

        model.eval()
        self.model     = model
        self.is_loaded = True
        log.info("Model loaded and ready.")

    def unload(self) -> None:
        del self.model
        self.model     = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 256,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
        top_k:          int   = 50,
        do_sample:      bool  = True,
    ) -> Tuple[str, int]:
        """Run generation and return (decoded_text, n_tokens_generated)."""
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)
        n_input   = input_ids.shape[1]

        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_ids   = out[0][n_input:]
        n_tokens  = len(new_ids)
        decoded   = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return decoded.strip(), n_tokens

    def embed(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """Extract mean-pooled last-hidden-state embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            hidden    = out.hidden_states[-1]           # (B, T, D)
            mask      = enc["attention_mask"].unsqueeze(-1).float()
            pooled    = (hidden * mask).sum(1) / mask.sum(1)  # mean pool
            all_embeddings.extend(pooled.cpu().float().tolist())
        return all_embeddings
