"""Stage-1 inference: batch-generate completions and stream them to JSONL.

The output is a raw predictions file — one JSON object per example with
``prompt``, ``predict``, and ``label`` fields. Metrics are computed
offline from this file by :mod:`src.evaluation.evaluator`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import torch
from datasets import load_dataset
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["Inference"]


class Inference:
    """Batch-generates with a causal LM and writes predictions JSONL."""

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device | str,
        batch_size: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def _pos_cap(self) -> int:
        cfg = self.model.config
        n = getattr(cfg, "n_positions", None) or getattr(
            cfg, "max_position_embeddings", None
        )
        if n is None:
            raise ValueError(
                "Model config exposes neither n_positions nor max_position_embeddings."
            )
        return int(n)

    def run(
        self,
        eval_paths: Sequence[str],
        predictions_path: str,
        test_ratio: float = 1.0,
    ) -> None:
        """Generate predictions over the concatenation of ``eval_paths``."""
        if not 0.0 < test_ratio <= 1.0:
            raise ValueError(f"test_ratio must be in (0, 1]; got {test_ratio}")
        dataset = load_dataset("json", data_files={"test": list(eval_paths)})["test"]
        if test_ratio < 1.0:
            dataset = dataset.shuffle(seed=42).select(
                range(int(len(dataset) * test_ratio))
            )
        prompts: list[str] = list(dataset["input"])
        labels: list[str] = list(dataset["output"])

        cap = self._pos_cap()
        was_training = self.model.training
        self.model.eval()
        total = len(prompts)

        os.makedirs(os.path.dirname(predictions_path) or ".", exist_ok=True)
        try:
            with open(predictions_path, "w") as writer:
                for start in tqdm(range(0, total, self.batch_size), desc="Inference"):
                    end = min(start + self.batch_size, total)
                    batch_prompts = prompts[start:end]
                    batch_labels = labels[start:end]

                    inputs = self.tokenizer(
                        list(batch_prompts),
                        return_tensors="pt",
                        max_length=cap,
                        padding=True,
                        truncation=True,
                        padding_side="left",
                    ).to(self.device)

                    gen_kwargs = dict(
                        max_length=cap,
                        num_return_sequences=1,
                        use_cache=True,
                        do_sample=self.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    if self.do_sample:
                        gen_kwargs["temperature"] = self.temperature
                        if self.top_k is not None:
                            gen_kwargs["top_k"] = self.top_k
                        if self.top_p is not None:
                            gen_kwargs["top_p"] = self.top_p
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, **gen_kwargs)

                    full_text = self.tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                    predicts = [
                        t[len(p):].strip()
                        for t, p in zip(full_text, batch_prompts, strict=True)
                    ]

                    for prompt, predict, label in zip(
                        batch_prompts, predicts, batch_labels, strict=True
                    ):
                        writer.write(
                            json.dumps(
                                {"prompt": prompt, "predict": predict, "label": label}
                            )
                            + "\n"
                        )
        finally:
            self.model.train(was_training)
