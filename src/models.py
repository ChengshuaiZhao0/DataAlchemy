"""Model and tokenizer factories for DataAlchemy pretraining.

Two architectures are supported for the from-scratch experiments:

    GPT-2 : ``transformers.GPT2Config`` / ``GPT2LMHeadModel``.
    LLaMA : ``transformers.LlamaConfig`` / ``LlamaForCausalLM``.

Both use the same symbolic vocabulary trained with ByteLevel-BPE (GPT-2) or
SentencePiece (LLaMA) over the pretraining corpus.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from src.constants import (
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

Arch = Literal["GPT", "Llama"]
AttnImpl = Literal["sdpa", "eager", "flash_attention_2"]

ARCH_GPT: Arch = "GPT"
ARCH_LLAMA: Arch = "Llama"

# LLaMA's SwiGLU FFN inner dim is conventionally 8/3 * hidden_size
# (so that 3 * intermediate_size matrix-multiplies cost ~8 * hidden_size^2).
LLAMA_DEFAULT_FFN_RATIO = 8 / 3

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN]


def _check_arch(arch: str) -> Arch:
    if arch not in (ARCH_GPT, ARCH_LLAMA):
        raise ValueError(
            f"Unknown architecture: {arch!r} (expected {ARCH_GPT!r} or {ARCH_LLAMA!r})"
        )
    return arch  # type: ignore[return-value]


@dataclass
class ModelSpec:
    """Hyper-params for a from-scratch causal LM."""

    arch: Arch
    hidden_size: int
    intermediate_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_positions: int = 256
    rope_theta: float = 10000.0
    attn_implementation: AttnImpl = "sdpa"


def build_tokenizer(
    tokenizer_dir: str,
    arch: Arch,
    corpus_files: Iterable[str] | None = None,
    vocab_size: int = 64,
    n_positions: int = 256,
):
    """Train a fresh tokenizer from ``corpus_files``, or reload from disk.

    - If ``corpus_files`` is provided, always train fresh (overwriting any
      files already at ``tokenizer_dir``). This avoids silently reusing a
      stale tokenizer when the corpus or vocab_size changes.
    - If ``corpus_files`` is None, reload from disk; raise if absent.
    - ``n_positions`` bounds the SentencePiece ``max_sentence_length`` so the
      tokenizer training cap tracks the model context length.
    """
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast, LlamaTokenizerFast

    arch = _check_arch(arch)

    if corpus_files is None:
        if arch == ARCH_GPT and os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
            return GPT2TokenizerFast.from_pretrained(
                tokenizer_dir,
                pad_token=PAD_TOKEN,
                bos_token=BOS_TOKEN,
                eos_token=EOS_TOKEN,
                unk_token=UNK_TOKEN,
                mask_token=MASK_TOKEN,
            )
        if arch == ARCH_LLAMA and os.path.exists(
            os.path.join(tokenizer_dir, "tokenizer.model")
        ):
            tok = LlamaTokenizerFast(
                vocab_file=os.path.join(tokenizer_dir, "tokenizer.model")
            )
            tok.add_special_tokens(
                {
                    "pad_token": PAD_TOKEN,
                    "bos_token": BOS_TOKEN,
                    "eos_token": EOS_TOKEN,
                    "unk_token": UNK_TOKEN,
                    "mask_token": MASK_TOKEN,
                }
            )
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
        raise FileNotFoundError(
            f"Tokenizer not found under {tokenizer_dir!r} and no corpus supplied to train one."
        )

    logger.info("Training tokenizer (arch=%s, vocab_size=%d) → %s", arch, vocab_size, tokenizer_dir)
    os.makedirs(tokenizer_dir, exist_ok=True)
    from datasets import load_dataset

    # Corpus files are JSONL — render `input + " " + output` per record so the
    # BPE / SentencePiece trainer sees the actual text the LM will encounter.
    train_set = load_dataset(
        "json", data_files={"train": list(corpus_files)}
    )["train"]

    def _iter_corpus_text():
        for record in train_set:
            yield f"{record['input']} {record['output']}"

    if arch == ARCH_GPT:
        raw = ByteLevelBPETokenizer()
        raw.train_from_iterator(
            _iter_corpus_text(),
            vocab_size=vocab_size,
            min_frequency=1,
            special_tokens=SPECIAL_TOKENS,
        )
        raw.save_model(tokenizer_dir)
        return GPT2TokenizerFast.from_pretrained(
            tokenizer_dir,
            pad_token=PAD_TOKEN,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            unk_token=UNK_TOKEN,
            mask_token=MASK_TOKEN,
        )

    if arch == ARCH_LLAMA:
        import sentencepiece as spm

        corpus_path = os.path.join(tokenizer_dir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as fout:
            for line in _iter_corpus_text():
                fout.write(line + "\n")

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=os.path.join(tokenizer_dir, "tokenizer"),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_piece=UNK_TOKEN,
            bos_piece=BOS_TOKEN,
            eos_piece=EOS_TOKEN,
            pad_piece=PAD_TOKEN,
            user_defined_symbols=[MASK_TOKEN],
            max_sentence_length=n_positions,
            input_sentence_size=0,
            shuffle_input_sentence=True,
            hard_vocab_limit=False,
        )

        tok = LlamaTokenizerFast(
            vocab_file=os.path.join(tokenizer_dir, "tokenizer.model")
        )
        tok.add_special_tokens(
            {
                "pad_token": PAD_TOKEN,
                "bos_token": BOS_TOKEN,
                "eos_token": EOS_TOKEN,
                "unk_token": UNK_TOKEN,
                "mask_token": MASK_TOKEN,
            }
        )
        tok.save_pretrained(tokenizer_dir)
        return tok

    raise ValueError(f"Unknown architecture: {arch!r}")


def build_model(spec: ModelSpec, tokenizer):
    """Construct a from-scratch causal LM matching ``spec``."""
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        LlamaConfig,
        LlamaForCausalLM,
    )

    arch = _check_arch(spec.arch)
    vocab_size = tokenizer.vocab_size
    pad_id = getattr(tokenizer, "pad_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    if arch == ARCH_GPT:
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=spec.n_positions,
            n_embd=spec.hidden_size,
            n_layer=spec.n_layer,
            n_head=spec.n_head,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            attn_implementation=spec.attn_implementation,
        )
        return GPT2LMHeadModel(config)

    config = LlamaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=spec.n_positions,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.n_layer,
        num_attention_heads=spec.n_head,
        hidden_act="silu",
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        rope_theta=spec.rope_theta,
        attn_implementation=spec.attn_implementation,
    )
    return LlamaForCausalLM(config)


def load_tokenizer_and_model(
    save_dir: str,
    arch: Arch,
    attn_implementation: AttnImpl = "sdpa",
):
    """Load the tokenizer and final model from a checkpoint saved by ``Trainer``.

    The model is loaded in bf16 when CUDA is available (matching the
    training dtype) and in fp32 otherwise. ``attn_implementation`` is
    passed through to ``from_pretrained`` so SDPA / FlashAttention kernels
    are used when supported.
    """
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        LlamaForCausalLM,
        LlamaTokenizerFast,
    )

    arch = _check_arch(arch)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if arch == ARCH_GPT:
        tokenizer = GPT2Tokenizer.from_pretrained(save_dir)
        model = GPT2LMHeadModel.from_pretrained(
            save_dir,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
    else:
        tokenizer = LlamaTokenizerFast.from_pretrained(save_dir)
        model = LlamaForCausalLM.from_pretrained(
            save_dir,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def num_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
