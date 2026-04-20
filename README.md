# Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens

[![Paper](https://img.shields.io/badge/Paper-arXiv:2508.01191-%23f2806bff.svg)](http://arxiv.org/abs/2508.01191) [![Code](https://img.shields.io/badge/Code-GitHub-%238a91faff.svg)](http://arxiv.org/abs/2508.01191) [![Daily](https://img.shields.io/badge/Daily_Paper-Hugging_Face-%2340D9B0.svg)](https://huggingface.co/papers/2508.01191)

This repository contains the official Python implementation of the framework described in the paper **"Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens."**

## News

- **[09/01/2025]** Our paper has been covered by **The New Yorker**, **SF Examiner**, **VentureBeat**, **Ars Technica**, **The Decoder**, **Slashdot**, **Unite.AI**, **WebProNews**, **Digital Watch Observatory**, and many other media outlets, blogs, newsletters, podcasts, and social platforms — see the full [media coverage list](docs/MEDIA_COVERAGE.md).
- **[08/07/2025]** Our paper achieves **#1 Paper of the day** at [Hugging Face](https://huggingface.co/papers/2508.01191) 🚀
- **[08/02/2025]** Our paper is available on [arXiv](http://arxiv.org/abs/2508.01191).
- **[08/01/2025]** GitHub repository created. Code release is coming soon.

## Introduction

Chain-of-Thought (CoT) prompting has been shown to improve Large Language Model (LLM) performance on various tasks. With this approach, LLMs appear to produce human-like reasoning steps before providing answers (a.k.a., CoT reasoning), which often leads to the perception that they engage in deliberate inferential processes. However, some initial findings suggest that CoT reasoning may be more superficial than it appears, motivating us to explore further. In this paper, we study CoT reasoning via a data distribution lens and investigate if CoT reasoning reflects a structured inductive bias learned from in-distribution data, allowing the model to conditionally generate reasoning paths that approximate those seen during training. Thus, its effectiveness is fundamentally bounded by the degree of distribution discrepancy between the training data and the test queries. With this lens, we dissect CoT reasoning via three dimensions: **task**, **length**, and **format**. To investigate each dimension, we design **DataAlchemy**, an isolated and controlled environment to train LLMs from scratch and systematically probe them under various distribution conditions. Our results reveal that CoT reasoning is a brittle mirage that vanishes when it is pushed beyond training distributions. This work offers a deeper understanding of *why* and *when* CoT reasoning fails, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning.

<p align="center">
  <img src="figure/illustration.png" alt="illustration" width="50%" /><br>
  <em>Figure 1:</em> The data distribution lens.
</p>


## Contribution

⭐ **Novel perspective.** We propose a data distribution lens for CoT reasoning, illuminating that its effectiveness stems from structured inductive biases learned from in-distribution training data. This framework provides a principled lens for understanding *why* and *when* CoT reasoning succeeds or fails.

⭐ **Controlled environment.** We introduce **DataAlchemy** an isolated experimental framework that enables training LLMs from scratch and systematically probing CoT reasoning. This controlled setting allows us to isolate and analyze the effects of distribution shifts on CoT reasoning without interference from complex patterns learned during large-scale pre-training.

⭐ **Empirical validation.** We conduct systematic empirical validation across three critical dimensions—*task*, *length*, and *format*. Our experiments demonstrate that CoT reasoning exhibits sharp performance degradation under distribution shifts, revealing that seemingly coherent reasoning masks shallow pattern replication.

⭐ **Real-world implication.** This work reframes the understanding of contemporary LLMs' reasoning capabilities and emphasizes the risk of over-reliance on CoT reasoning as a universal problem-solving paradigm. It underscores the necessity for proper evaluation methods and the development of LLMs that possess authentic and generalizable reasoning capabilities.

<p align="center">
  <img src="figure/main.png" alt="main" width="100%" /><br>
  <em>Figure 2:</em> DataAlchemy framework.
</p>

## Repository layout

```text
configs/              YAML configs for each experiment family
  pretrain/           task / length / format generalization examples
  sft/                task-level SFT example
data/                 empty by default — populated by scripts/generate_data.py
docs/                 supplementary docs (media coverage, etc.)
experiments/          minimal launchers, one per YAML config (generate → pretrain/sft → infer → evaluate)
figure/               paper figures referenced in this README
scripts/              thin CLI entrypoints
  generate_data.py            build one dataset file from a transformation spec
  apply_noise.py              inject token-level noise into existing CoT files
  pretrain_model.py           train-from-scratch loop
  sft_model.py                supervised fine-tuning loop
  model_inference.py          batch inference on test scenarios (writes predictions.jsonl)
  evaluate_predictions.py     score predictions.jsonl into results.json (no GPU needed)
src/                  core package
  atoms.py                    26-letter alphabet + element/str converters
  constants.py                special tokens (<s>, <pad>, <think>, <answer>, [F1], [F2], [F3])
  transformations.py          f_rot, f_pos, f_reverse, Composition, token↔transform parsers
  models.py                   GPT-2 / LLaMA factories trained from scratch
  data/                       generator.py, splits.py (ID/CMP/POOD/OOD), mixer.py, noise.py
  training/                   pretrain.py, sft.py, args.py (HF TrainingArguments builder)
  inference.py                Inference class (model.generate → predictions.jsonl)
  evaluation/                 metrics.py (EM / edit-distance / BLEU), evaluator.py
  utils/                      config.py, cli.py, logging.py, run_logging.py, seed.py
tests/                pytest suite (fast logic tests + slow training integrations)

# Generated at runtime:
saves/                tokenizer + model checkpoints, one subdir per experiment
logs/                 per-run unified log files (logs/<experiment>/run_<timestamp>.log)
results/              evaluation metrics + predictions (results/<experiment>/<test_tag>/)
```

Naming follows the paper: atoms `A..Z`, element length `l`, transformation tokens `[F1] [F2] [F3]`
for `f_rot`, `f_pos`, `f_reverse`, compositional transformations `f_S`, and
scenarios `ID / CMP / POOD / OOD`.

## Installation

The repo is managed with [uv](https://docs.astral.sh/uv/). A `pip`-only flow is also supported.

### With uv (recommended)

```bash
uv sync                                    # creates .venv, installs runtime deps
uv sync --extra dev                        # add pytest / pytest-cov / ruff
uv run pytest                              # smoke-test the install (fast suite only)
uv run pytest -m slow                      # also run the training integration tests
```

<details>
<summary>With pip (alternative)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

</details>

## Quickstart

The standard DataAlchemy pipeline has four stages that run back-to-back:

1. **Generate data** — symbolic datasets from transformation specs (`scripts/generate_data.py`, optional `scripts/apply_noise.py`).
2. **Train** — pretrain from scratch (`scripts/pretrain_model.py`) or supervised fine-tune (`scripts/sft_model.py`).
3. **Inference** — batch-generate predictions on test scenarios (`scripts/model_inference.py`).
4. **Evaluate** — score predictions into metrics (`scripts/evaluate_predictions.py`).

One end-to-end launcher chains all four stages in a single call:

```bash
bash experiments/task_generalization.sh
```

When it finishes, aggregate metrics are in `results/task_generalization/<test_tag>/results.json`, per-sample predictions in `results/task_generalization/<test_tag>/predictions.jsonl`, and the full run log in `logs/task_generalization/run_<timestamp>.log`.

The sections below walk through each stage so you can run them individually or swap in your own configs. See [Experiments](#experiments) for the full list of pre-wired launchers.

## Stage 1 — Generate data

The `data/` folder is empty by default; all datasets are generated on the fly from symbolic rules — nothing is downloaded.

### Creating a dataset

One JSONL record per `(element, transformation-composition)` pair is written to `--output`.

A minimal two-step composition (`k=2`):

```bash
python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length 4 --output data/F1F2.jsonl
```

The CLI exposes four knobs that map directly onto the paper's axes:

| Flag | Axis | Notes |
| --- | --- | --- |
| `--transformations` | reasoning steps `k` and primitive choice | space-separated tokens; each token adds one step (`[F1]`, `[F2]`, `[F3]`) |
| `--element-length` | text length `l` | atom count per element; default `4` |
| `--rot-n`, `--pos-n` | transformation parameters | default `13` for ROT, `1` for cyclic shift |
| `--no-cot`, `--downsample-to N` | format / size | drop reasoning trace, or subsample to `N` records |

<details>
<summary>More examples (k=1, k=3, length sweep, no-CoT, custom ROT/shift, subsampling)</summary>

```bash
# Single-step primitives (k=1)
python scripts/generate_data.py --transformations "[F1]" --element-length 4 --output data/F1.jsonl
python scripts/generate_data.py --transformations "[F2]" --element-length 4 --output data/F2.jsonl
python scripts/generate_data.py --transformations "[F3]" --element-length 4 --output data/F3.jsonl

# Full task-generalization set (k=2): F1F1, F1F2, F2F1, F2F2
python scripts/generate_data.py --transformations "[F1]" "[F1]" --element-length 4 --output data/F1F1.jsonl
python scripts/generate_data.py --transformations "[F2]" "[F1]" --element-length 4 --output data/F2F1.jsonl
python scripts/generate_data.py --transformations "[F2]" "[F2]" --element-length 4 --output data/F2F2.jsonl

# Deeper reasoning (k=3)
python scripts/generate_data.py --transformations "[F1]" "[F2]" "[F1]" --element-length 4 --output data/F1F2F1.jsonl

# Length sweep — fix the composition, vary l
for L in 2 3 4 5 6; do
    python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length $L --output data/F1F2_len$L.jsonl
done

# No-CoT variant — <answer> as the generation start, no <think> trace
python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length 4 --no-cot --output data/F1F2_nocot.jsonl

# Custom ROT shift and cyclic shift
python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length 4 --rot-n 7 --pos-n 2 --output data/F1F2_rot7_pos2.jsonl

# Subsampling — the full enumeration is 26^l per composition; cap it for quick iteration
python scripts/generate_data.py --transformations "[F1]" "[F2]" --element-length 6 --downsample-to 100000 --output data/F1F2_len6_100k.jsonl
```

</details>

### Injecting noise

```bash
python scripts/apply_noise.py --inputs data/F1F2.jsonl --output data/F1F2_hybrid_0p10.jsonl \
    --mode hybrid --p 0.10 --domain all --seed 42
```

| Flag | Values | Controls |
| --- | --- | --- |
| `--mode` | `insert` \| `delete` \| `modify` \| `hybrid` | noise operation applied to each selected token |
| `--domain` | `all` \| `element` \| `transformation` \| `instruction` | which part of the input the noise targets |
| `--p` | `0.0`–`1.0` | per-token corruption probability |
| `--seed` | int | RNG seed for reproducible noise |

### Dataset schema

Every generated dataset file is organized by JSON Lines. Each record bundles the fully rendered prompt (`input`) and target (`output`) with their constituent pieces (`element`, `transformation`, `instruction`, `reasoning`, `answer`), so downstream code can either train on the rendered strings directly or re-derive them from the components. The fields are:

| Field            | Type   | Meaning                                                                                |
|------------------|--------|----------------------------------------------------------------------------------------|
| `input`          | str    | What the LM conditions on: `element + " " + transformation + " " + instruction`.       |
| `output`         | str    | What the LM should produce: reasoning trace (if any) + `<answer>` + final element.     |
| `element`        | str    | Input element atoms.                                                                   |
| `transformation` | str    | Initial transformation tokens, space-joined.                                           |
| `instruction`    | str    | Output-start marker — `<think>` (CoT) or `<answer>` (no-CoT).                          |
| `reasoning`      | str    | Trace inside `output` before the final `<answer>`. Empty when no CoT.                  |
| `answer`         | str    | Final element after `<answer>`.                                                        |

> Invariants: `input == element + " " + transformation + " " + instruction`; the full rendered line is `input + " " + output`.

## Stage 2 — Train

Training is driven by YAML configs under [configs/pretrain/](configs/pretrain/) and [configs/sft/](configs/sft/). Pretraining builds a model from scratch; SFT fine-tunes an existing checkpoint on a small in-distribution slice. Any config key accepts a dotted CLI override — e.g. `--model.hidden_size=512 --train.lr=1e-3`.

### Pretrain from scratch

```bash
python scripts/pretrain_model.py --config configs/pretrain/task_generalization.yaml
```

Outputs: tokenizer + model checkpoint under `cfg.checkpoint.model_dir` (e.g. `saves/task_generalization/model/final`). Key knobs (defaults from [configs/_base/pretrain.yaml](configs/_base/pretrain.yaml)):

| Config key | Default | Controls |
| --- | --- | --- |
| `model.arch` | `GPT` | architecture — `GPT` or `Llama` |
| `model.hidden_size` / `n_layer` / `n_head` | `32` / `4` / `4` | transformer width and depth |
| `model.n_positions` | `256` | max sequence length |
| `model.attn_implementation` | `sdpa` | `sdpa` \| `eager` \| `flash_attention_2` |
| `train.num_epochs` | `10` | training epochs |
| `train.per_device_batch_size` | `1024` | per-GPU batch size |
| `train.lr` / `lr_scheduler_type` / `warmup_ratio` | `3e-3` / `cosine` / `0.1` | optimizer + schedule |
| `train.bf16` | `true` | bf16 mixed-precision training |
| `train.validation_ratio` | `0` | fraction held out for eval (0 = no eval split) |
| `dataset.train` | — | comma-separated JSONL paths |
| `system.seed` | `42` | RNG seed (torch / numpy / random) |

<details>
<summary>Supervised fine-tuning</summary>

SFT starts from a pretrained checkpoint and fine-tunes on a small in-distribution dataset. Make sure `cfg.checkpoint.pretrained_model_dir` exists before running:

```bash
python scripts/sft_model.py --config configs/sft/task_sft.yaml
```

Additional SFT-specific keys (defaults from [configs/_base/sft.yaml](configs/_base/sft.yaml)):

| Config key | Default | Controls |
| --- | --- | --- |
| `checkpoint.pretrained_model_dir` | — | path to the pretrained checkpoint to fine-tune |
| `checkpoint.tokenizer_dir` | — | tokenizer from the pretraining run (reused, not rebuilt) |
| `dataset.train_ratio` | `1.0` | fraction of `dataset.train` used for fine-tuning (e.g. `0.0001` for 1%) |
| `train.num_epochs` | `20000` | many epochs on a small slice |
| `train.per_device_batch_size` | `256` | smaller than pretrain |
| `train.lr` | `5e-3` | SFT default |

</details>

## Stage 3 — Inference

Load the trained checkpoint and batch-generate predictions on every test tag listed in `cfg.dataset.test`:

```bash
python scripts/model_inference.py --config configs/pretrain/task_generalization.yaml
```

Outputs: one `predictions.jsonl` per test tag at `results/<experiment>/<test_tag>/predictions.jsonl`. Key knobs:

| Config key / flag | Default | Controls |
| --- | --- | --- |
| `inference.per_device_batch_size` | `51200` | generations per forward pass — primary throughput knob |
| `inference.do_sample` | `false` | `true` enables sampling; otherwise greedy decoding |
| `inference.temperature` | `1.0` | sampling temperature (only used when `do_sample: true`) |
| `inference.top_k` | `50` | top-k sampling cutoff |
| `inference.top_p` | `1.0` | nucleus (top-p) sampling cutoff |
| `dataset.test` | — | list of JSONL paths; each entry becomes one `<test_tag>` dir |
| `dataset.test_ratio` | `1.0` | fraction of each test file to evaluate |
| `--checkpoint` (CLI) | `cfg.checkpoint.model_dir` | override which checkpoint to load |

## Stage 4 — Evaluate

Score predictions into metrics. This step is **CPU-only and re-scorable offline** — you can re-run it without touching the model:

```bash
python scripts/evaluate_predictions.py --config configs/pretrain/task_generalization.yaml
```

Outputs: `results/<experiment>/<test_tag>/results.json` with exact-match accuracy, average edit distance, and average BLEU (see [src/evaluation/metrics.py](src/evaluation/metrics.py)). A combined summary is appended to the run log footer.

| Config key | Default | Controls |
| --- | --- | --- |
| `evaluate.split_reasoning` | `true` | when `true`, score reasoning trace and final answer separately |
| `evaluate.num_workers` | `8` | CPU workers for scoring (`0` = `min(8, cpu_count())`; `1` = single-process) |
| `dataset.test` | — | same test spec as Stage 3; each tag expects a matching `predictions.jsonl` |

## Experiments

Each launcher chains the four stages above end-to-end, tees full stdout/stderr into `logs/<experiment>/run_<timestamp>.log`, and prints the result directory at the end.

| Launcher | Description |
| --- | --- |
| [experiments/task_generalization.sh](experiments/task_generalization.sh) | Task generalization: pretrain on compositions `F1∘F1`, `F1∘F2`, `F2∘F1` and evaluate on the held-out composition `F2∘F2`. |
| [experiments/length_generalization.sh](experiments/length_generalization.sh) | Length generalization: pretrain at element length `l=4` and evaluate at unseen lengths `l=2` and `l=3`. |
| [experiments/format_generalization.sh](experiments/format_generalization.sh) | Format generalization: pretrain on clean `F1` / `F2` and evaluate under hybrid token-level noise at `p=0.10` and `p=0.20`. |
| [experiments/task_sft.sh](experiments/task_sft.sh) | SFT recovery: fine-tune the task-gen checkpoint on a small slice of `F2∘F2` to probe how little in-distribution data restores OOD performance. |


## Hardware & Runtime

Stage-by-stage wall-clock on a single **NVIDIA A100-SXM4-40GB**:

| Launcher | Generate | Train | Inference | Evaluate | **Total** |
| --- | ---: | ---: | ---: | ---: | ---: |
| `experiments/task_generalization.sh` | ~1m | ~11m | ~3m | ~10s | **~15m** |
| `experiments/length_generalization.sh` | ~1m | ~5m | ~10s | ~1s | **~6m** |
| `experiments/format_generalization.sh` | ~2m | ~5m | ~50m | ~20s | **~60m** |
| `experiments/task_sft.sh` | ~30s | ~5m | ~3m | ~10s | **~10m** |

## Citation

If our repo helped you out, we'd love it if you gave us a citation! Thanks for supporting our work!

```tex
@article{zhao2025chain,
  title={Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens},
  author={Zhao, Chengshuai and Tan, Zhen and Ma, Pingchuan and Li, Dawei and Jiang, Bohan and Wang, Yancheng and Yang, Yingzhen and Liu, Huan},
  journal={arXiv preprint arXiv:2508.01191},
  year={2025}
}
```
