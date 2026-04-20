"""YAML-backed hierarchical config with dot access and CLI overrides.

Example
-------
>>> cfg = Config()
>>> cfg.load("configs/pretrain/task_generalization.yaml")
>>> cfg.update(["--train.num_epochs=20"])  # CLI override
>>> cfg.train.num_epochs
20

Config files may include a shared base via a top-level ``_base`` key
(string path or list of paths, each resolved relative to the including
file). The base is loaded first; the child's keys then deep-merge on
top, with later entries in a ``_base`` list overriding earlier ones.
"""

from __future__ import annotations

import builtins
import os
from ast import literal_eval
from typing import Any

import yaml

_BASE_KEY = "_base"


class Config(dict):
    """A dict that also supports attribute access and dotted lookup."""

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        d: Any = self
        for part in key.split("."):
            if not isinstance(d, dict) or part not in d:
                raise AttributeError(key)
            d = d[part]
        return d

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def load(self, fpath: str) -> None:
        data = _load_yaml_with_base(fpath)
        self.update(data)

    def update(self, other: builtins.dict | list | tuple) -> None:  # type: ignore[override]
        if isinstance(other, dict):
            for k, v in other.items():
                if isinstance(v, dict):
                    if not isinstance(self.get(k), Config):
                        self[k] = Config()
                    self[k].update(v)
                else:
                    self[k] = v
            return

        # Treat sequences as CLI-style ``--a.b=c`` overrides.
        opts = list(other)
        i = 0
        while i < len(opts):
            opt = opts[i]
            if opt.startswith("--"):
                opt = opt[2:]
            if "=" in opt:
                key, value = opt.split("=", 1)
                i += 1
            else:
                key, value = opt, opts[i + 1]
                i += 2
            try:
                value = literal_eval(value)
            except Exception:
                pass
            cur: Any = self
            parts = key.split(".")
            for sub in parts[:-1]:
                cur = cur.setdefault(sub, Config())
            cur[parts[-1]] = value

    def dict(self) -> builtins.dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.items():
            out[k] = v.dict() if isinstance(v, Config) else v
        return out

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.dict(), f)


configs = Config()


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins on conflict)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_with_base(
    fpath: str, _seen: tuple[str, ...] = ()
) -> dict:
    """Load a YAML file, resolving ``_base`` includes relative to its directory."""
    fpath = os.path.abspath(fpath)
    if fpath in _seen:
        cycle = " → ".join((*_seen, fpath))
        raise ValueError(f"Cyclic _base include detected: {cycle}")
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    with open(fpath) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return data  # Non-dict root; nothing to merge against.

    base_spec = data.pop(_BASE_KEY, None)
    if base_spec is None:
        return data

    base_paths = [base_spec] if isinstance(base_spec, str) else list(base_spec)
    here = os.path.dirname(fpath)
    merged: dict = {}
    for rel in base_paths:
        resolved = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(here, rel))
        merged = _deep_merge(
            merged, _load_yaml_with_base(resolved, _seen + (fpath,))
        )
    return _deep_merge(merged, data)


def load_config(path: str, overrides: list[str] | tuple[str, ...] | None = None) -> Config:
    """Convenience one-shot loader."""
    cfg = Config()
    cfg.load(path)
    if overrides:
        cfg.update(list(overrides))
    return cfg


def _split_line(line: str) -> list[str]:
    parts = [p.strip() for p in line.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("Empty dataset line.")
    return parts


def parse_train_spec(spec: str) -> list[str]:
    """Parse ``cfg.dataset.train`` into a flat list of file paths.

    The training spec is a single line; commas separate files that are
    concatenated into one training set.
    """
    if not isinstance(spec, str):
        raise TypeError(
            f"dataset.train must be a string with comma-separated paths; got {type(spec).__name__}"
        )
    return _split_line(spec)


def parse_test_spec(spec: list[str]) -> list[list[str]]:
    """Parse ``cfg.dataset.test`` into one file-list per evaluation unit.

    Each element is one line; commas within a line concatenate files
    into a single evaluation unit.
    """
    if not isinstance(spec, list) or not spec:
        raise TypeError(
            "dataset.test must be a non-empty list of comma-separated path strings."
        )
    return [_split_line(line) for line in spec]
