"""Config loading, seeding, logging and run manifests.

Every numeric constant in this project lives in ``configs/base.yaml``. Modules
read them through :func:`cfg_get` rather than hard-coding values, so an
experiment is fully described by a config hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG = "configs/base.yaml"

# Resolved once so paths work no matter which directory the CLI is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load a YAML config, resolving ``path`` against the repo root if relative."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config at {p} did not parse to a mapping")
    cfg["_config_path"] = str(p)
    return cfg


_MISSING = object()


def cfg_get(cfg: dict, dotted_key: str, default: Any = _MISSING) -> Any:
    """Look up a nested key, e.g. ``cfg_get(cfg, "geometry.floor.min_inlier_ratio")``.

    Raises KeyError when the key is absent and no default was given -- a missing
    constant should fail loudly rather than silently fall back to a magic number.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(f"missing config key: {dotted_key}")
            return default
        node = node[part]
    return node


def resolve_path(cfg: dict, dotted_key: str) -> Path:
    """Resolve a ``paths.*`` config entry to an absolute path under the repo root."""
    value = Path(cfg_get(cfg, dotted_key))
    return value if value.is_absolute() else REPO_ROOT / value


def config_hash(cfg: dict, length: int = 12) -> str:
    """Stable short hash of the config, ignoring bookkeeping keys."""
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:length]


def seed_everything(seed: int) -> None:
    """Seed every source of randomness we touch (RANSAC, sklearn, augmentation)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def git_sha(short: bool = True) -> str:
    """Current commit SHA, or ``"unknown"`` outside a repo / before the first commit."""
    args = ["git", "rev-parse", "--short" if short else "HEAD"]
    if short:
        args.append("HEAD")
    try:
        out = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def get_logger(stage: str, cfg: dict) -> logging.Logger:
    """Logger writing to both stderr and ``logs/<stage>.log``."""
    log_dir = resolve_path(cfg, "paths.logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"deadbug.{stage}")
    logger.setLevel(cfg_get(cfg, "logging.level", "INFO"))
    if logger.handlers:  # already configured in this process
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    fh = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def write_manifest(out_dir: str | Path, cfg: dict, stage: str, **extra: Any) -> Path:
    """Record what produced the contents of ``out_dir``.

    Each pipeline stage writes one of these so a later run can tell whether its
    inputs were built with the same config and code.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / cfg_get(cfg, "run.manifest_name", "_manifest.json")
    payload = {
        "stage": stage,
        "config_hash": config_hash(cfg),
        "config_path": cfg.get("_config_path"),
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def read_manifest(out_dir: str | Path, cfg: dict) -> dict | None:
    """Read a stage manifest, or None when the stage has not run."""
    path = Path(out_dir) / cfg_get(cfg, "run.manifest_name", "_manifest.json")
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
