#!/usr/bin/env python3
"""Validate the portable seq-cifar configuration matrix."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "configs/main/teformer_scifar.yml": "TIMv2_e3_seq",
    "configs/baselines/qkformer_scifar.yml": "TIMv2_bl_qk_seq",
    "configs/baselines/sdt_scifar.yml": "TIMv2_bl_sdt_seq",
    "configs/baselines/spikformer_scifar.yml": "TIMv2_bl_spikf_seq",
    "configs/ablations/binary_ssa_scifar.yml": "TIMv2_e3_seq_binary",
    "configs/ablations/ternary_ssa_scifar.yml": "TIMv2_e3_seq_ternary",
    "configs/ablations/ternary_shiftmax_scifar.yml": "TIMv2_e3_seq_ternary_shiftmax",
    "configs/smoke/teformer_scifar_smoke.yml": "TIMv2_e3_seq",
}
REQUIRED_KEYS = {
    "data_dir", "dataset", "num_classes", "img_size", "mean", "std",
    "scale", "ratio", "epochs", "model", "step", "patch_size",
    "in_channels", "embed_dim", "num_heads", "depths", "batch_size",
    "val_batch_size", "lr", "workers", "output", "device",
}
LIST_LENGTHS = {"mean": 3, "std": 3, "scale": 2, "ratio": 2}
FORBIDDEN_ABSOLUTE_PATH = re.compile(r"(?<![.\w-])/(?:data|mnt)/")


def load_and_validate(relative_path: str) -> list[str]:
    path = ROOT / relative_path
    errors: list[str] = []
    if not path.is_file():
        return [f"{relative_path}: file is missing"]

    text = path.read_text(encoding="utf-8")
    match = FORBIDDEN_ABSOLUTE_PATH.search(text)
    if match:
        errors.append(
            f"{relative_path}: contains forbidden machine path prefix {match.group()!r}"
        )

    try:
        config: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return errors + [f"{relative_path}: invalid YAML: {exc}"]
    if not isinstance(config, dict):
        return errors + [f"{relative_path}: YAML root must be a mapping"]

    missing = sorted(REQUIRED_KEYS - config.keys())
    if missing:
        errors.append(f"{relative_path}: missing keys: {', '.join(missing)}")

    expected_scalars = {
        "dataset": "scifar",
        "num_classes": 10,
        "img_size": 32,
        "step": 4,
        "model": EXPECTED_MODELS[relative_path],
        "device": 0,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            errors.append(
                f"{relative_path}: {key} must be {expected!r}, got {config.get(key)!r}"
            )

    expected_epochs = 1 if "/smoke/" in relative_path else 400
    if config.get("epochs") != expected_epochs:
        errors.append(
            f"{relative_path}: epochs must be {expected_epochs}, got {config.get('epochs')!r}"
        )

    for key, expected_length in LIST_LENGTHS.items():
        value = config.get(key)
        if not isinstance(value, list) or len(value) != expected_length:
            errors.append(
                f"{relative_path}: {key} must be a list of length {expected_length}"
            )

    if "/smoke/" in relative_path:
        smoke_values = {"batch_size": 8, "val_batch_size": 8, "workers": 2, "log_interval": 1, "warmup_epochs": 0, "cooldown_epochs": 0}
        for key, expected in smoke_values.items():
            if config.get(key) != expected:
                errors.append(f"{relative_path}: {key} must be {expected}")

    for key in ("data_dir", "output"):
        value = config.get(key)
        if not isinstance(value, str) or not value.startswith("./"):
            errors.append(f"{relative_path}: {key} must be a ./ relative path")
    return errors


def validate_all() -> list[str]:
    errors: list[str] = []
    actual = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "configs").rglob("*.yml")
    }
    expected = set(EXPECTED_MODELS)
    for extra in sorted(actual - expected):
        errors.append(f"unexpected YAML config: {extra}")
    for relative_path in EXPECTED_MODELS:
        errors.extend(load_and_validate(relative_path))
    return errors


def main() -> int:
    errors = validate_all()
    if errors:
        print("Configuration validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Validated {len(EXPECTED_MODELS)} seq-cifar configuration files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
