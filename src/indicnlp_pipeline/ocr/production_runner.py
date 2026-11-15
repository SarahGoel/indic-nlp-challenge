"""Resumable, checkpointed production helper for full-volume OCR.

This module holds the reusable, testable logic used by
``scripts/11_run_full_ocr.py``:

* deterministic preprocessing-variant application,
* atomic JSON writes,
* an append-safe status manifest with resume support,
* a configuration fingerprint that prevents stale reuse.

Raw scans are read-only inputs. No function here writes inside
``data/raw_scans/``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from indicnlp_pipeline.preprocessing.image_prep import (
    denoise_gaussian,
    deskew,
    enhance_contrast,
    grayscale,
    threshold_adaptive,
    threshold_otsu,
)

SUPPORTED_VARIANTS = (
    "grayscale",
    "denoised",
    "contrast",
    "otsu",
    "adaptive_threshold",
    "deskewed",
)

#: Statuses that represent a completed unit of work and may be reused on resume.
COMPLETED_STATUSES = frozenset({"success", "empty_output"})


class ProductionRunnerError(RuntimeError):
    """Raised when the production OCR runner cannot safely continue."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Frozen preprocessing parameters for one production variant."""

    variant: str
    adaptive_method: str = "gaussian"
    adaptive_block_size: int = 41
    adaptive_constant: float = 15.0
    denoise_kernel_size: int = 5
    denoise_sigma: float = 0.0
    contrast_clip_limit: float = 2.0
    contrast_tile_grid_size: int = 8

    def __post_init__(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported preprocessing variant: {self.variant!r}. "
                f"Supported: {', '.join(SUPPORTED_VARIANTS)}."
            )


def apply_variant(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """Apply the frozen preprocessing chain and return the variant image."""
    gray = grayscale(image)

    if config.variant == "grayscale":
        return gray

    denoised = denoise_gaussian(
        gray,
        kernel_size=config.denoise_kernel_size,
        sigma=config.denoise_sigma,
    )

    if config.variant == "denoised":
        return denoised

    contrast = enhance_contrast(
        denoised,
        clip_limit=config.contrast_clip_limit,
        tile_grid_size=config.contrast_tile_grid_size,
    )

    if config.variant == "contrast":
        return contrast
    if config.variant == "otsu":
        return threshold_otsu(contrast)
    if config.variant == "adaptive_threshold":
        return threshold_adaptive(
            contrast,
            block_size=config.adaptive_block_size,
            constant=config.adaptive_constant,
            method=config.adaptive_method,
        )
    if config.variant == "deskewed":
        return deskew(contrast)

    raise ProductionRunnerError(  # pragma: no cover - guarded by __post_init__
        f"Unhandled preprocessing variant: {config.variant!r}."
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON document atomically via a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def build_config_fingerprint(
    *,
    preprocessing: PreprocessingConfig,
    engines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic description of the run configuration."""
    return {
        "preprocessing": {
            "variant": preprocessing.variant,
            "adaptive_method": preprocessing.adaptive_method,
            "adaptive_block_size": preprocessing.adaptive_block_size,
            "adaptive_constant": preprocessing.adaptive_constant,
            "denoise_kernel_size": preprocessing.denoise_kernel_size,
            "denoise_sigma": preprocessing.denoise_sigma,
            "contrast_clip_limit": preprocessing.contrast_clip_limit,
            "contrast_tile_grid_size": preprocessing.contrast_tile_grid_size,
        },
        "engines": {
            name: dict(sorted(params.items()))
            for name, params in sorted(engines.items())
        },
    }


def fingerprint_digest(fingerprint: dict[str, Any]) -> str:
    """Return a stable SHA-256 digest for a configuration fingerprint."""
    encoded = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_output_path(
    output_root: Path,
    page_id: str,
    engine: str,
    variant: str,
) -> Path:
    """Return the deterministic JSON path for one page/engine/variant."""
    return output_root / engine / f"{page_id}__{variant}.json"


def load_status_manifest(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load a status manifest keyed by (page_id, engine, variant)."""
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}

    if not path.is_file():
        return entries

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProductionRunnerError(
                    f"Invalid JSON in status manifest {path} line {line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise ProductionRunnerError(
                    f"Non-object record in status manifest {path} line {line_number}."
                )
            key = (
                str(record.get("page_id")),
                str(record.get("engine")),
                str(record.get("variant")),
            )
            entries[key] = record

    return entries


def is_reusable(entry: dict[str, Any] | None, output_path: Path) -> bool:
    """Decide whether a prior status entry may be reused unchanged.

    A completed entry is reused only when its recorded output still exists and
    its recorded hash still matches. This prevents blindly reusing stale or
    corrupted results.
    """
    if not entry or entry.get("status") not in COMPLETED_STATUSES:
        return False

    recorded_hash = entry.get("output_sha256")
    if not isinstance(recorded_hash, str) or not output_path.is_file():
        return False

    return sha256_file(output_path) == recorded_hash


def summarize_status(
    entries: dict[tuple[str, str, str], dict[str, Any]],
    engines: Iterable[str],
) -> dict[str, Any]:
    """Summarize manifest statuses by engine."""
    per_engine: dict[str, dict[str, int]] = {engine: {} for engine in engines}
    total = 0

    for (page_id, engine, variant), record in entries.items():
        del page_id, variant
        if engine not in per_engine:
            per_engine[engine] = {}
        status = str(record.get("status", "unknown"))
        per_engine[engine][status] = per_engine[engine].get(status, 0) + 1
        total += 1

    return {"total_records": total, "per_engine": per_engine}
