
"""Command-line entry point for Phase 5 multilingual translation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from indicnlp_pipeline.translation.translation_runner import (
    run_translations,
)


def _csv_values(value: str) -> list[str]:
    """Parse a comma-separated CLI argument."""
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Create the translation command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Translate verified Indic NLP ground-truth segments."
    )
    parser.add_argument(
        "--source-languages",
        type=_csv_values,
        default=["hin"],
        help="Comma-separated source codes (default: hin).",
    )
    parser.add_argument(
        "--target-languages",
        type=_csv_values,
        default=[
            "hin", "ben", "guj", "mar", "tam", "tel",
            "kan", "mal", "pan", "asm", "ory", "urd",
        ],
        help="Comma-separated target language codes.",
    )
    parser.add_argument(
        "--engines",
        type=_csv_values,
        default=["indictrans2"],
        help="Comma-separated engines: indictrans2,gemini (default: indictrans2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Verified segments per source language (default: 2).",
    )
    parser.add_argument(
        "--all-segments",
        action="store_true",
        help="Process every eligible verified segment.",
    )
    parser.add_argument(
        "--gemini-model",
        help="Explicit Gemini model identifier; required for Gemini runs.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="IndicTrans2 inference device; defaults to automatic selection.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Prohibit downloading missing IndicTrans2 model files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing per-pair output files.",
    )
    return parser


def main() -> int:
    """Validate arguments, execute translation jobs, and report failures."""
    parser = build_parser()
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.all_segments and "--limit" in sys.argv:
        parser.error("Use either --all-segments or --limit, not both.")

    root = Path(__file__).resolve().parents[1]

    try:
        outputs = run_translations(
            root=root,
            source_languages=args.source_languages,
            target_languages=args.target_languages,
            engines=args.engines,
            limit=None if args.all_segments else args.limit,
            gemini_model=args.gemini_model,
            device=args.device,
            local_files_only=args.local_files_only,
            overwrite=args.overwrite,
        )

        logger.info("Translation runner finished. Output files: {}", len(outputs))
        for output in outputs:
            logger.info("Output: {}", output.relative_to(root))
        return 0

    except Exception as exc:
        logger.exception("Translation runner failed: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
