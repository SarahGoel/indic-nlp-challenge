"""Command-line entry point for the Indic NLP pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_preprocess_scans.py",
    "02_run_ocr.py",
    "03_normalize_and_segment.py",
    "04_translate.py",
    "05_evaluate.py",
    "06_generate_error_logs.py",
    "07_generate_visualizations.py",
]


def main() -> int:
    """Run pipeline stages sequentially, stopping on the first failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-stage",
        choices=SCRIPTS,
        default=SCRIPTS[0],
        help="Stage script to start from.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    start = SCRIPTS.index(args.from_stage)

    for script in SCRIPTS[start:]:
        path = root / "scripts" / script
        if not path.is_file():
            parser.error(f"Required script not found: {path}")
        result = subprocess.run([sys.executable, str(path)], cwd=root)
        if result.returncode:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
