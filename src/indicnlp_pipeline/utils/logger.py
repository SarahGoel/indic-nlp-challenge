from pathlib import Path
import sys

from loguru import logger


def configure_logger(
    log_directory: Path | None = None,
    log_level: str = "INFO",
) -> None:
    """Configure Loguru for console and file logging."""
    if log_directory is None:
        log_directory = Path("logs")

    log_directory = log_directory.resolve()
    log_directory.mkdir(parents=True, exist_ok=True)

    log_file = log_directory / "pipeline.log"

    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level,
    )

    logger.add(
        log_file,
        level=log_level,
        encoding="utf-8",
        rotation="10 MB",
        retention=5,
        enqueue=True,
    )