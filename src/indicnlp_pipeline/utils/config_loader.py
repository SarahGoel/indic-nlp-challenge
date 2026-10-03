from pathlib import Path
from typing import Any

import yaml


def find_repository_root(start_path: Path | None = None) -> Path:
    """Locate the repository root containing configs/config.yaml."""
    current = (start_path or Path(__file__)).resolve()

    if current.is_file():
        current = current.parent

    for path in (current, *current.parents):
        if (path / "configs" / "config.yaml").is_file():
            return path

    raise FileNotFoundError(
        "Could not locate repository root containing configs/config.yaml."
    )


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file."""
    repository_root = find_repository_root()

    path = (
        config_path
        if config_path is not None
        else repository_root / "configs" / "config.yaml"
    )

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return config


def resolve_path(path_value: str | Path, repository_root: Path | None = None) -> Path:
    """Resolve a configured path relative to the repository root."""
    root = repository_root or find_repository_root()
    path = Path(path_value)

    if path.is_absolute():
        return path.resolve()

    return (root / path).resolve()


def resolve_config_paths(
    config: dict[str, Any],
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve configured repository paths into pathlib.Path objects."""
    root = repository_root or find_repository_root()

    resolved_config = dict(config)

    paths = config.get("paths", {})

    if not isinstance(paths, dict):
        raise ValueError("'paths' must be a mapping in config.yaml.")

    resolved_config["paths"] = {
        name: resolve_path(value, root)
        for name, value in paths.items()
    }

    return resolved_config