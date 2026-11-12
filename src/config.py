"""Project configuration and path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve all project paths to absolute paths."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "config.yaml"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config["_project_root"] = PROJECT_ROOT
    config["_config_path"] = path.resolve()

    for key, value in config.get("paths", {}).items():
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        config["paths"][key] = resolved.resolve()

    for key in ("raw_dir", "source_dir", "normalized_dir", "processed_dir",
                "features_dir", "errors_dir", "models_dir", "reports_dir", "tracking_dir"):
        if key not in config.get("paths", {}):
            raise KeyError(f"Missing required paths.{key} in configuration")
        config["paths"][key].mkdir(parents=True, exist_ok=True)

    #config.setdefault("mlflow", {})["tracking_uri"] = config["paths"]["tracking_dir"].as_uri()    
    # Preserve the tracking URI defined in config.yaml.
    if not config.get("mlflow", {}).get("tracking_uri"):
        raise ValueError(
            "Missing mlflow.tracking_uri in configuration file"
        )
        
    return config
