"""Load per-size best hyperparameter configs from neural/config/best/<size>.yaml."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "best"


def load_best_config(problem_size: int) -> dict:
    """Load the best config for a given problem size.

    Returns a flat dict compatible with train.py's cfg format.
    Raises FileNotFoundError if no config exists for that size.
    """
    path = CONFIG_DIR / f"{problem_size}.yaml"
    if not path.exists():
        available = sorted(int(p.stem) for p in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"No config for size {problem_size}. Available: {available}"
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Flatten the nested structure into the flat dict train.py expects
    cfg: dict = {}
    cfg.update(raw.get("model", {}))
    cfg.update(raw.get("training", {}))
    if "data" in raw and "path" in raw["data"]:
        cfg["data_path"] = raw["data"]["path"]

    return cfg


def get_model_config(problem_size: int) -> dict:
    """Load only the model architecture params needed for inference."""
    path = CONFIG_DIR / f"{problem_size}.yaml"
    if not path.exists():
        available = sorted(int(p.stem) for p in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"No config for size {problem_size}. Available: {available}"
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    model_cfg = raw.get("model", {})
    model_cfg["problem_size"] = problem_size
    return model_cfg
