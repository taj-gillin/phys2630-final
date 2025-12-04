import yaml
from pathlib import Path


def load_config(path: str):
    cfg_path = Path(path)
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str | Path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

