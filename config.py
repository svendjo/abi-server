"""Environment configuration, loaded from config/<APP_ENV>.yaml.

One YAML file per environment (config/local-dev.yaml, config/aws-prod.yaml); pick
one with the APP_ENV env var (default "local-dev"), e.g.:

    python server.py                      # local-dev
    APP_ENV=aws-prod python server.py     # aws-prod

The file carries the run flags (reload / use_ctc / use_trocr / debug_crops) and the
results-storage backend, so the run command itself stays just `python server.py`.
"""
import os
from pathlib import Path

import yaml

APP_ENV = os.environ.get("APP_ENV", "local-dev")
_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _load(env):
    path = _CONFIG_DIR / f"{env}.yaml"
    if not path.exists():
        have = ", ".join(sorted(p.stem for p in _CONFIG_DIR.glob("*.yaml"))) or "none"
        raise SystemExit(f"APP_ENV={env!r}: missing config {path} (available: {have})")
    with open(path) as f:
        return yaml.safe_load(f) or {}


CONFIG = _load(APP_ENV)

USE_CTC = bool(CONFIG.get("use_ctc"))
USE_TROCR = bool(CONFIG.get("use_trocr"))
DEBUG_CROPS = bool(CONFIG.get("debug_crops"))
RELOAD = bool(CONFIG.get("reload"))
RESULTS = CONFIG.get("results") or {}
