"""Small .env loader for local prompt baseline configuration."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding existing env vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_project_env(root: Path) -> None:
    load_env_file(root / ".env")
