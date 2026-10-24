"""Shared pytest setup: make the engine importable without installing it."""
import os
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "skills" / "semantic-motion" / "engine"
sys.path.insert(0, str(ENGINE))

# Optional real projects for integration tests (skipped when absent):
#   SM_TEST_PROJECTS=/data/runs/zh_talk_16x9.smproj:/data/runs/en_talk_16x9.smproj pytest tests
REAL_PROJECTS = [p for p in os.environ.get("SM_TEST_PROJECTS", "").split(os.pathsep) if p and Path(p).is_dir()]
