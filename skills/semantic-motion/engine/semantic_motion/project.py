"""Project directory: all state for one video lives here so work is resumable and editable.

<name>.smproj/
  project.json          source, settings, stage records (inputs hash -> freshness)
  analysis/             media.json, audio.npz, asr_raw.json, transcript.json, vision.npz, shots.json
  brief.md              LLM-facing structured description of the video
  plan.json             the semantic motion plan (authored by Claude or `sm auto-plan`)
  plans/                numbered snapshots of every plan revision (undo / diff)
  build/                timeline.json, compile_report.json
  renders/              preview.mp4, final.mp4, overlay.mov
  qa/                   report.json, report.md, contact_sheet.jpg, timeline.png, frames/
  history.jsonl         append-only event log
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .util import SMError, log, read_json, sha1_file, write_json

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts", ".3gp", ".flv", ".wmv", ".mpg", ".mpeg"}


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.state_path = self.root / "project.json"
        if not self.state_path.exists():
            raise SMError(f"not a Semantic Motion project: {self.root}")
        self.state = read_json(self.state_path)

    # ------------------------------------------------------------------ creation / lookup
    @staticmethod
    def default_root(video: Path) -> Path:
        return video.with_name(video.stem + ".smproj")

    @classmethod
    def create(cls, video: str | Path, root: Optional[str | Path] = None, **settings: Any) -> "Project":
        video = Path(video).resolve()
        if not video.exists():
            raise SMError(f"video not found: {video}")
        root = Path(root).resolve() if root else cls.default_root(video)
        root.mkdir(parents=True, exist_ok=True)
        for d in ("analysis", "plans", "build", "renders", "qa"):
            (root / d).mkdir(exist_ok=True)
        sp = root / "project.json"
        if sp.exists():
            st = read_json(sp)
            if Path(st["source"]["path"]) != video:
                st["source"]["path"] = str(video)
        else:
            st = {"engine_version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "source": {"path": str(video), "sha1_head": sha1_file(video, limit=8 << 20), "size": video.stat().st_size},
                  "settings": {"language": None, "asr_model": "large-v3-turbo", "platform": "auto", "style": "balanced"},
                  "stages": {}}
        for k, v in settings.items():
            if v is not None:
                st["settings"][k] = v
        write_json(sp, st)
        p = cls(root)
        p.event("create", source=str(video))
        return p

    @classmethod
    def open(cls, target: str | Path, create: bool = True, **settings: Any) -> "Project":
        t = Path(target)
        if t.is_dir() and (t / "project.json").exists():
            p = cls(t)
            if any(v is not None for v in settings.values()):
                for k, v in settings.items():
                    if v is not None:
                        p.state["settings"][k] = v
                p.save()
            return p
        if t.is_file() and t.suffix.lower() in VIDEO_EXT:
            root = cls.default_root(t.resolve())
            if (root / "project.json").exists():
                return cls.open(root, **settings)
            if create:
                return cls.create(t, **settings)
        if t.is_file() and t.name == "project.json":
            return cls(t.parent)
        raise SMError(f"cannot open project from '{target}' (give a video file or a .smproj directory)")

    # ------------------------------------------------------------------ paths / state
    def p(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def source(self) -> Path:
        return Path(self.state["source"]["path"])

    @property
    def settings(self) -> dict:
        return self.state["settings"]

    def save(self) -> None:
        write_json(self.state_path, self.state)

    def event(self, kind: str, **data: Any) -> None:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": kind, **data}
        with open(self.p("history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def stage(self, name: str) -> dict:
        return self.state.get("stages", {}).get(name, {})

    def stage_fresh(self, name: str, key: str, outputs: list[str]) -> bool:
        st = self.stage(name)
        return bool(st.get("key") == key and all(self.p(o).exists() for o in outputs))

    def mark_stage(self, name: str, key: str, **info: Any) -> None:
        self.state.setdefault("stages", {})[name] = {"key": key, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
        self.save()
        self.event(f"stage:{name}", **{k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))})

    # ------------------------------------------------------------------ loaders
    def media(self):
        from .media import MediaInfo

        d = read_json(self.p("analysis", "media.json"))
        d.pop("orientation", None)
        return MediaInfo(**d)

    def transcript(self) -> dict:
        return read_json(self.p("analysis", "transcript.json"))

    def vision(self):
        from .vision import VisionData

        return VisionData.load(str(self.p("analysis", "vision.npz")))

    def audio(self):
        import numpy as np

        from .audio import AudioFeatures

        return AudioFeatures.from_npz(np.load(self.p("analysis", "audio.npz")))

    def shots(self) -> list:
        return read_json(self.p("analysis", "shots.json"))

    def has(self, *parts: str) -> bool:
        return self.p(*parts).exists()

    # ------------------------------------------------------------------ plans
    def plan(self) -> dict:
        if not self.has("plan.json"):
            raise SMError("no plan.json yet — write one (see reference/plan-schema.md) or run `sm auto-plan`")
        return read_json(self.p("plan.json"))

    def snapshots(self) -> list[Path]:
        return sorted(self.p("plans").glob("plan.*.json"))

    def snapshot_plan(self, note: str = "") -> Optional[Path]:
        """Store the current plan.json as the next numbered snapshot (skips if identical to the last one)."""
        if not self.has("plan.json"):
            return None
        cur = self.p("plan.json").read_bytes()
        snaps = self.snapshots()
        if snaps and snaps[-1].read_bytes() == cur:
            return snaps[-1]
        n = int(snaps[-1].stem.split(".")[1]) + 1 if snaps else 1
        dst = self.p("plans", f"plan.{n:03d}.json")
        dst.write_bytes(cur)
        self.event("snapshot", file=dst.name, note=note)
        return dst

    def undo(self) -> Path:
        snaps = self.snapshots()
        cur = self.p("plan.json").read_bytes() if self.has("plan.json") else b""
        # find the latest snapshot that differs from the current plan
        cands = [s for s in snaps if s.read_bytes() != cur]
        if not cands:
            raise SMError("nothing to undo")
        prev = cands[-1]
        if cur:
            self.snapshot_plan(note="before undo")
        shutil.copyfile(prev, self.p("plan.json"))
        self.event("undo", restored=prev.name)
        return prev

    def write_plan(self, plan: dict, note: str = "") -> None:
        write_json(self.p("plan.json"), plan)
        self.snapshot_plan(note)


def find_project(target: Optional[str]) -> Project:
    if target:
        return Project.open(target)
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        if (d / "project.json").exists():
            return Project(d)
    cands = list(cwd.glob("*.smproj"))
    if len(cands) == 1:
        return Project(cands[0])
    raise SMError("no project given; pass a video or .smproj path")


def env_threads() -> int:
    try:
        return int(os.environ.get("SM_THREADS", "0"))
    except ValueError:
        return 0
