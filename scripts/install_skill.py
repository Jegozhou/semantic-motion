#!/usr/bin/env python3
"""Install the Semantic Motion skill into the Claude Code skill roots.

The agent's own file sandbox usually cannot write outside its workspace, so this
script is meant to be run once by the user (or by an unrestricted shell):

    python3 "/path/to/Semantic Motion/scripts/install_skill.py"

It copies SKILL.md + references/ into every root that exists:

    ~/.claude/skills/semantic-motion/     (Claude Code)
    ~/.dsh/skills/semantic-motion/        (DeepSeek Harness)
    <repo>/.dsh/skills/semantic-motion/   (project-local copy, already present)

The skill body references $SM_ROOT (the repo) for code and the venv; installing
only copies the instructions, never the multi-hundred-MB artifacts.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "semantic-motion"
SRC = ROOT / ".dsh" / "skills" / NAME
DEFAULT_TARGETS = [
    Path.home() / ".claude" / "skills" / NAME,
    Path.home() / ".dsh" / "skills" / NAME,
]


def validate(src: Path) -> dict:
    md = src / "SKILL.md"
    if not md.exists():
        return {"ok": False, "error": f"missing {md}"}
    body = md.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", body, re.S)
    if not m:
        return {"ok": False, "error": "missing YAML frontmatter"}
    fm = m.group(1)
    name = re.search(r"^name:\s*(\S+)", fm, re.M)
    desc = re.search(r"^description:\s*\|?\s*\n?", fm, re.M)
    problems = []
    if not name or name.group(1) != NAME:
        problems.append(f"frontmatter name must be {NAME}")
    if not desc:
        problems.append("frontmatter needs a description")
    if len(body) < 800:
        problems.append("SKILL.md is suspiciously short")
    for section in ("## Quick start", "## Directing", "## Honest limits"):
        if section not in body:
            problems.append(f"missing section {section}")
    return {"ok": not problems, "problems": problems, "bytes": len(body)}


def install(target: Path, dry_run: bool = False) -> dict:
    if dry_run:
        return {"target": str(target), "status": "would install"}
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "SKILL.md", target / "SKILL.md")
    ref_src = SRC / "references"
    if ref_src.exists():
        ref_dst = target / "references"
        if ref_dst.exists():
            shutil.rmtree(ref_dst)
        shutil.copytree(ref_src, ref_dst)
    return {"target": str(target), "status": "installed",
            "files": sorted(p.name for p in target.rglob("*") if p.is_file())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", default=None,
                    help="override install roots (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    check = validate(SRC)
    report = {"source": str(SRC), "validation": check, "installs": []}
    if not check["ok"]:
        print(json.dumps(report, indent=2) if args.json else f"FAIL: {check}")
        return 1
    targets = [Path(t) for t in args.target] if args.target else DEFAULT_TARGETS
    failed = 0
    for t in targets:
        try:
            report["installs"].append(install(t, args.dry_run))
        except OSError as exc:
            failed += 1
            report["installs"].append({"target": str(t), "status": "denied",
                                       "error": str(exc)})
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for row in report["installs"]:
            print(f"{row['status']:10s} {row['target']} {row.get('error','')}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
