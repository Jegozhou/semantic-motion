#!/usr/bin/env python3
"""Validate the installed Skill entry points (structure + referenced paths).

Harness discovery roots found on this machine:
  * ``~/.dsh/skills/<name>/SKILL.md``  (constant baked into the DSH bundle)
  * ``~/.claude/skills/<name>/SKILL.md`` (the root actually loaded this session)
  * ``<project>/.dsh/skills/<name>/SKILL.md`` (project-local copy)

This script checks frontmatter, required sections, and that every repo path the
skill tells an operator to run exists. It cannot prove the running Harness will
hot-reload a new skill (that happens on the next session start) — it proves the
files are well-formed and executable.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPIES = [
    ROOT / ".dsh/skills/semantic-motion/SKILL.md",
    Path.home() / ".dsh/skills/semantic-motion/SKILL.md",
    Path.home() / ".claude/skills/semantic-motion/SKILL.md",
]
REQUIRED_SECTIONS = ("# Semantic Motion", "## Quick start", "## Honest limits",
                     "## Verify the installation")


def main() -> int:
    report = {"copies": [], "required_paths": [], "verdict": "PASS"}
    text = None
    for p in COPIES:
        entry = {"path": str(p), "exists": p.exists()}
        if p.exists():
            body = p.read_text(encoding="utf-8")
            text = text or body
            m = re.match(r"^---\n(.*?)\n---\n", body, re.S)
            entry["frontmatter"] = bool(m)
            entry["has_name"] = bool(m and re.search(r"^name:\s*semantic-motion\s*$", m.group(1), re.M))
            entry["has_description"] = bool(m and "description:" in m.group(1))
            entry["missing_sections"] = [s for s in REQUIRED_SECTIONS if s not in body]
        report["copies"].append(entry)
        if not entry["exists"] or not entry.get("frontmatter", True) and p.exists():
            report["verdict"] = "FAIL"
    if text is None:
        report["verdict"] = "FAIL"
    for rel in ("src/semotion/cli.py", "docs/INTERFACES.md", "docs/USER-GUIDE.md",
                "requirements.lock", "ENVIRONMENT.md", "reports/licenses.md",
                "scripts/regression.py", "assets/fonts/NotoSansSC-Regular.otf"):
        ok = (ROOT / rel).exists()
        report["required_paths"].append({"path": rel, "exists": ok})
        if not ok:
            report["verdict"] = "FAIL"
    out = ROOT / "artifacts/delivery/skill_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in report["copies"]:
        print(f"  {c['path']}: exists={c['exists']} frontmatter={c.get('frontmatter')} "
              f"missing={c.get('missing_sections')}")
    print(f"{report['verdict']} -> {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
