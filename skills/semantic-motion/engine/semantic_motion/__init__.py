"""Semantic Motion engine.

Pipeline: analyze (media + audio + ASR + vision) -> brief (LLM-facing) -> plan (semantic DSL,
authored by Claude or the heuristic auto-planner) -> compile (anchors, timing, layout, style ->
resolved timeline) -> render (deterministic Skia compositor) -> QA (objective checks + review sheets).
"""
__version__ = "0.1.0"
RENDERER_VERSION = "1"
