---
description: Add semantic motion graphics to a talking-head video (or revise an existing project)
argument-hint: <video-or-project> [instructions, e.g. "9:16 抖音, 克制一点" or "把数字放左边"]
---

Use the **semantic-motion** skill for this request: $ARGUMENTS

The CLI is `bash "${CLAUDE_PLUGIN_ROOT}/skills/semantic-motion/scripts/sm"` (if that variable is not set,
use `bash scripts/sm` inside the skill's directory).

1. If the target is a video without a `.smproj` next to it: run `sm doctor` (and `scripts/setup.sh`
   if needed), then `sm analyze <video>` (pass `--platform`/`--language` if the user mentioned them).
2. Read `brief.md`, direct the plan following the skill's design guide, `sm validate`, `sm preview`,
   look at the preview, then `sm build`.
3. If the target is an existing project, treat the instructions as a revision (see
   `reference/revision-guide.md`), rebuild, and show what changed with `sm diff`.
4. Before finishing, view `qa/contact_sheet.jpg` and read `qa/report.md`. Optionally hand both to the
   `motion-critic` agent for an independent review. Report the output path, the list of graphics
   (time → what → why) and the QA result.
