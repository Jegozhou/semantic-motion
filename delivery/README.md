# Delivery — Semantic Motion

每个目录含成片、Motion IR、渲染清单、指标 JSON 与数值校验。

- `01_g1_keyword_number_9x16/` — 9:16 1080x1920 · 关键词+数字（Smart Balanced）
- `02_g2_list_card_16x9/` — 16:9 1920x1080 · 列表+卡片+数字
- `03_g2_real_face_avoidance/` — 9:16 · 真实人脸（合成夹具）· 四类动效组合
- `04_g3_camera_cut_safety/` — 9:16 · 双镜头 pan/zoom + 真实人脸 · 切镜安全
- `05_g1_real_footage_16x9/` — 16:9 1280x720 · 公有领域真人影像

## G5 — LLM 选动效 + Remotion + 字幕

- `demo_llm_remotion/` — before/after + storyboard：LLM 决策 5 个元素（关键词/数字/列表）+ 逐词简体字幕
- `demo_llm_face/` — 真实人脸素材：LLM 决策 7 个元素，遮脸 0
- 成片与工程：`artifacts/g5/llm_a/`、`artifacts/g5/llm_face/`（含 `motion_ir.json`、`captions.json`、`remotion_props.json`、`metrics.json`）
- 报告：`reports/gate-G5.md`；契约：`docs/REMOTION.md`
