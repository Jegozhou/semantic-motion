# Gate G5 — LLM 选动效 + Remotion 渲染 + 逐词字幕

**判定：PASS**（新增可选渲染后端与 LLM 导演层；原有 `pillow_ffmpeg` 后端不受影响）
日期：2026-10-24（Asia/Shanghai）｜契约：`docs/REMOTION.md`（remotion-1.0.0）

---

## 1. 这一闸门解决什么

用户要求：**让 LLM 自己判断该用什么 Remotion 动效，跟随时间戳与字幕**。

G1–G4 的实现是"规则导演 + 自研栅格化 + ffmpeg overlay"。G5 增加两条能力，且不推翻旧路径：

1. **LLM 导演层**：LLM 阅读词级转录 + 语义锚点 + 画面信息，输出结构化决策（动效类型、理由、停留时长、置信度），经 8 条规则校验后冻结进 Motion IR。
2. **Remotion 渲染后端**：动效由 React 组件（spring/easing 驱动）渲染，字幕按词级时间烧录；同一份 IR 可切换后端。

## 2. 验收证据

| 项 | 结果 | 证据 |
|---|---|---|
| Remotion 可安装、可渲染 | 4.0.538，系统 Chrome，1080×1920 渲染成功 | `remotion/out/props_test.mp4` |
| LLM 决策被真正采用 | fx_a 5 元素 / fx_face 7 元素，`decision_model=deepseek-v4.1-flash` | `artifacts/llm/*.decisions.json` |
| 决策通过校验 | `validate_decisions` → `[]`（8 条规则） | 同上 + `tests/test_llm_director.py` |
| 端到端渲染（9:16） | 605 / 623 帧，24.2 / 24.92 s，h264+aac | `artifacts/g5/llm_a/`、`artifacts/g5/llm_face/` |
| 音频保持 | 时长差 0.000 s、RMS 差 0.009 dB | `render_manifest.json` |
| 时间精度 | P50 5.75 / 15.38 ms，P90 23.54 / 29.08 ms | `metrics.json → timing` |
| 遮脸 | 0 / 133、0 / 150 采样帧 | `position.observed.face_overlap` |
| 框外像素 | 0.0 | `position.raster_containment` |
| 切镜安全 | PASS | `position.cut_safety` |
| 字幕轨 | 10 / 10 条有效 cue，词级帧号；重复文案被抑制 4 / 2 条 | `captions.json`、`remotion_props.json` |
| 可复现 | 两次渲染 max_pixel_diff 3.5e-05（容差 0.5）→ PASS | `artifacts/g5/determinism/determinism.json` |
| 回归测试 | **144 passed**（含 52 项 LLM 导演/字幕测试） | `pytest tests/ -q` |

## 3. LLM 与规则导演的差异（同一素材）

| 素材 | 规则导演 | LLM 导演 |
|---|---|---|
| fx_a（24.2 s） | 5 元素：痛点、其实是节奏、九成、卡片「都卡在三个问题」、流程 | 5 元素：痛点、其实是节奏、九成、**三步列表**（不知道什么时候加字 / 加了又挡住脸 / 该一次就要重做一遍）、流程 |
| fx_face（24.9 s） | 5 元素，列表只合并两步，保留与原则句重复的卡片 | 7 元素：三个、九成、**完整三步列表**、关键是别挡住脸、70%、结论卡、底线 |

LLM 的取舍理由写在每条 decision 的 `reason` 里（例如"三步并列按口播顺序逐行揭示，比三条独立关键词更克制""结论句需要完整呈现，卡片承载最合适"），并冻结进 IR，可复查。
**诚实说明**：LLM 输出的可复现范围是"冻结后的决策与 IR"，不是"换一个会话还会做同样选择"（规格 §3.5 原文要求）。

## 4. 字幕（跟随时间戳）

* 由 `transcript.words` 生成，**每屏 ≤2 行、9:16 每行 ≤22 显示单位、16:9 ≤30**（CJK=1，ASCII=0.5，英文不拆词）。
* 每条 cue 带 `startFrame/endFrame/lines/words[]`（词级帧号），首帧严格等于该屏首词的帧。
* 显示层统一**简体**（1:1 折叠，保留 `raw_text` 不破坏偏移），行尾标点去掉。
* **去重**：当同一时刻的动效文案与字幕完全重复（例如关键词卡"其实是节奏"与字幕"其实是节奏"），字幕让位并被记录（`captionsSuppressed`），避免同字叠两遍；只覆盖大部分文案时才抑制（"痛点"不会压掉整句字幕）。

## 5. 性能（实测，本机）

| 项目 | 数值 |
|---|---|
| Remotion 渲染 1080×1920 / 605 帧（24.2 s） | ≈50 s |
| Remotion 渲染 1080×1920 / 623 帧（24.9 s） | ≈50 s |
| 旧后端同素材 | ≈3 s（但动效表现力受限） |
| 磁盘 | `remotion/node_modules` 238 MB；headless shell 首次下载失败后改用系统 Chrome |

## 6. 契约与实现

* `docs/REMOTION.md`：props schema、LLM 决策 schema、8 条校验、字幕轨、度量复用。
* `src/semotion/remotion_bridge.py`：IR → props（含列表行 reveal 帧、数字 prefix/suffix、字幕去重）→ 渲染 → manifest。
* `src/semotion/llm_director.py`（子智能体交付，1395 行）：提示词、校验、to_plan、caption_cues。
* `remotion/src/`：`Keyword / Number / List / Card / Captions` 五个白名单组件；无 LLM 生成代码被执行（props 只含数据）。
* CLI：`run --backend remotion --decisions <json> --captions`。

## 7. 已知限制（如实）

| ID | 项 | 说明 |
|---|---|---|
| M1 | LLM 决策来自本会话（harness 模型），非外部 API | 已冻结 JSON + 提示词全文；换模型需重新生成并重校验 |
| M2 | Remotion 非逐帧比特一致 | 两次渲染差异 3.5e-05（Chrome 抗锯齿/编码），按**显式容差 0.5** 判定 |
| M3 | Remotion 许可 | 个人/≤3 员工/非营利免费；更大营利组织需 Company License（`reports/licenses.md`） |
| M4 | 首屏 headless-shell 启动失败 | 改用系统 Chrome（`--browser-executable`），已在 bridge 内自动探测 |
| M5 | 字幕以单行为主 | 长句 2 行路径由单测覆盖，真实素材偏短句 |
| M6 | 真人语音真值 | 仍为 NOT RUN（与 G1 同一限制） |
| M7 | 视觉跟踪 / 帧级局部重渲染 | 仍未实现（P2） |

## 8. 复现

```bash
cd /Volumes/ZHITAI/SemanticMotion && export PYTHONPATH=src
# LLM 决策 → IR → Remotion → 字幕 → 指标
.venv/bin/python -m semotion.cli run --fixture fx_face_zh_portrait --outdir artifacts/g5/llm_face \
  --gate G5 --backend remotion \
  --decisions artifacts/llm/fx_face_zh_portrait.decisions.json --captions
# 旧后端对照
.venv/bin/python -m semotion.cli run --fixture fx_face_zh_portrait --outdir artifacts/g5/legacy_face --gate G5
# 测试
.venv/bin/python -m pytest tests/ -q     # 144 passed
```

## 9. 结论

G5 **PASS**：LLM 的选择被真实执行到成片上，时间跟随词级时间戳、字幕逐词同步、真实人脸遮脸 0，且所有结论都有可复查的 JSON/视频与测试。
它是**增量交付**：G0–G4 的旧后端与既有证据仍然有效；选择哪个后端由 `--backend` 决定，二者共用同一份 Motion IR 与度量。
