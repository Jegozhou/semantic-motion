# Semantic Motion · 语义动效导演（Claude Code Skill / Plugin）

把一条**原始口播视频**交给 Claude，它会像专业动效设计师一样：听懂你在说什么（语义、逻辑、数字、情绪、节奏、停顿），
自己决定**在哪里、加什么动效**——关键词、数据计数、图表、列表、卡片、流程/关系图、姓名条、指示标注、镜头推进，
或者为某个独特想法临时组合一个程序化动画；每个动效**精确锚定到说出的那个字/词**，**避开人脸、看懂手势与留白**再落位；
保持原人物画面与原声不变；渲染后**自动质检、自动修复**；之后你可以用一句话改稿（“数字放左边”“换蓝色”“删掉那个”“撤销”）。
最终交付成片 + 可编辑工程（plan / timeline / 带 alpha 的图形层）。

> English: A Claude Code skill that turns a raw talking-head video into a motion-designed one. Claude reads a structured
> brief of the speech and the frame, decides what deserves a graphic, and writes a semantic plan; a deterministic local engine
> anchors every graphic to the spoken words (frame-accurate), solves a face-safe layout, renders with Skia, runs QA with
> self-repair, and supports natural-language revisions with undo. Chinese + English, 16:9 + 9:16. No API keys, no paid services.

## 效果与验证
真实素材（Wikimedia Commons 公开许可视频）上的完整运行、QA 指标与联系表见 **[docs/TEST_REPORT.md](docs/TEST_REPORT.md)**：
中文横屏、中文竖屏（抖音安全区）、英文横屏（含改稿 → diff → 撤销流程）、快语速 + 大量 B-roll 的高难度中文片段均通过自动 QA：音频逐比特一致、帧数/时长一致、
A/V 偏移 0、图形外区域 PSNR ≥ 45 dB、落点与锚点偏差 ≤ 1 帧、人脸零遮挡；同一工程两次渲染逐帧哈希一致。词级时间精度另有**合成**基准（TTS 真值，已明确标注）。

## 安装

### 方式 A：作为插件（推荐）
在 Claude Code 中：
```
/plugin marketplace add Jegozhou/semantic-motion
/plugin install semantic-motion@semantic-motion
```
（仓库为私有时，需要本机 git 能访问该仓库，例如已 `gh auth login`。）

### 方式 B：只装技能
```bash
git clone https://github.com/Jegozhou/semantic-motion.git
cp -r semantic-motion/skills/semantic-motion ~/.claude/skills/        # 个人级；或复制到项目的 .claude/skills/
```

### 首次初始化（两种方式都需要，一次即可）
第一次使用时 Claude 会自动运行 `sm doctor`，缺什么就运行安装脚本；也可以手动：
```bash
bash ~/.claude/skills/semantic-motion/scripts/setup.sh        # 插件方式下脚本位于插件目录的 skills/semantic-motion/scripts/
```
它会在 `~/.semantic-motion/` 创建独立 Python 虚拟环境、安装依赖、下载视觉模型（约 25 MB）与 Whisper large-v3-turbo（约 1.6 GB，免费、无需账号）。
- 需要：macOS 或 Linux（Windows 请用 WSL2）、Python ≥ 3.10、ffmpeg；建议 8 GB 内存。无需 GPU。
- Linux 若缺 `libEGL`/`libGLESv2`，脚本会提示安装命令。Apple Silicon 会额外尝试安装 mlx-whisper 加速。
- 慢机器/英文素材可 `--asr-model small`；中文建议保持 large-v3-turbo。

## 使用
直接对 Claude 说：
- “给 `~/Videos/talk.mp4` 加动效，发抖音，克制一点”
- “Add motion graphics to talk.mp4 for YouTube”
- 或用斜杠命令：`/motion ~/Videos/talk.mp4 9:16 抖音`

Claude 会：`sm analyze`（ASR + 声学精修 + 视觉分析，生成 `brief.md`）→ 阅读后写 `plan.json` → `sm validate` → `sm preview`
（看静帧）→ `sm build`（渲染 + 自动修复 + QA + 联系表）→ 亲自查看联系表与报告 → 交付并逐条说明“几秒、加了什么、为什么”。

改稿：在同一会话或之后任何时候说“把 +18 放左边、换成蓝色、删掉 With you in spirit、列表早一点出来”，Claude 修改计划并重建；
`sm diff` 显示字段级改动，`sm undo` 回到上一版。

### 产出（`<video>.smproj/`）
| 路径 | 内容 |
|---|---|
| `renders/final.mp4` | 成片（原画面 + 原声 + 动效） |
| `renders/overlay.mov` | `--overlay`：ProRes 4444 带 alpha 的纯图形层，供剪辑软件二次调整 |
| `plan.json` / `plans/` | 语义计划与每一版快照（可撤销） |
| `build/timeline.json` | 精确到帧/像素的时间线（可编辑、可复现） |
| `qa/report.md`, `qa/contact_sheet.jpg`, `qa/timeline.png` | 质检报告、成片抽帧联系表、时间线条带图 |
| `brief.md`, `analysis/` | 分析结果（逐字时间、停顿、重音、镜头、人脸/身体/手势、画面文字） |
| `history.jsonl` | 操作日志（可随时中断、恢复） |

### CLI 速查（`bash skills/semantic-motion/scripts/sm <命令>`；安装脚本会为它加可执行权限）
`doctor` · `setup` · `analyze` · `brief` · `transcript` · `auto-plan` · `validate` · `compile` · `preview` · `build [--preview|--overlay|--determinism]` ·
`render` · `qa` · `export` · `status` · `diff` · `undo` · `snapshot`

## 元素库
`keyword` 关键词花字 · `stat` 数字计数 · `list` 随说随出的列表 · `flow` 流程/因果 · `compare` 对比 · `chart` 柱/条/折线/环形图 ·
`quote` 金句 · `title` 章节卡 · `lower_third` 姓名条 · `callout` 指示标注 · `captions` 字幕（可选，检测到烧录字幕时拒绝）·
`camera` 推镜/冲镜 · `custom` 由图层 + 关键帧组合的程序化动画（关键帧可锚定到说出的词）。
详见 [skills/semantic-motion/reference/elements.md](skills/semantic-motion/reference/elements.md)。

## 性能（2 核 CPU、无 GPU 沙箱实测）
分析 ≈ 1–1.5× 视频时长（ASR ≈ 1×、视觉 ≈ 1×）；渲染 720p ≈ 29 fps、1080p 横屏 ≈ 12 fps、1080×1920 竖屏 ≈ 9.4 fps。
Apple Silicon / 多核机器会快得多。长视频先 `sm build --preview`（半分辨率）再出终版。

## 已知限制（诚实说明）
- 计划质量取决于 Claude 的导演判断；无 LLM 的 `sm auto-plan` 只是保守基线，在快语速/误识别多的素材上可能几乎不加东西。
- ASR 会误识别专有名词（如“百老汇→百老会”、“OpenClaw→OpenCloud”）：用计划里的 `corrections` 纠正，或提供字幕文件 `--transcript`。
- 测试素材是公开发布的成片（带烧录字幕/姓名条/跳切），竖屏样片是由横屏中心裁切派生的，不是原生竖屏拍摄。
- 词级时间精度基准使用 TTS 合成语音作为真值（真实人声没有逐词真值）；真实素材上我们验证的是“图形落点 = 锚点词时间”（0 ms 偏差）与人工目检。
- 纯 CPU 渲染，4K 输出较慢；HDR 素材会 tone-map 为 SDR BT.709 输出。

## 开发
```bash
python -m pip install -r skills/semantic-motion/requirements.txt pytest
python -m pytest -q tests                                   # 单元测试
SM_TEST_PROJECTS=/path/a.smproj:/path/b.smproj python -m pytest -q tests   # 加上真实工程的集成测试
python tests/bench_timing.py --out /tmp/sm_bench            # 合成时间基准（edge-tts）
```
设计与架构：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · 决策记录：[docs/DECISIONS.md](docs/DECISIONS.md) · 进度：[docs/STATUS.md](docs/STATUS.md) ·
素材与署名：[docs/TEST_FOOTAGE.md](docs/TEST_FOOTAGE.md)

## 许可
MIT（见 [LICENSE](LICENSE)）。测试视频的版权归原作者，许可与署名见 docs/TEST_FOOTAGE.md。
本项目借鉴了“语义驱动动效”的公开理念，但代码、设计系统与流程均为独立实现。
