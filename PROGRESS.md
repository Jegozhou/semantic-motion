# PROGRESS.md — Semantic Motion 可恢复进度台账

> 唯一事实来源。任何"完成"必须对应本文件中的证据路径；未运行写 `NOT RUN`，失败写 `FAIL`，受阻写 `BLOCKED`。
> 最近更新：G0 PASS；G1 链路已打通（真实成片+指标），等待校准素材凑够 ≥50 独立事件后判定。

## 0. 当前状态一屏

| 闸门 | 状态 | 证据 |
|---|---|---|
| **G0 预检与基线** | **PASS** | `ENVIRONMENT.md`、`artifacts/env/env_probe.json`、`artifacts/g0/baseline_report.json`（verdict=PASS）、`reports/gate-G0.md` |
| G1 最小语义闭环 | **IN PROGRESS** | 链路已打通：`artifacts/g1/runner/gate_summary.json`（fx_a：timing/position/render PASS，n=3 探索性）、`artifacts/g1/pd_a/`（真实人脸素材：position/render PASS）；**待 ≥50 独立事件校准素材** |
| G2 四类动效 | NOT RUN | 依赖 G1 |
| G3 高级稳定性 | NOT RUN | 依赖 G2 |
| G4 完整交付 | NOT RUN | 依赖 G3 |

预算：本地零付费；已用磁盘 < 2 GB（不含 1.42 GB whisper medium 缓存，位于 `/Volumes/ZHITAI/models` 与 `~/.cache/whisper`）；并发子智能体 2（上限 3）。

## 1. 已完成（可复查）

### G0.1 环境预检
- 命令：`.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json`
- 产物：`artifacts/env/env_probe.json`（host / paths / tools / capabilities / network）
- 结论：macOS 26.2 arm64、Apple M4 10 核、16 GB、工作盘 307 GB 可用；ffmpeg 8.1（480 滤镜，**无 libass/freetype ⇒ 无 drawtext/subtitles**）；Python 3.12 venv（numpy/pillow/opencv）；faster-whisper medium int8 CPU（RTF 0.31）。

### G0.2 工程初始化
- `pyproject.toml`、`.venv`、`src/semotion/{media,fixtures,baseline,semantic,verify,cli}.py`
- 契约：`docs/INTERFACES.md`（contracts-1.0.0）
- 数据契约 schema：`semotion.transcript/1`、`semotion.vision/1`、`semotion.anchors/1`、`semotion.plan/1`、`semotion.motion-ir/1`、`semotion.render/1`、`semotion.metrics/1`

### G0.3 真实可播放基线（透传 + 重编码）
- 输入：`tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4`（1080×1920, 25 fps, 24.2 s, h264+aac, sha256 见 report）
- 输出：`artifacts/g0/baseline_remux.mp4`、`artifacts/g0/baseline_reencode.mp4`
- 报告：`artifacts/g0/baseline_report.json` → `verdict: PASS`
  - remux：605/605 帧哈希一致、解码 PCM 音频哈希一致、时长/流/几何一致
  - re-encode：605 帧最大像素差 0.0118（阈值 12）、时长差 0.000 s、h264+aac
- 复现：
  ```bash
  PYTHONPATH=src .venv/bin/python -m semotion.cli baseline run --input tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4 --out artifacts/g0
  ```

### G0.4 合成校准素材（SYNTH_KNOWN）
- 生成器：`src/semotion/fixtures.py`（逐单元 TTS + 静音拼接，起止时间由 ffprobe 实测，**不经 ASR**）
- 已生成：`fx_a_zh_portrait`（1080×1920，14 个语音单元，24.20 s）
- 真值：`tests/fixtures/clips/fx_a_zh_portrait/ground_truth.json`（units[] 含 start_s/end_s/start_frame/end_frame/audio_sha256，truth=SYNTH_KNOWN）
- 多镜头能力：`scenes` + `cuts_at_s`，已用 `fx_test_cuts` 验证（5.0 s 切镜，slow_pan 第二镜头）

## 1.5 G1 已跑通的实测（2026-10-24 17:4x）

| 项 | 结果 | 证据 |
|---|---|---|
| 端到端（竖屏合成素材 fx_a） | `run --fixture fx_a_zh_portrait` 全链路 | `artifacts/g1/runner/fx_a_zh_portrait/run_summary.json` |
| 时间（真实渲染 onset vs 独立单元真值） | P50 45.8ms / P90 68.0ms / max 68.0ms，**n=3（探索性，不作 P90 达标宣称）** | 同上 `metrics.json` → `timing.rows` |
| 渲染保真度（onset vs IR 锚点） | 中位 40ms = 1 帧 | `metrics.json` → `timing.fidelity` |
| 位置（计划值） | 遮脸 0、越界 0、碰撞 0 | `metrics.json` → `position.planned` |
| 位置（观测值：渲染帧 vs 源帧差异包围盒） | 遮脸 0、越界 0 | `position.observed` |
| 栅格包含性（画布外/框外像素占比） | 0.0（138,548/138,548 像素在声明框内） | `position.raster_containment` |
| 可复现性（同 IR 渲染两次） | **逐帧哈希完全一致**，605/605 帧 | `gate_summary.json` → `determinism` |
| 真实人脸素材（PD，1280×720，英文） | 1 个 number 元素 `[885,262,150,181]`，与人脸 union `[491,87,280,392]` 交集 0；position/render PASS | `artifacts/g1/pd_a/`、抽帧 `/tmp/pd_render.png`（人工目检） |
| 数值语义准确性 | 锚点「九成」→ 计数动画终值 **90%**（人工抽帧核对 8.55s） | 目检；自动化校验待补 |

**度量反向验证（证明指标不是空转）**：把 400×120 的框故意压在真实人脸框上 → `position=FAIL`（observed=1.0、planned=1.0）；把 onset 故意延后 300ms → `timing=FAIL`（300.0ms）。证据脚本输出见对话记录与 `artifacts/g1/probe/`。

## 2. 进行中（G1）

| 子任务 | Owner | 写域 | 状态 | 备注 |
|---|---|---|---|---|
| ASR + 词级时间（faster-whisper，缓存，能量起音细化） | speech-vision | `src/semotion/asr.py` | 实现中 | 目标：转录 JSON + 起音误差统计 |
| 人脸/镜头/安全区（YuNet, 5 Hz） | speech-vision | `src/semotion/vision.py` | 实现中 | 无脸时降级 |
| 语义锚点分类 + 真值加载 | speech-vision | `src/semotion/anchors.py` | 实现中 | 时间只来自词/段/能量起音 |
| Motion IR（8 条不变量）+ 哈希 | motion-render | `src/semotion/ir.py` | 实现中 | |
| 布局求解（避脸/边界/碰撞/降级） | motion-render | `src/semotion/design.py` | 实现中 | |
| 四类动效栅格化（RGBA PNG） | motion-render | `src/semotion/raster.py`, `effects/` | 实现中 | 先 keyword 打通 |
| ffmpeg overlay 合成渲染 | motion-render | `src/semotion/render.py` | 实现中 | 确定性 + 音频保持 |
| 导演规则（Smart Balanced 节奏预算） | lead | `src/semotion/semantic.py` | ✅ 已写 | 无 LLM，可解释；普通段零动效 |
| 度量引擎（时间/位置/渲染/确定性） | lead | `src/semotion/verify.py` | ✅ 已写，待集成验证 | 渲染 onset 由解码像素测得 |
| CLI 端到端 | lead | `src/semotion/cli.py` | ✅ 已写 | `run --fixture fx_a_zh_portrait` |

## 3. 下一步（按顺序，可直接执行）

1. speech-vision / motion-render 交付模块 + 单测全绿（等待回报）。
2. Lead 集成：`PYTHONPATH=src .venv/bin/python -m semotion.cli run --fixture fx_a_zh_portrait --outdir artifacts/g1/run_a --gate G1`
3. 检查成片（抽帧肉眼 + 指标）：字卡是否与语义对齐、是否避脸、是否越界。
4. 写 `reports/gate-G1.md`；只有 metrics 的 timing/position/render 全 PASS 才记 G1 PASS。
5. G2：补 number/list/card 与 16:9 素材（`fx_b_zh_landscape`），两条不同素材成片 + 组合测试。
6. G3：多镜头素材、降级策略、局部重渲染、确定性双跑、异常恢复、回归脚本。
7. G4：`~/.dsh/skills/semantic-motion/SKILL.md`、使用指南、指标与限制、许可证清单。

## 4. 已知阻碍与风险

| 项 | 影响 | 处置 |
|---|---|---|
| 无真人中文口播素材（用户未提供） | HUMAN 真值维度 `NOT RUN`；审美验收只能称"内部评估" | 用 SYNTH_KNOWN 校准 + 公开 PD 素材（待引入），报告中如实标注覆盖范围 |
| ffmpeg 无 libass/freetype | 不能用 drawtext/subtitles | 已定架构：Python 栅格化 RGBA + overlay（`docs/INTERFACES.md` §1） |
| openai-whisper MPS 在 word_timestamps 崩溃 | 不能用 GPU 加速词级时间 | 改用 faster-whisper CPU int8（RTF 0.31），已固化 |
| 系统盘仅剩 3 GB | 不能写 `/tmp` 大文件 | 中间产物一律写 `/Volumes/ZHITAI/SemanticMotion/artifacts`；`/tmp` 只放小文件 |
| 子智能体无独立沙箱 | 不能宣称严格隔离/独立盲测 | 按文件写域隔离；QA 称"内部回归"，不称独立盲测 |
| ffmpeg GPL 构建 | 分发捆绑有 GPL 义务 | 作为外部依赖调用，不随 Skill 分发；`reports/licenses.md` 记录 |

## 5. 恢复指令（新会话从这里开始）

```bash
cd /Volumes/ZHITAI/SemanticMotion
export PYTHONPATH=src
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json          # 环境仍一致？
.venv/bin/python -m semotion.cli baseline run --input tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4 --out artifacts/g0   # 基线仍 PASS？
ls artifacts/g1/*/run_summary.json 2>/dev/null && cat artifacts/g1/*/run_summary.json | head -40
```
恢复规则：先验算已有产物哈希，不重复昂贵任务（ASR/渲染有缓存）；从本文件"下一步"第 1 条继续。

## 6. 变更记录

- G0：环境预检、工程初始化、基线 PASS、契约 v1.0.0、合成素材生成器、多镜头支持、导演规则、度量引擎、CLI。
