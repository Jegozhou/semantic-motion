# PROGRESS.md — Semantic Motion 可恢复进度台账

> 唯一事实来源。任何"完成"必须对应本文件中的证据路径；未运行写 `NOT RUN`，失败写 `FAIL`，受阻写 `BLOCKED`。
> 最近更新：**v2 交付完成（2026-10-24 23:5x）**。v2 用「轮廓感知布局 + 排版化动效 + 渲染后客观质检」替换了 v1 的「人脸框 + 半透明面板」，真实素材成片全项 PASS。证据：`reports/FINAL-REPORT-V2.md`、`artifacts/v2/final_zq/`。

## v2 状态一屏

| 项 | 状态 | 证据 |
|---|---|---|
| 轮廓感知感知层（Apple Vision：人像掩码/姿态/手/烧录文字/安静区） | **DONE** | `src/semotion/perception.py`、`artifacts/p2/perception/`（125 采样，冷启 18.7 s，缓存 0.05 s，2 次确定性一致） |
| 词/字级时间对齐（DP + 声学停顿） | **DONE** | `src/semotion/align.py`；合成真值 pooled n=147 **P50 2.75 / P90 7.56 ms**，打乱对照 98.88 ms |
| 字幕（按语义段，≤2 行，词内不断） | **DONE** | `src/semotion/captions.py`；真实素材 10 cues，min display 1.32 s，mid-word 0 |
| v2 设计系统 + 程序化动效 | **DONE** | `src/semotion/{theme2,effects2,design2,render2}.py`、`docs/DESIGN.md` |
| 轮廓感知布局（scrim 沿人像逐行淡出） | **DONE** | `motion_ir.json → backdrop.profile`；5/5 元素人像重叠 **0 px** |
| 渲染后客观质检 + 修复器 | **DONE** | `src/semotion/verify2.py`、`repair.py`、`docs/QA.md`；负向对照（故意遮脸→FAIL→修复→PASS）在 `artifacts/p2/qa/` |
| 端到端命令 | **DONE** | `python -m semotion.pipeline run --video … --out …` |
| 真实素材成片 | **DONE** | `delivery/06_v2_real_footage_16x9/final.mp4`（CC BY 3.0 采访，1920×920，25 s） |
| 双跑确定性 | **DONE** | 625/625 帧 mean \|Δ\| = 0.0（`artifacts/v2/final_zq/qa/qa_metrics.json`） |
| 严格 ink 起始帧（真正的"字落在词上"） | **DONE** | 5/5 元素 `ink_onset_delta_frames = 0`（ink-only pass，严格阈值 1 帧） |
| 渲染后质检总判定 | **PASS** | 11/11 项 PASS：`artifacts/v2/final_zq/qa/qa_metrics.json` |
| Skill 安装 | **DONE（仓库内已热加载）**；`~/.claude` 与 `~/.dsh` 的写入被本会话沙箱拒绝 | `.dsh/skills/semantic-motion/`、`scripts/install_skill.py`（用户侧一条命令即可） |
| 人工真值时间验收 | **NOT RUN** | 无真人标注语料；合成真值不能冒充真人精度 |

**v2 一键复现**

```bash
cd "/Volumes/ZHITAI/Semantic Motion" && export PYTHONPATH=src
.venv/bin/python -m semotion.pipeline run --video inputs/zhengqinwen_25s.mp4 --out out/ref \
  --anchors artifacts/dl/anchors_llm.json --decisions artifacts/llm/download_zhengqinwen.decisions.json
```

---


## 0. 当前状态一屏

| 闸门 | 状态 | 证据 |
|---|---|---|
| **G0 预检与基线** | **PASS** | `ENVIRONMENT.md`、`artifacts/env/env_probe.json`、`artifacts/g0/baseline_report.json`（verdict=PASS）、`reports/gate-G0.md` |
| G1 最小语义闭环 | **PASS** | 链路已打通：`artifacts/g1/runner/gate_summary.json`（fx_a：timing/position/render PASS，n=3 探索性）、`artifacts/g1/pd_a/`（真实人脸素材：position/render PASS）；**待 ≥50 独立事件校准素材** |
| G2 四类动效 | **PASS** | `reports/gate-G2.md`；20 格动效矩阵、真实人脸 198 帧遮脸 0、数值确定性校验全 PASS |
| G3 高级稳定性 | **PASS** | `reports/gate-G3.md`；切镜安全 0 违规、人脸位移 103px/净距 192px、5 个降级场景、局部修改、4 类异常恢复、自动回归 11/11 |
| G4 完整交付 | **PASS** | `reports/gate-G4.md`、`reports/FINAL-REPORT.md`；Skill 已加载+调用；指南/许可/5 条成片齐备；最终回归 11/11 |

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

## 1.4 G1 最终验收（PASS，2026-10-24 18:2x）

| 项 | 结果 | 证据 |
|---|---|---|
| 校准集时间精度 | **n=56，P50 18.70 ms / P90 50.15 ms / max 86.15 ms（阈值 100 ms）→ PASS，可作 P90 宣称** | `artifacts/g1/final_calibration/gate_summary.json` |
| 产品集（Smart Balanced） | 5 条素材 timing/position/render 全 PASS；n=19，P50 16.83 / P90 22.04 / max 24.87（探索性） | `artifacts/g1/final_product/gate_summary.json` |
| 真实素材（公有领域真人） | fx_pd_a/b：position/render PASS，遮脸 0/8、0/6；timing **NOT RUN**（无人工真值） | `artifacts/g1/final_realface/*/metrics.json` |
| 可复现 | 同 IR 双跑 605/605 帧哈希一致，max_pixel_diff 0.0 | `final_product/gate_summary.json → determinism` |
| 数值语义（OCR 实证） | fx_a/b/c/d/e/t1/t3 全 PASS；t2 无数字元素 NOT RUN | 各目录 `number_check.json` |
| 回归测试 | **92 passed / 27 s** | `.venv/bin/python -m pytest tests/ -q` |
| 闸门报告 | `reports/gate-G1.md` | — |

## 1.5 G1 过程记录（早期实测，保留）

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

## 1.6 G2 最终验收（PASS，2026-10-24 18:5x）

| 项 | 结果 | 证据 |
|---|---|---|
| 四类动效单项 + 组合 | 20 格矩阵：18 格可测全 PASS，2 格 NO_ANCHORS（英文素材限制） | `artifacts/g2/matrix/*/matrix.json`、`reports/gate-G2.md` |
| 画幅 | 9:16（fx_a/fx_face）与 16:9（fx_b/fx_pd）均通过 | 同上 |
| 真实人脸避脸 | fx_face：人脸 125/125 帧检出；**观测遮挡 0/198 帧**、越界 0、包含性 0 | `artifacts/g2/face_stress/fx_face_zh_portrait/metrics.json` |
| 数值语义 | 确定性复算（期望 vs 半值对照）全部 PASS | 各 `number_check.json`（schema v2） |
| 两条不同素材成片 | fx_a（合成）+ fx_pd_real_16x9_a（公有领域真人）+ fx_face（真人肖像合成） | `artifacts/g2/matrix/` |
| 回归测试 | 92 passed | `.venv/bin/python -m pytest tests/ -q` |

## 1.7 G3 最终验收（PASS，2026-10-24 19:2x）

| 项 | 结果 | 证据 |
|---|---|---|
| 镜头变化 | fx_face_pan（2 镜头，9.0 s 切点）：跨切点覆层 **0**；修复"延后导致 601 ms 脱节"→ max 27.94 ms | `artifacts/g3/pan/fx_face_pan/metrics.json` |
| 人物运动 | 人脸中心位移 103.41 px；覆层固定、最小净距 192 px；遮脸 0/175 帧（static-safe，不宣称跟随） | 同上 |
| 自动降级 | 5 场景全 PASS：无人脸 / 人脸居中 / 人脸近满屏（丢弃全部覆层）/ 320×568 小画幅（像素指标 NOT RUN）/ 空计划透传 | `artifacts/g3/stability/stability.json` |
| 局部修改 | 单元素编辑：窗内 6.86、窗外 0.0549；非法修改被拒；原 IR 回退逐帧一致 | 同上 `partial_update` |
| 异常恢复 | 非法 IR / 不可写目标 / 损坏输入 / 失败续跑：4/4 | 同上 `recovery` |
| 自动回归 | **11/11 步骤 PASS，3 分 42 秒** | `artifacts/regression/summary.json` |
| 回归测试 | 92 passed | `.venv/bin/python -m pytest tests/ -q` |

## 1.10 真实下载素材实测（2026-10-24 22:3x）

| 项 | 结果 | 证据 |
|---|---|---|
| 素材 | Wikimedia Commons 郑钦文采访（CC BY 3.0），裁掉原字幕条 → 1920×920 / 25 s | `inputs/zhengqinwen_25s.mp4` |
| LLM 选词跨度 + 动效 | 4 元素（card / keyword / list×3行 / keyword），跳过 2 个 | `artifacts/llm/download_zhengqinwen.decisions.json`、`artifacts/dl/anchors_llm.json` |
| 指标 | position PASS（遮脸 0/279 帧、框外像素 0、切镜 PASS）、render PASS、**timing NOT RUN**（无人工真值） | `artifacts/dl/run/metrics.json` |
| 修复 | 高运动素材像素差分假通过 → overlay-only 验证渲染；LLM 计划切镜守卫；横屏字幕字号；卡片自适应字号；验证产物不覆盖主产物 | `reports/download-run.md` §5 |
| 演示 | before/after + 分镜 | `delivery/demo_downloaded/` |

## 1.9 G5 — LLM 导演 + Remotion 后端 + 字幕（PASS，2026-10-24 22:0x）

| 项 | 结果 | 证据 |
|---|---|---|
| Remotion 渲染后端 | 4.0.538 + 系统 Chrome；1080×1920 605/623 帧，约 50 s | `artifacts/g5/llm_a/`、`artifacts/g5/llm_face/` |
| LLM 导演 | fx_a 5 元素 / fx_face 7 元素，8 条规则校验通过 | `artifacts/llm/*.decisions.json` |
| 指标 | timing P90 23.54 / 29.08 ms；遮脸 0/133、0/150；框外像素 0；切镜 PASS | 各 `metrics.json` |
| 字幕 | 词级帧号、简体、无行尾标点、重复文案抑制（4/2 条） | `captions.json`、`remotion_props.json` |
| 可复现 | 两次渲染 max_pixel_diff 3.5e-05（显式容差 0.5）→ PASS | `artifacts/g5/determinism/` |
| 测试 | 144 passed（含 52 项 LLM/字幕） | `pytest tests/ -q` |
| 契约 | `docs/REMOTION.md`（props/决策/字幕/8 条校验） | — |
| 许可 | Remotion License（个人/≤3 员工/非营利免费；更大营利组织需购证） | `reports/licenses.md` §3.5 |

## 1.8 G4 最终交付（PASS，2026-10-24 19:4x）

| 项 | 结果 | 证据 |
|---|---|---|
| Skill 加载 | `semantic-motion` 出现在本会话技能目录（实时热加载） | 会话技能目录；`artifacts/delivery/skill_check.json` |
| Skill 调用 | `skill(name="semantic-motion")` 返回完整内容与 base dir | tool 返回内容 |
| Skill 安装位置 | 项目内 + `~/.dsh/skills` + `~/.claude/skills`，结构校验 3/3 PASS | `scripts/check_skill.py` |
| 使用指南 | `docs/USER-GUIDE.md` | — |
| 依赖与许可 | `reports/licenses.md` + `requirements.lock`(30 包) + `assets/fonts/OFL.txt` | — |
| 测试成片 | `delivery/01..05_*/`（MP4 + IR + manifest + metrics + number_check + contact sheet） | 35 个文件 |
| 最终回归 | 11/11 PASS，约 3 分 42 秒 | `artifacts/regression/summary.json` |
| 最终报告 | `reports/FINAL-REPORT.md`、`reports/gate-G4.md` | — |

## 2. 进行中（无；以下为 G1 过程记录，保留）

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

## 2.5 后续（需要外部输入或额外工作，非本轮范围）

1. 真人素材 + 人工标注（HUMAN）→ 把时间精度从合成校准升级为真人验收（L1）。
2. 留出集盲测（L2）与人工审美评审（L3）。
3. 分发合规：ffmpeg / YuNet 方案调整（L7）。
4. P2 能力：视觉跟踪（L4）、帧级局部重渲染（L5）、英文列表/卡片规则（L6）。

## 3. 下一步（历史：G1 集成清单，已完成）

1. speech-vision / motion-render 交付模块 + 单测全绿（等待回报）。
2. Lead 集成：`PYTHONPATH=src .venv/bin/python -m semotion.cli run --fixture fx_a_zh_portrait --outdir artifacts/g1/run_a --gate G1`
3. 检查成片（抽帧肉眼 + 指标）：字卡是否与语义对齐、是否避脸、是否越界。
4. 写 `reports/gate-G1.md`；只有 metrics 的 timing/position/render 全 PASS 才记 G1 PASS。
5. G2：补 number/list/card 与 16:9 素材（`fx_b_zh_landscape`），两条不同素材成片 + 组合测试。
7. G2：逐项动效矩阵（keyword/number/list/card 单独 + 组合）、9:16 与 16:9、两条不同素材（合成 fx_a + 真实 fx_pd_a）、真实素材避脸压力测试（calibration 策略扩大样本）→ `reports/gate-G2.md`。
8. G3：镜头变化/人物运动降级、局部重渲染（只重渲染一个元素）、可复现双跑、异常恢复、自动回归脚本 → `reports/gate-G3.md`。
9. G4：`~/.dsh/skills/semantic-motion/SKILL.md`、使用指南、指标与限制、许可证清单终审 → `reports/gate-G4.md`。

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
