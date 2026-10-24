# Semantic Motion 使用指南

版本：1.0（G4 交付）｜对应代码：`src/semotion/`（contracts-1.0.0）
环境预检：`ENVIRONMENT.md`｜接口契约：`docs/INTERFACES.md`｜闸门报告：`reports/gate-G0..G4.md`

---

## 1. 这是什么

给**真人口播视频**自动加"克制、语义相关"的文字动效，同时保留原画面与原声：

```
原视频 → 词级转录 → 语义锚点（含能量起音细化）→ 导演决策 → Motion IR → 栅格化+合成 → 成片 + 指标报告
```

四种 P1 动效：**关键词强调 / 数字计数 / 步骤列表 / 信息卡片**。
普通句子可以完全不加动效（这是默认行为，不是失败）。

## 2. 安装与依赖

```bash
cd /Volumes/ZHITAI/SemanticMotion
export PYTHONPATH=src

# Python 3.12 venv（已建好）
.venv/bin/python -V
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json   # 检查环境
```

必需：`ffmpeg`/`ffprobe`（**必须含 `overlay` 滤镜**；不需要 `drawtext`/`libass`）、
本地 CTranslate2 Whisper 模型快照（离线）、`assets/fonts/NotoSansSC-Regular.otf`。
可选：`tesseract`（仅旧版 OCR 校验用；现行数值校验是确定性复算，不需要 OCR）。

依赖锁定：`requirements.lock`；许可证清单：`reports/licenses.md`。

## 3. 三条使用路径

### 3.1 一键跑完（推荐先跑这条确认环境）

```bash
.venv/bin/python -m semotion.cli run --input /path/to/talk.mp4 --outdir out/talk
# 或内置合成素材：
.venv/bin/python -m semotion.cli run --fixture fx_a_zh_portrait --outdir out/demo
```

产物目录：

| 文件 | 说明 |
|---|---|
| `talk_motion.mp4` | 成片（原声保留，动效已烧录） |
| `motion_ir.json` | **可编辑工程**（Motion IR） |
| `transcript.json` / `vision.json` / `anchors.json` / `plan.json` | 中间结果，可单独复用 |
| `render_manifest.json` | 渲染命令、工具版本、哈希 |
| `metrics.json` | 时间/位置/渲染三项指标与结论 |
| `run_summary.json` | 本次运行的汇总（含每步耗时与结论） |

退出码：`0` 成功；`3` IR 非法；`4` 媒体错误；`5` 生成失败；`7` 渲染失败；`8` 度量未通过。

### 3.2 分步执行（便于排查与复用缓存）

```bash
.venv/bin/python -m semotion.cli asr run --video talk.mp4 --out work/transcript.json
.venv/bin/python -m semotion.cli vision analyze --video talk.mp4 --out work/vision.json
.venv/bin/python -m semotion.cli anchors build --transcript work/transcript.json --video talk.mp4 --out work/anchors.json
.venv/bin/python -m semotion.cli plan --anchors work/anchors.json --vision work/vision.json \
    --transcript work/transcript.json --canvas 1080x1920 --fps 25 --out work/plan.json
.venv/bin/python -m semotion.cli ir build --transcript work/transcript.json --vision work/vision.json \
    --anchors work/anchors.json --plan work/plan.json --source talk.mp4 \
    --width 1080 --height 1920 --fps 25 --out work/motion_ir.json
.venv/bin/python -m semotion.cli render --ir work/motion_ir.json --source talk.mp4 --out out/final.mp4
```

ASR/vision 结果按 `sha256` 缓存于 `artifacts/cache/`，重复运行不会重复计算。

### 3.3 修改工程后重渲染

直接编辑 `motion_ir.json` 的文本、时间窗、颜色、`layout.box`，然后：

```bash
.venv/bin/python -m semotion.cli render --ir motion_ir.json --source talk.mp4 --out out/v2.mp4
```

渲染前会校验 8 条不变量（帧号换算、锚点 ±250 ms、box 在画布内、遮脸 ≤1%、
轨道归属唯一、非设计碰撞需声明、IR 哈希、文案可溯源）。**非法修改会被拒绝并报错**，
不会产出"看起来成功但其实是坏的"视频。
若确实要改写成转录里没有的文案，在元素里加 `"edited": true`（并建议写 `edit_note`）。

## 4. 画幅与素材

* 画幅跟随源视频的分辨率：`ir build` 的 `--width/--height` 必须与源一致。
* 9:16 / 16:9 / 1:1 都支持：安全区由 `vision.safe_zones` + 人脸框推导。
* 竖屏建议源 ≥1080×1920，横屏 ≥1920×1080；更低分辨率仍能跑，但像素级位置指标会标记 `NOT RUN`
  （低分辨率压缩噪声与动效信号同量级，不能可靠判定）。
* 画面内无可靠人脸时，启用"保守主体带"（避开画面中部），这是有意的降级策略。

## 5. 指标怎么读

`metrics.json` 三块：

| 块 | 含义 | 通过标准 |
|---|---|---|
| `timing` | 渲染后动效实际出现时刻 vs **独立**语义真值 | P90 ≤ 100 ms（n<50 只作探索性报告） |
| `position` | 遮脸率 / 越界率 / 框外像素比 / 非设计碰撞 / 切镜安全 | 遮脸 0、越界 0、碰撞 ≤1% |
| `render` | 时长、帧数、音频保持 | 时长差 ≤1 帧、音频 RMS 差 ≤0.5 dB |

`fidelity_ms` 是"渲染出现时刻 vs IR 里的锚点时间"，反映渲染保真（约 1 帧）；
`error_ms` 才是"语义时间准确度"，两者不要混用。

数值动效额外用 `scripts/check_number_semantics.py` 做**确定性复算**：
按 IR 重新栅格化期望值 → 以面板透明度合成到源帧 → 与真实渲染帧比对，
并用"半值"贴片做对照，证明该检查本身能失败。

## 6. 质量与回归

```bash
.venv/bin/python -m pytest tests/ -q                 # 92 项
.venv/bin/python scripts/regression.py --gate G4     # 全链路（约 4 分钟）
.venv/bin/python scripts/gate_runner.py --gate G1 \
    --fixtures fx_t1_zh_phrase fx_t2_zh_phrase fx_t3_zh_phrase --style calibration
.venv/bin/python scripts/effect_matrix.py --fixture fx_face_zh_portrait --out out/matrix
.venv/bin/python scripts/g3_stability.py --out artifacts/g3/stability
```

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `E_RENDER_FAILED ... resolution ... != expected` | IR 画幅与源不一致 | 用源真实分辨率重新 `ir build` |
| `E_IR_INVALID: content.text ... not found` | 文案被改写但未标记 | 加 `"edited": true`，或改回可溯源的文本 |
| 数字元素没有计数动画 | 文案中没有可解析的数量词 | 检查 `content.numeric`；支持 九成/一半/三个/70%/兩倍 等 |
| 动效被跳过 | 节奏预算 / 低置信度 / 切镜冲突 / 无避脸空间 | 看 `plan.notes.skipped` 与 `provenance.skipped_decisions` 的原因 |
| 英文素材没有列表/卡片 | 规则是中文优先 | 预期行为（`NO_ANCHORS`），不会编造锚点 |
| 位置指标 `NOT RUN` | 画幅 <480 px | 输出更高分辨率素材 |

## 8. 边界（必须知道）

1. **时间精度结论来自合成校准集**（56 个独立事件，P90 ≈ 50 ms）。真人语音的人工标注验收 **未做**，
   不能外推。
2. **无留出集**：所有素材为 dev/calibration，测试非盲。
3. **不做视觉跟踪**：覆层固定，避让人脸并集；不承诺"跟着人走"。
4. **不执行生成代码**：只有声明式 IR + 白名单原语；不存在动态代码沙箱。
5. **审美是主观的**：自动指标只覆盖时间/位置/可复现/数值正确；审美需人工评审。
6. ffmpeg 为 GPL 构建，作为**外部依赖**使用，不随本 Skill 分发（见许可证清单）。
