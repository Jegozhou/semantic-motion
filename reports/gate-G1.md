# Gate G1 — 最小语义闭环

**判定：PASS（工程闭环成立）**，附明确范围限制。
日期：2026-10-24（Asia/Shanghai）｜依据：`docs/semantic-motion-spec.md` §2/§3.1/§3.5/§6/§7

> 一句话：真实输入 → 转录/对齐（词级）→ 语义锚点 → 关键词/数字/列表/卡片动效 → 静态安全布局 → MP4 + Motion IR + 日志 + 可量测指标，全链路真实跑通；时间指标在 56 个独立事件上 **P90 = 50.15 ms（阈值 100 ms）**。
> **不是**"全部完成"：G2/G3/G4 仍未运行（见 `PROGRESS.md`）。

---

## 1. 交付链路（全部为本机实际运行产物）

```
输入视频 → ffmpeg 解封装 → faster-whisper 词级转录(semotion.transcript/1)
        → YuNet 人脸/镜头/安全区(semotion.vision/1)
        → 语义锚点 + 能量起音细化(semotion.anchors/1)
        → Smart Balanced 导演决策(semotion.plan/1)
        → Motion IR(semotion.motion-ir/1，8 条不变量校验)
        → Pillow 逐帧 RGBA 栅格化 + ffmpeg overlay 合成
        → MP4 + render_manifest(semotion.render/1) + metrics(semotion.metrics/1)
```

命令（单条复现）：

```bash
cd /Volumes/ZHITAI/SemanticMotion && export PYTHONPATH=src
.venv/bin/python -m semotion.cli run --fixture fx_a_zh_portrait --outdir artifacts/g1/demo --gate G1
# 全套闸门证据：
.venv/bin/python scripts/gate_runner.py --gate G1 --out artifacts/g1/final_calibration \
    --fixtures fx_t1_zh_phrase fx_t2_zh_phrase fx_t3_zh_phrase --style calibration
.venv/bin/python scripts/gate_runner.py --gate G1 --out artifacts/g1/final_product \
    --fixtures fx_a_zh_portrait fx_b_zh_landscape fx_c_zh_cuts fx_d_zh_portrait fx_e_zh_landscape --determinism
.venv/bin/python scripts/gate_runner.py --gate G1 --out artifacts/g1/final_realface \
    --fixtures fx_pd_real_16x9_a fx_pd_real_16x9_b --language en
```

## 2. 成片与工程文件（真实路径）

| 素材 | 成片 | Motion IR | 报告 |
|---|---|---|---|
| fx_a（9:16 合成，24.2 s） | `artifacts/g1/final_product/fx_a_zh_portrait/fx_a_zh_portrait_motion.mp4` | `.../motion_ir.json` | `.../metrics.json` |
| fx_b（16:9 合成，26.18 s） | `artifacts/g1/final_product/fx_b_zh_landscape/fx_b_zh_landscape_motion.mp4` | 同上 | 同上 |
| fx_c（9:16 三镜头，20.08 s） | `artifacts/g1/final_product/fx_c_zh_cuts/fx_c_zh_cuts_motion.mp4` | 同上 | 同上 |
| fx_d（9:16，第二音色） | `artifacts/g1/final_product/fx_d_zh_portrait/fx_d_zh_portrait_motion.mp4` | 同上 | 同上 |
| fx_e（16:9，百分比/结论） | `artifacts/g1/final_product/fx_e_zh_landscape/fx_e_zh_landscape_motion.mp4` | 同上 | 同上 |
| **真实素材** fx_pd_a（1280×720 公有领域真人） | `artifacts/g1/final_realface/fx_pd_real_16x9_a/fx_pd_real_16x9_a_motion.mp4` | 同上 | 同上 |
| **真实素材** fx_pd_b | `artifacts/g1/final_realface/fx_pd_real_16x9_b/fx_pd_real_16x9_b_motion.mp4` | 同上 | 同上 |
| 校准集 fx_t1–t3（各 20 单元） | `artifacts/g1/final_calibration/fx_t*/fx_t*_motion.mp4` | 同上 | 同上 |

抽帧复核：`artifacts/g1/final_product/frames/*.png`、`artifacts/g1/final_realface/frames/*.png`（人工目检用；本报告 §5 记录了目检结论）。

## 3. 指标（全部由 `verify.py` 从实际渲染像素测得）

### 3.1 时间精度（核心验收项）

度量方法：**渲染后**逐帧解码，在元素自己的 layout.box 内计算 |Δ像素|，取越过噪声门限的首帧为该元素的实际出现时刻；目标是 `SYNTH_KNOWN` 单元起点（逐单元 TTS 拼接、ffprobe 实测，**不经 ASR**）。误差 = |实际 onset − 独立真值|。

| 数据集 | 事件数 n | P50 | P90 | Max | 阈值 | 判定 | 证据 |
|---|---|---|---|---|---|---|---|
| **校准集 fx_t1–t3（calibration 策略）** | **56** | **18.70 ms** | **50.15 ms** | 86.15 ms | ≤100 ms | **PASS（可达 P90 宣称）** | `artifacts/g1/final_calibration/gate_summary.json` |
| 产品集 fx_a–e（Smart Balanced） | 19 | 16.83 ms | 22.04 ms | 24.87 ms | ≤100 ms | PASS（**探索性**，n<50） | `artifacts/g1/final_product/gate_summary.json` |
| 真实素材 fx_pd_a/b | — | — | — | — | — | **NOT RUN**（无人工词级真值） | `artifacts/g1/final_realface/*/metrics.json` |

逐条证据在 `metrics.json → timing.rows`（含 `rendered_onset_s / target_onset_s / error_ms / fidelity_ms`）。
帧量化：25 fps ⇒ 1 帧 = 40 ms；P90 50 ms ≈ 1.25 帧。

补充：ASR 词起点相对真值的原始偏差（能量起音细化前）为 132/215/500 ms（P50/P90/max，n=103），细化后 8/19/38 ms——该对比由 `tests/eval_timing.py` 生成（`artifacts/anchors/timing_report.json`），说明**时间准确性的提升来自音频起音细化，而不是 ASR 自证**。

### 3.2 位置（避脸/越界/碰撞/包含性）

| 素材 | 计划遮脸 | 观测遮脸（渲染帧差异 vs 人脸框） | 越界 | 框外像素比 | 碰撞 | 人脸检出 |
|---|---|---|---|---|---|---|
| fx_a | 0 | 0 / 84 | 0 | 0.0000 | 0 | 0（合成素材无真人脸） |
| fx_b | 0 | 0 / 133 | 0 | 0.0013 | 0 | 0 |
| fx_c | 0 | 0 / 113 | 0 | 0.0020 | 0 | 0 |
| fx_d | 0 | 0 / 133 | 0 | 0.0011 | 0 | 0 |
| fx_e | 0 | 0 / 89 | 0 | 0.0080 | 0 | 0 |
| **fx_pd_a（真实人脸）** | 0 | **0 / 8** | 0 | 0.0000 | 0 | 120 帧检出（union [491,87,280,392]） |
| **fx_pd_b（真实人脸）** | 0 | **0 / 6** | 0 | 0.0000 | 0 | 120 帧检出 |

阈值：遮脸 = 覆盖人脸安全框面积 >1% 即计为一次；目标 0。越界目标 0。碰撞 = 重叠超过较小元素 5% 且未声明 `collision_ok`，目标 ≤1%。
说明：真实素材的采样帧数偏少（8/6），是**证据强度不足**而非"通过"；G2 会用 calibration 策略在真实素材上做压力测试并补齐样本。

### 3.3 渲染与可复现

| 检查 | 结果 | 证据 |
|---|---|---|
| 时长/帧数一致性 | fx_b delta 0.017 s（≤1 帧），其余 0.000–0.001 s | 各 `metrics.json → render` |
| 音频保持 | 解码 PCM 时长差 ≤0.05 s、RMS 差 ≤0.5 dB（AAC 重编码不允许采样级一致） | 同上 |
| 同 IR 双跑逐帧一致 | **605/605 帧哈希完全一致，max_pixel_diff 0.0** | `artifacts/g1/final_product/gate_summary.json → determinism` |
| IR 校验 | 全部元素通过 8 条不变量；hold ≤0.8 s；onset ±250 ms | `motion_ir.json` 由 `ir.validate()` 放行后才会渲染 |

### 3.4 数值语义准确性（OCR 实证）

方法：取数字元素**最后一帧**，按 IR box 裁切，tesseract OCR（百分比用数字白名单，带中文单位的用 chi_sim+eng），与 IR 中冻结的 `content.numeric.final` 比对。脚本：`scripts/check_number_semantics.py`。

| 素材 | 结果 | 校验值（示例） |
|---|---|---|
| fx_a | PASS | 九成 → `90%` |
| fx_b | PASS | 百分之七十 → `70%` |
| fx_c | PASS | 10% |
| fx_d | PASS | 三个 → `3`（`3秒` OCR 读作种/秒属字符混淆，数字一致） |
| fx_e | PASS | 87% / 2秒 / 50%（一半） |
| fx_t1 | PASS | 90% / 6次 / 30% |
| fx_t3 | PASS | 4 |
| fx_t2 | NOT RUN（该片无数字元素） | — |

证据：各目录下 `number_check.json`。
本轮修复的真实缺陷：`一半` 曾被解析为 `1半`（语义错误）、繁体 `兩倍` 未被识别——已修（`effects/number.py` 增加繁简折叠与成语量词映射），修后 OCR 全部一致。

## 4. 覆盖与真值声明（诚实边界）

| 维度 | 状态 | 说明 |
|---|---|---|
| 中文普通话语义闭环 | ✅ 已验证 | 合成 TTS（Tingting/Meijia），126 个语音单元 |
| 时间精度 | ✅ 校准集 56 事件 P90 50.15 ms | 真值 = `SYNTH_KNOWN`（逐单元 TTS 拼接实测），**非真人语音** |
| 真人语音时间精度 | **NOT RUN** | 无人工标注；`HUMAN` 真值缺失，不能以机器估算自证 |
| 真人素材链路 | ✅ 跑通 | 公有领域真人视频（英文），避脸 0 覆盖；语言非中文，时间无真值 |
| 繁体/异体字 | ✅ 已处理 | ASR 偶发繁体，展示层统一转简体（opencc） |
| 留出集 | **无** | 全部 dev/calibration，测试**非盲**（manifest 已声明 `holdout: []`） |
| 审美验收 | **NOT RUN（主观）** | 仅内部目检，不宣称独立审美通过 |
| 自定义代码执行 | 未实现（按规格禁用） | 只用声明式 IR + 白名单原语 |

## 5. 人工目检记录（Lead，实际查看抽帧）

- `fx_a` 07.96 s：数字「九成」计数到 **90%**，金色数字、深色圆角卡，位于人物躯干前，不遮头部轮廓；08.55 s 终值仍为 90%，退出淡出正常。
- `fx_a` 03.44 s：关键词「痛点」96 px + 金色强调线，位于下部安全区，克制、可读。
- `fx_b` 04.00 s：列表卡片位于下三分之一，显示「① 字幕出现的时间」，序号圆点 + 简体文本，未压头部。
- `fx_d` 06.48 s：浅色背景下深色数字卡「2个」对比度充足。
- **fx_pd**（真实人脸）：数字/关键词卡片位于人脸右侧安全列，与人脸框无交集；成片在 1280×720 下清晰可读。
- 发现并已修复的问题：列表曾显示「③ 第三」缺少内容 → 现从转录词流补齐「① 字幕出现的时间 / ② 动效占用的画面 / ③ 人物被遮挡的面积」；繁体「所以我們先對齊」→ 现为简体「所以我们先对齐」。

## 6. 性能与资源（实测，非估算）

| 操作 | 耗时 |
|---|---|
| 单条 24 s 1080×1920 端到端（含 ASR 缓存命中） | ≈ 4 s |
| 1080p 24 s 渲染（栅格化+编码） | ≈ 3 s |
| 校准集 3 条（各 40 s，20 单元，calibration 策略） | 29 s |
| ASR 冷启动（24 s 音频，medium int8 CPU） | 12 s；缓存命中 <0.1 s |
| 全量回归测试 | 92 tests / 27 s |

## 7. 已知缺陷与限制（类型化）

| ID | 问题 | 影响 | 处置 |
|---|---|---|---|
| L1 | 无 HUMAN 真值 / 无 holdout | 时间结论仅限合成校准 | 待用户提供真人标注；报告中标注 NOT RUN |
| L2 | 合成素材无人脸 | 避脸只能用 2 条真实素材验证 | G2 用 calibration 策略扩大样本 |
| L3 | ASR 偶发字符错误（口→抠、改→该等，7/60 单元） | 前缀匹配少算事件 | 已如实计数；不修补数据 |
| L4 | 繁简表/OpenCC 覆盖有限 | 罕见字仍可能漏匹配 | 已装 opencc，逐步替换手写表 |
| L5 | keyword >2 行以省略号收尾 | 极长关键词显示不全 | 记录 warning；G2 复核 |
| L6 | 列表单独序号（"第一"）无内容句时会重复 | 语义重复 | 已用转录词流补齐；无内容时保留原文 |
| L7 | ffmpeg 为 GPL 构建 | 分发限制 | 作为外部依赖，不分发（见 `reports/licenses.md`） |

## 8. 复现与校验命令

```bash
cd /Volumes/ZHITAI/SemanticMotion && export PYTHONPATH=src
.venv/bin/python -m pytest tests/ -q                              # 92 passed
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json
.venv/bin/python -m semotion.cli baseline run --input tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4 --out artifacts/g0
.venv/bin/python scripts/gate_runner.py --gate G1 --out artifacts/g1/final_calibration --fixtures fx_t1_zh_phrase fx_t2_zh_phrase fx_t3_zh_phrase --style calibration
.venv/bin/python scripts/check_number_semantics.py artifacts/g1/final_product/fx_b_zh_landscape/motion_ir.json artifacts/g1/final_product/fx_b_zh_landscape/fx_b_zh_landscape_motion.mp4 /tmp/num.json
```

## 9. 结论

- **G1 PASS**：真实输入到成片/IR/日志/指标的闭环成立，时间与位置指标达标且有反向验证（故意违规会 FAIL）。
- **明确未完成**：G2（四类动效的逐项与组合验收、9:16/16:9 双画幅、两条不同素材）、G3（镜头/运动、局部重渲染、异常恢复、自动回归）、G4（Skill 入口、指南、许可清单终审）。
- 不得把本报告读作"项目完成"；也不得把 `SYNTH_KNOWN` 结论外推为真人语音验证。
