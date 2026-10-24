# Gate G4 — 完整交付

**判定：PASS**
日期：2026-10-24（Asia/Shanghai）｜依据：`docs/semantic-motion-spec.md` §7 G4、§8、§9

---

## 1. 验收条件对照

| G4 要求 | 结果 | 证据 |
|---|---|---|
| Harness 能**加载** Skill | ✅ 本会话技能目录中出现 `semantic-motion`（实时热加载） | 本报告 §2；`artifacts/delivery/skill_check.json` |
| Harness 能**调用** Skill | ✅ 调用 `skill(name="semantic-motion")` 返回完整 SKILL 内容与 base dir | §2（对话记录 + skill tool 返回） |
| 使用指南 | ✅ `docs/USER-GUIDE.md`（安装/三条路径/改工程/画幅/指标解读/FAQ/边界） | 文件存在且被技能引用 |
| 所有可行指标 + 剩余限制 | ✅ G0–G3 报告 + 本报告 §5 汇总表；限制逐条列出 | `reports/gate-G0..G3.md` |
| 依赖与许可清单 | ✅ `reports/licenses.md` + `requirements.lock` + `assets/fonts/OFL.txt` | 三份文件齐备 |
| 测试视频 | ✅ `delivery/` 5 条成片（含 IR、渲染清单、指标、数值校验、联系表） | `delivery/*/` |
| 交付报告 | ✅ `reports/FINAL-REPORT.md` | 本目录 |

## 2. Skill 加载与调用（真实证据）

安装位置（三处，均已校验 frontmatter + 章节完整性）：

```
/Volumes/ZHITAI/SemanticMotion/.dsh/skills/semantic-motion/SKILL.md   (项目内，规格 §8 指定路径)
~/.dsh/skills/semantic-motion/SKILL.md                                (DSH 包内常量 .dsh/skills)
~/.claude/skills/semantic-motion/SKILL.md                             (本会话实际加载的根目录)
```

* **加载证据**：创建后本会话的可用技能目录被刷新，`semantic-motion` 出现在其中（描述与触发词为本文所写）。
* **调用证据**：`skill(name="semantic-motion")` 返回 `skill_content`，base dir 解析到项目内 SKILL.md。
* **结构校验**：`scripts/check_skill.py` → PASS（3/3 副本、frontmatter、必需章节、引用的 8 个仓库路径均存在）。

> 边界：Harness 的热加载把新技能加入了本会话目录（已观察），但"下一会话自然发现"不做绝对承诺；本项目只声明**当前已加载且可调用**这一可复核事实。

## 3. 交付物清单

### 3.1 成片（`delivery/`）

| 目录 | 画幅/素材 | 内容 | 关键结论 |
|---|---|---|---|
| `01_g1_keyword_number_9x16` | 1080×1920 合成口播 | 关键词 + 数字 | timing P90 22.0 ms（全项目 G1 产品集口径） |
| `02_g2_list_card_16x9` | 1920×1080 合成口播 | 列表 + 卡片 + 数字 | 16:9 布局通过，数值 `70%` 复算通过 |
| `03_g2_real_face_avoidance` | 1080×1920 真人照片合成 | 四类动效组合（13 元素） | 遮脸 **0/198 帧**，框外像素 0 |
| `04_g3_camera_cut_safety` | 1080×1920 双镜头 pan/zoom | 切镜 + 移动人脸 | 跨切点覆层 **0**，max timing 27.94 ms |
| `05_g1_real_footage_16x9` | 1280×720 公有领域真人影像 | 关键词 + 数字 | 遮脸 0/8，位置 PASS |

每个目录含：成片 MP4、`motion_ir.json`、`render_manifest.json`、`metrics.json`、`number_check.json`（有数字时）、`contact_sheet.png`、`README.md`。

### 3.2 工程与文档

| 文件 | 作用 |
|---|---|
| `docs/semantic-motion-spec.md` | 需求与验收基准（未被修改） |
| `docs/INTERFACES.md` | contracts-1.0.0：7 个 JSON schema + 模块签名 + 错误码 + 8 条 IR 不变量 |
| `docs/USER-GUIDE.md` | 使用指南 |
| `ENVIRONMENT.md` | 环境预检与预算（G0） |
| `PROGRESS.md` | 闸门状态、证据路径、恢复命令、已知阻碍 |
| `reports/gate-G0..G4.md`、`reports/FINAL-REPORT.md`、`reports/licenses.md` | 证据报告 |
| `tests/fixtures/manifest.json` | 13 条素材（9 条 SYNTH_KNOWN + 2 条真实影像 + 2 条真人脸合成），含许可、真值等级、可用/不可用范围 |
| `requirements.lock` | 30 个包的锁定版本 |

### 3.3 代码

`src/semotion/`：`media / asr / vision / anchors / semantic / ir / design / raster / render / effects / verify / baseline / fixtures / cli`，
测试 `tests/test_*.py` 共 **92 项**，脚本 `scripts/`：`env_probe / gate_runner / effect_matrix / g3_stability / check_number_semantics / regression / check_skill / make_pd_fixture`。

## 4. 复现（一条命令验证交付是否仍然成立）

```bash
cd /Volumes/ZHITAI/SemanticMotion && export PYTHONPATH=src
.venv/bin/python scripts/regression.py --gate G4     # 11 步，约 4 分钟
```

最近一次完整回归：**11/11 PASS**（`artifacts/regression/summary.json`，含每步 argv、退出码、耗时、输出尾部）。

## 5. 指标总表（全部来自真实运行）

| 指标 | 数值 | 结论 | 证据 |
|---|---|---|---|
| 时间精度（校准集，n=56） | P50 18.70 / P90 50.15 / max 86.15 ms | PASS（P90 ≤100 ms） | `artifacts/g1/final_calibration/gate_summary.json` |
| 时间精度（产品集，n=19） | P50 16.83 / P90 22.04 / max 24.87 ms | PASS（探索性，n<50） | `artifacts/g1/final_product/gate_summary.json` |
| 渲染保真（onset vs IR 锚点） | 中位 ≈1 帧（40 ms @25fps） | 达标 | `metrics.json → timing.fidelity` |
| 遮脸率（真实人脸） | 0 / 198 帧（G2 压力）、0 / 175 帧（G3 运动） | PASS（目标 0） | `artifacts/g2/face_stress/...`、`artifacts/g3/pan/...` |
| 越界率 | 0（全部素材） | PASS（目标 0） | 各 `metrics.json` |
| 框外像素比 | 0–0.008 | PASS（≤0.03） | `position.raster_containment` |
| 非设计碰撞 | 0 | PASS（≤1%） | `position.collision` |
| 切镜跨覆 | 0 | PASS | `position.cut_safety` |
| 可复现（同 IR 双跑） | 605/605 与 534/534 帧哈希一致，max_pixel_diff 0.0 | PASS | `gate_summary.json → determinism` |
| 数值语义 | 全部数字元素确定性复算 PASS（含半值对照可失败） | PASS | `number_check.json` |
| 回归测试 | 92 passed | PASS | pytest |

## 6. 未完成与剩余限制（不隐瞒）

| ID | 项 | 状态 | 说明 / 需要的资源 |
|---|---|---|---|
| L1 | 真人语音时间精度（HUMAN 真值） | **NOT RUN** | 需用户提供 1–3 条真人短口播 + 人工标注的关键词时间；当前结论只覆盖合成校准 |
| L2 | 留出集（holdout） | **无** | 所有素材 dev/calibration，测试非盲；如需盲测须新增未参与调参的素材并在 manifest 标 holdout |
| L3 | 审美验收 | **主观，未做独立评审** | 自动指标只覆盖时间/位置/可复现/数值；审美需人工评审 |
| L4 | 人物跟踪 / 跟随布局 | **未实现（P2）** | 当前为 static-safe；不宣称跟随稳定 |
| L5 | 帧级局部重渲染 | **未实现（P2）** | 现为整片重渲染（24 s ≈3 s）；已实现"改一个元素→重渲染并验证窗外不变" |
| L6 | 英文列表/卡片语义 | 部分 | 英文仅有数字/关键词规则，列表与结论卡需中文；素材缺锚点时标 `NO_ANCHORS` |
| L7 | ffmpeg GPL / YuNet 条款 | 待处理 | 闭源分发前必须替换或取得合规路径（`reports/licenses.md` §4） |
| L8 | 低分辨率（<480 px）像素指标 | NOT RUN | 压缩噪声与动效同量级，如实标注而非给假 PASS |
| L9 | 自定义代码/动效沙箱 | 未实现（按规格禁用） | 只允许声明式 IR + 白名单原语 |

## 7. 结论

**G4 PASS**：Skill 已安装并被 Harness 加载与调用；使用指南、契约、环境、进度、四份闸门报告、最终报告、许可证清单与 5 条测试成片齐备；一条命令可在约 4 分钟内复现全部证据链。
项目**不宣称**"全部功能完成"：L1–L9 已逐条列出，其中 L1（真人语音验收）与 L7（许可证合规）是后续需要外部输入或额外工作的事项。
