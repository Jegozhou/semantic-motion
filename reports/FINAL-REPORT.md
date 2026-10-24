# Semantic Motion — 最终交付报告

日期：2026-10-24（Asia/Shanghai）｜依据：`docs/semantic-motion-spec.md` §9
项目路径：`/Volumes/ZHITAI/SemanticMotion`

---

## 1. 一句话结论

**G0–G4 五个闸门全部 PASS**，每个闸门都有真实可播放成片、可编辑 Motion IR、运行日志与可复查指标。
时间精度在 **56 个独立事件**上达到 **P90 = 50.15 ms**（阈值 100 ms）；真实人脸上遮脸率 **0/198 帧**；同 IR 双跑 **逐帧一致**；全链路自动回归 **11/11 通过（约 4 分钟）**。

**未完成项已逐条列在 §6**，其中真人口播时间验收（HUMAN 真值）与许可证合规（ffmpeg GPL / YuNet）需要外部输入或额外工作，本次**没有**用合成数据或自评冒充。

## 2. 闸门结果

| 闸门 | 判定 | 核心证据 | 报告 |
|---|---|---|---|
| G0 预检与基线 | **PASS** | 环境探测 JSON；透传 605/605 帧哈希一致、音频 PCM 哈希一致；重编码 605 帧 max 像素差 0.0118 | `reports/gate-G0.md` |
| G1 最小语义闭环 | **PASS** | 校准集 n=56：P50 18.70 / P90 50.15 / max 86.15 ms；产品集 5 素材全 PASS ；真实影像位置 PASS | `reports/gate-G1.md` |
| G2 四种核心动效 | **PASS** | 20 格动效矩阵（keyword/number/list/card/combo × 4 素材）；真实人脸遮脸 0/198；数值确定性复算全 PASS | `reports/gate-G2.md` |
| G3 高级稳定性 | **PASS** | 切镜跨覆 0；人脸位移 103.41 px / 净距 192 px；5 个降级场景；局部修改窗外差 0.0549；4 类异常恢复；自动回归 11/11 | `reports/gate-G3.md` |
| G4 完整交付 | **PASS** | Skill 已加载并可调用；指南/契约/许可/5 条成片齐备 | `reports/gate-G4.md` |

## 3. 已实现能力（对应规格）

| 规格条目 | 实现 | 证据 |
|---|---|---|
| §3.1 中文口播转录、词级对齐 | faster-whisper（CPU int8，本地快照，离线）；词级时间 + char span | `artifacts/asr/*.transcript.json` |
| §3.1 不凭空猜时间戳 | 锚点时间只取 ASR 词/段区间 + 音频能量起音细化（±250 ms 窗） | `anchors.json → refine` |
| §3.1 低置信度不冒充精确 | conf<0.6 不驱动精确动效，记 `low_confidence` | `plan.notes.skipped` |
| §3.2 人脸/镜头/安全区 | YuNet（≥5 Hz）+ 镜头切换检测 + 归一化安全区 | `vision.json` |
| §3.2 人物优先、退化保守 | 无脸→保守主体带；有脸→人脸框切块；无空间→丢弃覆层 | `reports/gate-G3.md` §3 |
| §3.3 智能均衡导演 | 规则引擎（no LLM）：角色优先、节奏预算、覆盖率上限、普通句零动效 | `src/semotion/semantic.py` |
| §3.4 四类 P1 动效 | keyword / number / list / card，9:16 与 16:9 均通过 | `artifacts/g2/matrix/` |
| §3.5 Motion IR + 可复现 | schema 化 IR、8 条不变量、哈希封印、诊断即报错 | `docs/INTERFACES.md` §3.5 |
| §3.5 解帧比较而非字节比较 | 逐帧解码哈希 + 像素容差 | `verify.determinism` |
| §3.6 动态代码安全 | 不执行生成代码；只消费校验过的 IR 与白名单原语 | 无 `eval/exec`；渲染命令固定 |
| §4 干净室 | 只依据规格文档与公开通用知识实现 | `reports/licenses.md` §5 |
| §5 真值分级 | `SYNTH_KNOWN` / `MACHINE_WEAK` / (`HUMAN` 缺失→NOT RUN) | `tests/fixtures/manifest.json` |
| §6 可计算指标 | 时间/位置/渲染/数值四类，全部落 JSON + 结论 | 各 `metrics.json` |
| §8 可续跑 | `PROGRESS.md` + 缓存（ASR/vision/anchors）+ 恢复命令 | `PROGRESS.md` §5 |

## 4. 关键数字（可核验）

```
时间精度（SYNTH_KNOWN 校准集，独立于 ASR）  n=56  P50 18.70 ms  P90 50.15 ms  max 86.15 ms   PASS
时间精度（产品集 Smart Balanced）           n=19  P50 16.83 ms  P90 22.04 ms  max 24.87 ms   探索性
ASR 原始偏差 → 起音细化后                    n=103  132/215/500 ms → 8/19/38 ms
遮脸率（真实人脸，G2 压力测试）             0 / 198 帧                                        PASS
遮脸率（真实人脸，G3 镜头运动）             0 / 175 帧                                        PASS
越界率 / 框外像素比                         0 / 0–0.008                                       PASS
非设计碰撞                                  0                                                 PASS
切镜跨覆                                    0（2 镜头、9.0 s 切点）                            PASS
可复现（同 IR 双跑）                        605/605 与 534/534 帧哈希一致，max_diff 0.0       PASS
数值语义                                    全部数字元素确定性复算 PASS（半值对照可失败）        PASS
自动回归                                    11/11 步 PASS，约 3 分 42 秒                        PASS
单元/集成测试                               92 passed                                          PASS
```

## 5. 交付物位置

| 内容 | 路径 |
|---|---|
| 成片 + IR + 指标（5 条） | `delivery/01..05_*/`（含 contact sheet） |
| Skill 入口 | `.dsh/skills/semantic-motion/SKILL.md`（另装 `~/.dsh/skills/`、`~/.claude/skills/`） |
| 使用指南 | `docs/USER-GUIDE.md` |
| 接口契约 | `docs/INTERFACES.md` |
| 环境与预算 | `ENVIRONMENT.md`、`artifacts/env/env_probe.json` |
| 进度与恢复 | `PROGRESS.md` |
| 闸门报告 | `reports/gate-G0.md` … `reports/gate-G4.md` |
| 许可清单 | `reports/licenses.md`、`requirements.lock`、`assets/fonts/OFL.txt` |
| 回归证据 | `artifacts/regression/summary.json` |
| 素材清单 | `tests/fixtures/manifest.json`（13 条，含许可与真值等级） |

## 6. 未完成 / 未验证（如实）

| ID | 项 | 状态 | 缺什么 |
|---|---|---|---|
| L1 | 真人语音时间精度 | **NOT RUN** | 用户提供 1–3 条真人短口播 + 人工关键词时间（`HUMAN`） |
| L2 | 留出集盲测 | **无** | 未参与调参的素材 + 独立标注 |
| L3 | 审美验收 | **主观，未独立评审** | 人工评审 |
| L4 | 人物跟踪 / 跟随布局 | 未实现（P2） | 需要跟踪算法与稳定性验收 |
| L5 | 帧级局部重渲染 | 未实现（P2） | 现为整片重渲染，24 s ≈3 s |
| L6 | 英文列表/结论卡 | 部分 | 英文语义规则需扩展（现有数字/关键词） |
| L7 | 许可证合规分发 | 未完成 | 替换/调整 ffmpeg 与 YuNet 方案 |
| L8 | <480 px 画幅像素指标 | NOT RUN | 更高分辨率素材，或更鲁棒的低分辨率度量 |
| L9 | 自定义动效代码沙箱 | 未实现（按规格禁用） | 真隔离沙箱才可开放 |

## 7. 复现与恢复

```bash
cd /Volumes/ZHITAI/SemanticMotion && export PYTHONPATH=src
.venv/bin/python scripts/regression.py --gate G4          # 全链路证据（约 4 分钟）
.venv/bin/python -m pytest tests/ -q                      # 92 tests
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json
.venv/bin/python -m semotion.cli run --fixture fx_a_zh_portrait --outdir out/demo
```

中断恢复：先读 `PROGRESS.md` → 核对已有产物哈希 → 从"下一步"继续；ASR/vision/anchors 有按 sha256 的缓存，不重复昂贵计算。

## 8. 需要的授权或资源（若要继续推进）

1. **真人素材 + 人工标注**（1–3 条，中文，30–60 s），用于把时间精度从"合成校准"升级为"真人验收"。
2. **分发形态决定**：是否随产品分发 ffmpeg？若否，保持"用户自备 + 版本探测"即可；若是，需换 LGPL/videotoolbox 路径。
3. **是否要视觉跟踪（P2）**：需要接受"跟踪失败时回退静态布局"的额外复杂度。
4. **审美评审**：自动指标无法替代人工观感判断。

## 9. 诚实声明

* 本报告中每个 PASS 都对应磁盘上的文件、命令与数值；FAIL / NOT RUN / BLOCKED 均按实际状态标注。
* 未把"代码已写""计划已定""合成数据"当作"功能已验收"。
* 合成 TTS 校准集**不能**代表真人语音精度；本报告没有把 P90=50 ms 外推为真人结论。
* 干净室：实现只依据 `docs/semantic-motion-spec.md` 与公开通用技术知识，未参考任何竞品源码或内部资料。
