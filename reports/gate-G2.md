# Gate G2 — 四种核心动效

**判定：PASS**（四类动效单项 + 组合、9:16 与 16:9、真实人脸避脸、数值语义准确性均有实测证据）
日期：2026-10-24（Asia/Shanghai）｜依据：`docs/semantic-motion-spec.md` §3.4 / §6 / §7

---

## 1. 验收条件对照（逐条对应规格）

| G2 要求 | 结果 | 证据 |
|---|---|---|
| 关键词动效（字符/短语锚定、入场退出） | ✅ 单项通过（4 条素材） | `artifacts/g2/matrix/*/keyword/matrix.json` |
| 数字动效（数值语义准确、计数、准确触发） | ✅ 单项通过 + **数值经确定性复算校验** | `.../number/number_check.json` |
| 步骤列表（多锚点顺序揭示、退场） | ✅ 单项通过（中文素材） | `.../list/metrics.json` |
| 信息卡片（留白、避脸、缩放碰撞） | ✅ 单项通过（中文素材） | `.../card/metrics.json` |
| 四类**组合**测试 | ✅ 4/4 素材 combo 通过（元素 10–14 个） | `.../combo/metrics.json` |
| 9:16 支持 | ✅ fx_a / fx_face（1080×1920） | 同上 |
| 16:9 支持 | ✅ fx_b（1920×1080）/ fx_pd（1280×720） | 同上 |
| 避免遮脸 | ✅ 真实人脸素材 198 采样帧 **0 覆盖率** | §4 |
| 两条不同素材真实成片 | ✅ 合成口播 fx_a + **公有领域真人素材** fx_pd_a（另有真人脸合成素材 fx_face） | §2 |

**未被本闸门覆盖（不冒充）**：英文素材无中文列表/卡片锚点（`NO_ANCHORS`，如实 NOT RUN）；真人语音时间精度缺人工真值（NOT RUN）；审美属主观评估（仅内部目检）。

## 2. 成片清单（真实文件）

| 素材 | 画幅 | 族 | 成片 |
|---|---|---|---|
| fx_a_zh_portrait（合成口播） | 9:16 1080×1920 | keyword / number / list / card / combo | `artifacts/g2/matrix/fx_a/<effect>/fx_a_zh_portrait_<effect>.mp4` |
| fx_b_zh_landscape（合成口播） | 16:9 1920×1080 | 同上 | `artifacts/g2/matrix/fx_b/<effect>/fx_b_zh_landscape_<effect>.mp4` |
| **fx_face_zh_portrait**（真人照片合成 + 中文口播） | 9:16 1080×1920 | 同上 | `artifacts/g2/matrix/fx_face/<effect>/fx_face_zh_portrait_<effect>.mp4` |
| **fx_pd_real_16x9_a**（公有领域真人影像，英文） | 16:9 1280×720 | keyword / number / combo（list、card 无锚点） | `artifacts/g2/matrix/fx_pd/<effect>/...mp4` |
| 真实人脸压力测试 | 9:16 | calibration（13 元素覆盖全片） | `artifacts/g2/face_stress/fx_face_zh_portrait/fx_face_zh_portrait_motion.mp4` |

素材来源与许可：`tests/fixtures/manifest.json`。
`fx_face_zh_portrait` = 程序生成背景 + **公有领域/CC 授权真人肖像照片**（Pete Souza 官方肖像，CC BY 3.0，Wikimedia Commons）合成，**仅为测试夹具**，不得当作真实拍摄素材对外使用。
`fx_pd_real_*` = Wikimedia Commons「Weekly Address (2009-01-24)」，美国政府作品 **公有领域**。

## 3. 四类动效矩阵（逐格实测）

命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/effect_matrix.py --fixture fx_a_zh_portrait --out artifacts/g2/matrix/fx_a
# 同样的 --fixture fx_b_zh_landscape / fx_face_zh_portrait / fx_pd_real_16x9_a
```

| 素材 | keyword | number | list | card | combo |
|---|---|---|---|---|---|
| fx_a（9:16） | PASS（4 元素） | PASS（2）+数值 PASS | PASS（1） | PASS（1） | PASS（10）+数值 PASS |
| fx_b（16:9） | PASS（5） | PASS（2）+数值 PASS（时间 NOT RUN 见注1） | PASS（1） | PASS（2） | PASS（11）+数值 PASS |
| fx_face（9:16 真人脸） | PASS（6） | PASS（3）+数值 PASS | PASS（1） | PASS（5） | PASS（13）+数值 PASS |
| fx_pd（16:9 真人影像） | PASS（13，时间 NOT RUN 注2） | PASS（1）+数值 PASS | **NO_ANCHORS**（注3） | **NO_ANCHORS**（注3） | PASS（14）+数值 PASS |

注 1：fx_b 的 number 锚点（"三个"）不在语音单元开头，无独立真值可对照 → timing `NOT RUN`（不是失败）。
注 2：fx_pd 为英文素材，无人工词级真值 → timing 一律 `NOT RUN`。
注 3：该英文片段不含中文列表/秩序词与结论句，规则层不产出对应锚点；**这是素材限制**，已如实标记 `NO_ANCHORS`，未用伪造锚点填空。

每格指标（完整 JSON：`<cell>/metrics.json`）：

| 指标 | 结果 |
|---|---|
| timing（元素 onset vs 独立单元真值） | 18/20 格可测；P50 2.7–24.2 ms，P90 2.7–77.4 ms（全部 ≤100 ms） |
| position（计划 + 观测遮脸/越界/包含性/碰撞） | 18/20 格 PASS（2 格 NO_ANCHORS 无元素可测） |
| render（时长/帧数/音频/RMS） | 20/20 PASS |

**数值语义校验（v2，确定性）**：`scripts/check_number_semantics.py`
方法：取元素计数已稳定且不透明度最高的帧 → 用同一栅格器按 IR 复算"期望值"贴片 → 以面板透明度合成到源帧裁切区 → 与真实渲染帧逐像素比较；同时用"半值"贴片做对照，要求期望值明显更优。
实测分离度：期望值 mean ≈2.1 / p99 ≈16，半值对照 p99 ≈80–140。全部数字元素 **PASS**（fx_a/b/face/pd 的 number 与 combo 格均 PASS）。

## 4. 避脸与安全布局（真实人脸）

素材 fx_face：真人照片经 YuNet 检出 **125/125 采样帧**，face union `[471,374,284,357]`（1080×1920）。

| 指标 | 结果 | 阈值 |
|---|---|---|
| 计划遮挡（每个元素窗内采样） | 0 / 38 帧 | 0 |
| **观测遮挡**（渲染帧差异包围盒 ∩ 人脸框） | **0 / 198 帧** | 0 |
| 越界（观测） | 0 / 198 帧 | 0 |
| 栅格包含性（框外像素占比） | 0.000000 | ≤0.03 |
| 非设计碰撞 | 0 / 0 对 | ≤0.01 |

13 个元素（calibration 策略铺满全片）全部落在人脸安全框之外：关键词/数字/列表/卡片均位于人像下方留白区。
真实影像 fx_pd_a/b：观测遮挡 0/8、0/6 帧（英文素材元素较少，样本量小，已如实报告）。

**反向验证**：把 400×120 的贴片故意压在人脸框上时，`position` 立即 FAIL（observed=1.0）；说明该指标不是空转。

## 5. 人工目检记录（实际查看抽帧）

- fx_face list（`/tmp/g2_list2.png` 同源帧）：列表显示「① 对齐 / ② 布局 / ③ 渲染」，简体、序号圆点、三段式，位于人像下方，不压脸。
- fx_face number：`三个/九成/70%` 计数到终值正确（脚本复算 PASS）。
- fx_b 16:9：列表卡位于下三分之一，未压人物头部；卡片文字为简体。
- fx_pd 真人影像：关键词/数字卡落在人脸右侧安全列，1280×720 可读。
- 已修复的视觉缺陷：列表行曾丢失"第"字（"步是对齐"）→ 现为「① 对齐」；列表卡曾压在人物头部 → 引入"无人脸时保守主体带 + 有脸时人脸安全框"策略后回到下三分之一。

## 6. 该闸门中修复的真实缺陷（可追溯）

| 缺陷 | 影响 | 修复 |
|---|---|---|
| `一半` 被解析成 `1半` | 数值语义错误 | `effects/number.py` 成语量词映射 + 繁简折叠 |
| 繁体数字（`兩倍`）不识别 | 数字动效失效 | 同上（逐字折叠保持偏移） |
| 列表项只有序号（"① 第三"） | 语义空洞 | 从转录词流补齐步骤内容 + 序号/量词剥离 |
| 中文单位数字 OCR 误读 | 校验误判 | 校验改为确定性复算（v2），不再依赖 OCR |
| 无人脸时卡片压住人物头部 | 位置质量 | `design.py` 保守主体带（无可靠检测时的降级策略） |
| 16:9 首图卡过大/越界 | 布局质量 | 主体带 + 分区候选 + 字号阶梯降级 |

## 7. 复现命令

```bash
cd /Volumes/ZHITAI/SemanticMotion && export PYTHONPATH=src
.venv/bin/python scripts/effect_matrix.py --fixture fx_face_zh_portrait --out artifacts/g2/matrix/fx_face
.venv/bin/python scripts/gate_runner.py --gate G2 --out artifacts/g2/face_stress \
    --fixtures fx_face_zh_portrait --style calibration
.venv/bin/python scripts/check_number_semantics.py \
    artifacts/g2/matrix/fx_face/combo/motion_ir.json \
    artifacts/g2/matrix/fx_face/combo/fx_face_zh_portrait_combo.mp4 /tmp/num_check.json
.venv/bin/python -m pytest tests/ -q        # 92 passed
```

## 8. 结论与遗留

- **G2 PASS**：四类 P1 动效在 9:16 / 16:9 下单项与组合均通过真实渲染与量化验收；真实人脸上遮脸率 0（198 帧）、越界 0、包含性 0；数值语义经确定性复算全部正确。
- 遗留（不阻断 G2，但不得隐瞒）：英文素材缺中文列表/卡片锚点；真人语音时间精度仍 NOT RUN；审美仅内部评估；列表/卡片在极端窄可用区会触发字号降级（记录 warning）。
- 下一闸门：G3（镜头变化与人物运动、自动降级、局部重渲染、可复现、异常恢复、自动回归）。
