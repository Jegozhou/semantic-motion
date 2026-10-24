# Gate G3 — 高级稳定性

**判定：PASS**
日期：2026-10-24（Asia/Shanghai）｜依据：`docs/semantic-motion-spec.md` §3.2 / §3.5 / §6 / §7

---

## 1. 验收条件对照

| G3 要求 | 结果 | 证据 |
|---|---|---|
| 镜头变化（切镜）处理 | ✅ 切镜不跨覆：2 镜头素材 violations = 0 | `artifacts/g3/pan/fx_face_pan/metrics.json → position.cut_safety` |
| 人物运动处理 | ✅ 人脸位移 103.4 px，覆层固定、最小净距 192 px（**不宣称跟随**） | 同上 `position.subject_motion` |
| 自动降级 | ✅ 5 个降级场景（无人脸 / 人脸居中 / 人脸几乎满屏 / 320×568 小画幅 / 空计划） | `artifacts/g3/stability/stability.json → degradation` |
| 局部修改（单元素改动） | ✅ 只改一个元素：窗内变化 6.86、窗外最大 0.055（≤1.0 容差） | `stability.json → partial_update` |
| 可复现重渲染 | ✅ 同 IR 双跑 534/534 帧哈希一致 | `partial_update.identical_ir_reproducible` |
| 异常恢复 | ✅ 非法 IR 拒绝、不可写目标、损坏输入后重试、失败后续跑可复现 | `stability.json → recovery` |
| 自动回归 | ✅ 11 个步骤全 PASS（3 分 42 秒） | `artifacts/regression/summary.json` |

## 2. 镜头变化与人物运动

素材 `fx_face_pan`（9:16，1080×1920，21.38 s）：两个镜头（9.0 s 处硬切），第一镜头 slow pan、第二镜头 slow zoom，画面中为**真实可检测人脸**（照片合成，测试夹具）。

| 指标 | 结果 |
|---|---|
| 镜头数（自动检测） | 2（切点 9.00 s，与真值一致） |
| 覆层跨切点数 | **0**（30 ms 保护带；无法修剪的锚点直接跳过，不延后） |
| 人脸轨迹 | 1 条，107 采样帧；中心最大位移 **103.41 px** |
| 覆层与人脸最小净距 | **192 px**（32 次采样） |
| 观测遮脸 | 0 / 175 帧 |
| 时间精度（该素材） | n=5，P50 24.48 ms / P90 27.94 ms / max 27.94 ms |

**策略说明（不做过度承诺）**：本项目**不宣称**"跟随人物"，而是采用 **static-safe**：覆层位置固定，人脸并集作为障碍区；镜头切换时旧覆层在切点前收尾，新覆层在切点后重新布局。这是规格 §3.2 允许的保守退化，已写入 IR 的 `subject_motion.policy`。

修复的真实缺陷：切镜处原本会把锚点"延后到切点之后"，导致动作与语音脱节 **601 ms**；现改为"能修剪就修剪（保留起点），否则跳过"，该素材最大误差从 601 ms 降到 **27.94 ms**。

## 3. 自动降级（5 个场景，全部真实渲染）

`PYTHONPATH=src .venv/bin/python scripts/g3_stability.py --out artifacts/g3/stability`

| 场景 | 触发条件 | 行为 | 结果 |
|---|---|---|---|
| `no_face_detected` | 视觉无任何人脸 | 启用"保守主体带"，文字进下方安全区 | 8 元素，遮脸 0，越界 0 → PASS |
| `face_covers_center` | 人脸并集占画面中部 76%×62% | 安全区按人脸框切块，重新选位 | 8 元素，遮脸 0 → PASS |
| `face_covers_nearly_everything` | 人脸并集占 92%×90% | **放弃覆层**（8 个元素全部丢弃并记录 `no_face_free_space`），输出原片透传 | 0 元素，IR 合法，渲染成功 → PASS |
| `small_canvas_320x568` | 真实 320×568 源 | 布局成功；像素级位置指标因低分辨率噪声不可靠 → **NOT RUN**（不冒充通过） | 8 元素，渲染成功 → PASS |
| `empty_plan_zero_motion` | 导演判定"零动效" | IR 合法（tracks 可为空）、渲染为透传 | 0 元素，渲染成功 → PASS |

> 设计原则已写入契约：**主体保护优先于展示覆层**。宁可丢掉一个文字，也不覆盖人脸。

## 4. 局部修改与失败回退

脚本：`scripts/g3_stability.py → partial_update`

| 检查 | 结果 |
|---|---|
| 基线渲染 ↔ 同 IR 重渲染 | 534/534 帧哈希一致（max_pixel_diff 0.0） |
| 只修改一个元素文案（`content.edited=true`） | IR 哈希改变，验证通过 |
| 修改窗**外**画面保持不变 | 80 帧对比，最大平均差 **0.0549**（阈值 1.0，残差来自帧间编码） |
| 修改窗**内**画面确实变化 | 最大平均差 **6.8632** |
| 非法修改（box 越界） | 验证器拒绝：`layout.box [-50,-50,200,200] is not fully inside canvas` |
| 回退 | 重新用原 IR 渲染成功，输出与基线逐帧一致 |

契约变化（已记录）：`content.edited = true` 的元素允许偏离转录文本（人工编辑），但仍禁止生成侧编造事实。

## 5. 异常恢复

| 场景 | 结果 |
|---|---|
| 非法 IR（越界 / 未声明碰撞 / 篡改哈希） | 渲染前拒绝，返回 `E_IR_INVALID`，不产出半成品 |
| 目标路径不可写 | 抛错且**不留下部分文件** |
| 输入损坏（伪造 MP4） | ffmpeg 非零退出 → `E_RENDER_FAILED`；随后正常重试成功 |
| 失败后重跑 | 同一 IR 两次渲染逐帧一致（534 帧），缓存与临时目录不需要人工清理 |

## 6. 自动回归

`PYTHONPATH=src .venv/bin/python scripts/regression.py --out artifacts/regression --gate G3`

| 步骤 | 结果 | 耗时 |
|---|---|---|
| pytest（92 项） | PASS | 28.4 s |
| env_probe | PASS | 16.8 s |
| baseline（G0 透传/重编码） | PASS | 2.6 s |
| gate_g1_calibration（56 事件） | PASS | 28.1 s |
| gate_g1_product（5 素材 + 确定性） | PASS | 33.7 s |
| effect_matrix_fx_face | PASS | 23.7 s |
| effect_matrix_fx_pd | PASS | 15.2 s |
| gate_g2_face_stress | PASS | 6.2 s |
| g3_stability | PASS | 56.1 s |
| gate_g3_motion | PASS | 6.7 s |
| numeric_semantics（5 素材） | PASS | — |
| **总计** | **PASS（11/11）** | **3 分 42 秒** |

汇总文件：`artifacts/regression/summary.json`（含每条命令 argv、退出码、耗时、stdout 尾部）。

## 7. 新增/变更的接口与策略（都已写入 `docs/INTERFACES.md` 或代码注释）

1. IR 允许"零覆层"（`elements=[]` 且 `tracks=[]`），渲染为透传。
2. `content.edited=true` 用于标注人工编辑，豁免"文本必须可在转录中找到"的不变量。
3. 主体保护优先：降级后仍遮挡人脸 >1% 的元素会被丢弃，并记入 `provenance.skipped_decisions`。
4. 低分辨率画幅（<480 px）的像素级位置指标标记 `NOT RUN` + 原因，而不是给出可能误判的 PASS/FAIL。
5. `subject_motion` / `cut_safety` 纳入 `semotion.metrics/1`。

## 8. 遗留与限制（不隐瞒）

| 项 | 状态 | 说明 |
|---|---|---|
| 视觉跟踪 | **未实现**（按设计） | 只做 static-safe；不宣称跟随稳定 |
| 局部重渲染（只重编码受影响帧） | 未实现（当前为整片重渲染，24 s 成片约 3 s） | 规格列为 P2；已在报告注明 |
| 真人语音时间精度 | NOT RUN | 无人工标注，仍不可外推 |
| 留出集 | 无 | 测试非盲（manifest 已声明） |
| 极端画幅（<480 px）像素指标 | NOT RUN | 低分辨率噪声与覆层同量级，如实标注 |

## 9. 结论

G3 **PASS**：镜头/运动下的稳定性、自动降级、局部修改、可复现、异常恢复与自动回归均有真实运行证据。
未实现项（视觉跟踪、帧级局部重渲染）属 P2，已在报告与 `PROGRESS.md` 明确标注，不计入 G3 通过范围。
