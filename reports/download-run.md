# 真实下载素材实测：郑钦文采访（2026-10-24）

**目的**：用一条**真实下载**的中文口播素材，跑完整新链路——LLM 选动效 → Motion IR → Remotion 渲染 → 逐词字幕 → 指标。
**结论**：position / render **PASS**；timing **NOT RUN**（该素材没有人工词级真值，不能自证时间精度）。过程中修了 4 个真实缺陷（见 §5）。

---

## 1. 素材与授权

| 项 | 内容 |
|---|---|
| 来源 | Wikimedia Commons `File:【巴黎奥运会】郑钦文：奥运会是我的梦想 但我的梦想还远没结束.webm` |
| 许可 | **CC BY 3.0**（中国新闻社 / cnsvideo，经 Commons 上传） |
| 原始规格 | 1920×1080，52.44 s，AV1 + Opus |
| 本实验输入 | `inputs/zhengqinwen_25s.mp4`：裁掉底部 14%（原片自带中文字幕条）→ 1920×920，取 23–48 s 片段 25 s |
| 说明 | 原片仍有台标与麦克风牌；本次只评估动效链路，不声称素材可再分发 |

> 为什么要裁：原片已烧录中文字幕，若不裁会与我们生成的字幕重叠。裁剪只影响画幅，不改语音。

## 2. LLM 决策（真实依据，非模板）

锚点不是规则生成的（中文 ASR 给的是单字词，规则会切出"其实还远远没有结"这种半句）。本次让 **LLM 直接选词跨度**，时间仍严格来自词表：

| 锚点 | 文本 | 时间 | LLM 选择 | 理由（节选） |
|---|---|---|---|---|
| a00 | 奥运会是我的梦想没错 | 0.00–2.02 | **card** | 开篇命题，整句承载，卡片比关键词合适 |
| a01 | 但是我还有其他的梦想 | 2.02–3.70 | **keyword** | 全段语义转折，7 字做关键词更克制 |
| a02–a04 | 每一年我们都有比赛 / 每一年都有更大的困难 / 更大的山坡 | 11.18–16.54 | **list**（3 行逐行揭示） | 递进结构，比三个独立关键词更能表达层层加码 |
| a06 | 没有办法向前进步 | 22.68–24.32 | **keyword** | 与"停滞"构成因果，取后半句轻强调 |
| a05 | 停滞在一个位置 | 21.58–22.68 | **跳过** | 与 a06 相邻且语义连续，两个一起会过密 |

- 决策文件：`artifacts/llm/download_zhengqinwen.decisions.json`（`decision_model=deepseek-v4.1-flash`）
- LLM 选词跨度：`artifacts/dl/spans.json` → `scripts/llm_anchors.py` 生成 `artifacts/dl/anchors_llm.json`（每个时间都是词表里词的时间，未编造）
- 校验：`validate_decisions` → `[]`（8 条规则：白名单、锚点存在、时间窗、覆盖率、数字、plain、避免遮脸等）

## 3. 成片与指标

| 项 | 结果 |
|---|---|
| 成片 | `artifacts/dl/run/zhengqinwen_25s_motion.mp4`（1920×920，625 帧，25.003 s，h264+aac） |
| 覆盖策略 | 4 个元素 / 覆盖率 46.6% / 跳过 2 个（含 1 个重复锚点） |
| 字幕 | 10 条 cue 生成 → 6 条烧录（4 条因与元素文案重复被抑制）；简体、无行尾标点 |
| 时间精度 | **NOT RUN**（无人工词级真值；不以 ASR 自证） |
| 遮脸 | **0 / 279 采样帧**（人脸并集 `[932,192,266,383]`，与元素最小净距 42 px） |
| 越界 / 框外像素 | 0 / 0.0（226,828 个动效像素全在声明框内） |
| 切镜安全 | PASS（检测到 2 个镜头；末元素窗口在切点前收束） |
| 音频 | 时长差 0.000 s，RMS 差 0.01 dB |
| 耗时 | 主渲染 50.3 s + overlay 验证渲染 45.8 s + 其余 1.5 s |

产物：`artifacts/dl/run/{motion_ir.json, captions.json, remotion_props.json, render_manifest.json, metrics.json, overlay_pass.mp4}`；
演示：`delivery/demo_downloaded/{before_after.mp4, storyboard.png}`。

## 4. 复现

```bash
cd /Volumes/ZHITAI/SemanticMotion && export PYTHONPATH=src
# 1) 输入（Commons 原片 → 裁掉原字幕条 → 取 25s）
#    原始下载：assets/media/downloads/cns_zhengqinwen.webm  (CC BY 3.0)
# 2) LLM 选词跨度 → 锚点
.venv/bin/python scripts/llm_anchors.py --transcript artifacts/dl/t.json \
  --spans artifacts/dl/spans.json --out artifacts/dl/anchors_llm.json
# 3) 全链路
.venv/bin/python -m semotion.cli run --input inputs/zhengqinwen_25s.mp4 \
  --outdir artifacts/dl/run --gate G5 --backend remotion \
  --decisions artifacts/llm/download_zhengqinwen.decisions.json \
  --captions --anchors artifacts/dl/anchors_llm.json
```

## 5. 这次跑真素材暴露并修复的问题

| # | 问题 | 影响 | 修复 |
|---|---|---|---|
| 1 | 画面运动大时，"渲染帧 vs 源帧"差分阈值被噪声推到 206 → 观测样本 0，却报 PASS（**假通过**） | 位置指标失去意义 | 新增 **overlay-only 验证渲染**（隐藏源视频、只渲染动效层），遮脸/越界/包含性直接测量动效层；无 overlay 时对高噪素材明确标 NOT RUN |
| 2 | LLM 决策路径没有切镜守卫 | 末元素跨过切点（24.4 s）→ position FAIL | 切镜安全改为**渲染前置约束**：任何导演（LLM 或规则）产出的 plan 都过一遍 `_apply_shot_policy` |
| 3 | 字幕字号按画面高度计算 | 1920×920 横屏时字幕只有 28 px，看不清 | 横屏按宽度 2.4%、竖屏按高度 3.0% 计算 |
| 4 | 卡片字号不随框宽收缩 | "奥运会是我的梦想没错"被折成"…没 / 错" | 按框宽与字符宽度自适应字号（CJK≈1em，拉丁≈0.55em），并给强调条/内边距留预算 |
| 5 | overlay 验证渲染覆盖了主渲染的 manifest/props 产物 | 产物互相覆盖，审计困难 | 验证渲染的 manifest/props 按输出名分文件 |

## 6. 仍然的限制

- **时间精度 NOT RUN**：没有人工标注；这条素材只能证明"链路能跑、位置/渲染正确"，不能证明时间准。
- 素材自带台标与麦克风牌，动效已避开人脸，但无法避开这些台标（属素材本身）。
- 原片字幕条被裁掉 14%，画幅变为 1920×920（非标准 16:9）。
- LLM 决策是本会话模型产出，可复现的是冻结后的 IR（规格 §3.5）。
