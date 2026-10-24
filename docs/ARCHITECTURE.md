# 架构说明（Architecture）

## 1. 仓库结构

```
semantic-motion/
├── .claude-plugin/
│   ├── plugin.json            # Claude Code 插件清单
│   └── marketplace.json       # 单插件 marketplace（/plugin marketplace add Jegozhou/semantic-motion）
├── commands/motion.md         # /motion 斜杠命令：一句话跑完整流程
├── agents/
│   ├── motion-director.md     # 子代理：读 brief 写 plan（可选，用于长视频分工）
│   └── motion-critic.md       # 子代理：按评审清单审看联系表与 QA 报告
├── skills/semantic-motion/    # ← 技能本体（也可单独复制到 ~/.claude/skills/）
│   ├── SKILL.md               # 技能入口：工作流、导演原则、规则
│   ├── reference/             # 按需加载的参考：plan 结构、元素、设计指南、改稿映射、QA 代码
│   ├── scripts/sm             # CLI 包装器（使用 ~/.semantic-motion/venv）
│   ├── scripts/setup.sh       # 一次性安装：venv + 依赖 + 模型
│   ├── requirements.txt
│   └── engine/semantic_motion # Python 引擎
├── tests/                     # pytest 单元/集成测试 + 合成时间基准
└── docs/                      # STATUS / DECISIONS / ARCHITECTURE / TEST_REPORT / TEST_FOOTAGE
```

## 2. 数据流

```
video ──► analyze ─────────────────────────────► brief.md ──► Claude（导演）──► plan.json
           │ media probe (CFR, BT.709, HDR→SDR)                                   │
           │ audio features 10 ms (dB, VAD, onset, f0)                            ▼
           │ ASR (faster-whisper, 词级) → refine_words → 句/节拍切分        compile（确定性）
           │ vision (分割/人脸/姿态/手/运动/镜头切换/烧录文字/亮度/杂乱)      │ anchors → 精确时间
           ▼                                                                      │ timing → 入场/落点/退出
       analysis/*（按输入哈希缓存）                                              │ layout → 代价场 + 硬障碍求解
                                                                                  │ auto-repair
                                                                                  ▼
                                                     timeline.json ──► pre-QA ──► render (Skia + ffmpeg)
                                                                                  ▼
                                               final.mp4 / overlay.mov ──► post-QA ──► contact sheet / report
```

## 3. 引擎模块

| 模块 | 职责 |
|---|---|
| `media.py` | ffprobe 探测；ffmpeg 管道逐帧解码（恒定帧率 RGBA、显式色彩矩阵）与编码；音频直通 |
| `audio.py` | 10 ms 响度包络、自适应 VAD、停顿、onset、基频；强调分数 |
| `asr.py` | Whisper 后端（faster-whisper / mlx-whisper）；`refine_words`（soft_start + silence_end）；按“视觉节拍”切句（不在悬空虚词处断句、碎片合并） |
| `vision.py` | MediaPipe 分割/人脸/姿态，手部热区、运动、镜头切换（帧级）、DB 文字检测、亮度/杂乱网格；镜头构图与开放侧 |
| `brief.py` | 生成给 Claude 读的 `brief.md` |
| `anchors.py` / `text.py` | 锚点语法解析、字级时间轴、繁简/标点归一化、模糊匹配、中英文数词与数字变体 |
| `compiler.py` | plan → timeline：锚点、时间（lead-in、落点、spoken reveal、退出到停顿、最短阅读时间）、尺寸、版式、镜头运动、跨镜头裁剪、自动修复 |
| `layout.py` | 代价场与硬障碍（脸/头/安全区），网格穷举求解，box 统计 |
| `style.py` | 设计 token（单位 u = min(W,H)/100）、风格（balanced/minimal/bold）、强调色自动选择与对比度、平台安全区 |
| `elements/*` | 元素库：keyword, stat, list, flow, compare, chart, quote, title, lower_third, callout, captions, custom |
| `render/*` | 合成器（只在需要时作画，其他帧透传）、缓动、字体（CJK 优先回退）、逐字排版 |
| `qa.py` | pre/post QA、确定性检查、联系表、时间线条带图、报告 |
| `project.py` / `pipeline.py` | 项目目录、阶段缓存、快照/撤销/差异、历史日志 |
| `autoplan.py` | 启发式基线计划（无 LLM） |
| `cli.py` | `sm` 命令行 |

## 4. 关键不变量（由测试守护）
1. **确定性**：同一 plan + 分析 ⇒ 同一 timeline（`test_real_project_compiles_deterministically…`）；`--determinism` 渲染两次逐帧哈希一致。
2. **不遮脸**：求解器永远不会把框放进硬障碍格（`test_solver_never_places_on_hard_cells`，40 组随机场景）；pre-QA 对每个采样帧检查人脸框。
3. **QA 框 = 实际画面**：渲染 alpha 包围盒落在 QA 框 ± 阴影内（`test_rendered_footprint_matches_qa_box`）。
4. **原声不变**：post-QA 校验解码音频一致性与 A/V 偏移。
5. **锚点可解析**：spoken numbers、繁简、纠错表、模糊匹配均有单测。

## 5. 扩展点
- 新元素：在 `elements/` 新增 `el_<type>.py`，实现 `variants()`（候选自然尺寸）与 `draw(canvas, t)`（时间的纯函数），并在 `reference/elements.md` 记录字段。
- 新风格：`style.py` 中增加 token 组合。
- 新 ASR 后端：`asr.transcribe` 的后端分派。
