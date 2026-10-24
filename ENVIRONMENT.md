# ENVIRONMENT.md — Semantic Motion 环境与依赖预检（G0）

生成时间：2026-10-24（Asia/Shanghai）
机器：`/Volumes/ZHITAI/SemanticMotion`，项目内 `.venv`。
机器可读原始数据：`artifacts/env/env_probe.json`（由 `scripts/env_probe.py` 生成，命令见文末）。

---

## 1. 主机与资源

| 项目 | 实测值 |
|---|---|
| 操作系统 | macOS 26.2 (build 25C56), Darwin 25.2.0, arm64 |
| 机型 | Mac16,10 — Apple M4 |
| CPU | 10 逻辑核（4 性能 + 6 能效）；无独立 NVIDIA GPU |
| GPU | Apple M4 集成 GPU，Metal 4；PyTorch MPS 可用 |
| 内存 | 16 GB |
| 磁盘 | `/Volumes/ZHITAI` 剩余 **307 GB**；系统盘 `/` 仅剩 3 GB（**不要在系统盘写大文件**） |
| 显示器 | 1920×1080 |
| 网络 | pypi 200 / huggingface 200 / github 200 / upload.wikimedia.org 301（可达，走 https） |

**结论：** 本地可完成全部音视频处理、ASR（CPU/MPS）、图形栅格化与编码；无需云端付费。

## 2. 工具链

| 工具 | 版本 / 路径 | 状态 |
|---|---|---|
| ffmpeg | 8.1（homebrew, `--enable-gpl --enable-libx264 --enable-libx265 --enable-videotoolbox --enable-libvpx --enable-libsvtav1`） | ✅ |
| ffprobe | 8.1 | ✅ |
| ffmpeg 滤镜 | 480 个；`overlay / zoompan / geq / xfade / fade / scale` ✅；**`subtitles` / `ass` / `drawtext` ❌（未编译 libass/libfreetype）** | ⚠️ 见下 |
| Python (项目) | 3.12.13 (`uv venv .venv`)：numpy 2.5.4 / pillow 13.0.0 / opencv-python-headless 5.0.0 | ✅ |
| Python (ASR) | 3.14（brew openai-whisper 20250625_3 venv）：torch 2.10.0 + MPS | ✅ |
| Node | v24.15.0（仅工具链备用，流水线不依赖） | ✅ |
| ImageMagick | 7.1.2-18 Q16-HDRI | ✅（备用，主链路用 Pillow） |
| rsvg-convert / cairo / pango / librsvg | 已安装 | ✅（备用矢量栅格化） |
| tesseract / poppler | 已安装 | ✅（未使用） |
| macOS `say` | 中文音色可用：Tingting(zh_CN)、Meijia(zh_TW) 等；`Flo/Reed/Rocko` 等“个人音色”无本地数据，合成出空音频（0.016 s），**不可用** | ✅/⚠️ |
| 字体 | 系统：Hiragino Sans GB、STHeiti Light/Medium、Songti、Arial Unicode；**项目内置 OFL 字体 `assets/fonts/NotoSansSC-Regular.otf`（Noto Sans CJK SC，SIL OFL 1.1）** | ✅ |

### 关键技术约束（影响架构，已固化到 INTERFACES.md）

> **ffmpeg 无 libass/libfreetype ⇒ 无 `drawtext`/`subtitles`/`ass`。**
> 因此所有文字与图形一律由 Python（Pillow / NumPy）逐帧栅格化为 RGBA PNG，
> 再用 `overlay` 按时间窗合成。这也保证了渲染的确定性（不依赖滤镜版本的文字排版差异），
> 并避免了在渲染期执行任何外来文本（安全约束 §3.6）。

## 3. Harness 能力（实际可调用）

| 能力 | 状态 | 说明 |
|---|---|---|
| 子智能体 | ✅ | `spawn_teammate`（持久队友，共享文件系统）+ `subagent` / `subagent_codex` / `workflow` |
| 并发上限 | 本次采用 ≤3 个并发（规格 §1.4）；实际调度为 2 个专业 Agent + 总 Agent |
| 文件权限 | ✅ `danger-full-access`；审批提示关闭（无需交互授权） |
| Skill 根目录 | `~/.dsh/skills/<name>/SKILL.md`（已在 DSH 打包产物中确认常量 `.dsh/skills`）；`~/.claude/skills` 为当前会话已加载的 297 个 skill 来源 |
| 网络 | ✅ 可下载模型/字体/公开测试素材 |
| 子智能体隔离 | 共享同一工作目录（非独立容器）⇒ 按文件路径划分写域，避免覆盖；**不能**宣称严格沙箱隔离 |

## 4. 可用模型 / 依赖 / 许可证清单（G0 初审）

| 组件 | 用途 | 许可证 | 风险与处置 |
|---|---|---|---|
| openai-whisper (代码) | ASR | MIT | 可商用 ✅ |
| whisper 权重 (tiny/small/medium/large-v3) | ASR | OpenAI 以 MIT 发布 | 已缓存 `~/.cache/whisper/medium.pt`（1.42 GB）✅ |
| PyTorch | ASR 运行时 | BSD-3 | ✅ |
| YuNet 人脸检测 ONNX (`face_detection_yunet_2023mar.onnx`) | 人脸框 | opencv_zoo 仓库 Apache-2.0；训练集 WIDER FACE 为研究用途 | ⚠️ 仅用于本地检测；若对外分发模型文件需复核 WIDER FACE 条款 |
| OpenCV (headless 5.0.0) | 图像处理 | Apache-2.0 | ✅ |
| Pillow | 文字栅格化 | HPND | ✅ |
| NumPy | 数值 | BSD-3 | ✅ |
| Noto Sans CJK SC | 字幕/卡片字体 | SIL OFL 1.1 | ✅ 可随产品分发（必须保留许可证） |
| 系统字体 Hiragino/STHeiti | 备用 | Apple 系统许可 | ⚠️ **不可随产品分发**；产品默认字体固定为 Noto Sans CJK SC |
| ffmpeg (homebrew 构建) | 编解码 | **GPL-3.0**（因 `--enable-gpl` + libx264/x265） | ⚠️ 分发捆绑二进制须履行 GPL；我们把 ffmpeg 作为**外部依赖**调用，不随 Skill 分发；如未来需要闭源分发，改用系统 `videotoolbox` 编码或 LGPL 构建 |
| macOS `say` 语音 | 合成校准素材 | Apple 系统组件 | ⚠️ 仅用于本地测试素材，不随产品分发 |

**未使用**：任何第三方动效模板、对标项目代码/提示词/工程结构（干净室：仅依据 `docs/semantic-motion-spec.md` 与本机公开通用工具实现）。

## 5. 素材来源与真值分级

| 类别 | 来源 | 真值等级 | 用途 |
|---|---|---|---|
| 合成中文口播 | macOS `say`（Tingting/Meijia）+ 程序生成背景插画 | `SYNTH_KNOWN`（逐单元 TTS 拼接，起止时间由 ffprobe 实测，**不经 ASR**） | 时间精度校准、G1–G3 回归 |
| 公开测试素材 | 待 G1 引入（Wikimedia Commons 公有领域/PD-USGov），需登记许可 | `MACHINE_WEAK` 或 `HUMAN`（视是否人工标注） | 真实素材路径、面部保护 |
| 人工标注 | 用户尚未提供 | `HUMAN` | 目前 **NOT RUN** |

> 合成校准 ≠ 真人语音验证。报告中必须区分，不得以机器估算自证机器精准。

## 6. 资源预算（G0 实测）

| 操作 | 实测耗时/占用 | 预算 |
|---|---|---|
| 1080×1920 20 s x264 `-preset medium -crf 20` 编码 | 2.8 s（≈7× 实时） | 单条 30 s 成片渲染 < 30 s |
| 1080×1920 24 s 视频重编码 | 1.1 s | ✅ |
| whisper `tiny` CPU，24 s 音频 | 3.2 s | 试探用 |
| whisper `medium` CPU，24 s 音频 | **60 s**（下载 1.42 GB / 25 s 一次性） | 单条 30 s 口播 ≈ 75 s；**缓存复用，不重复下载** |
| whisper `medium` **MPS + word_timestamps** | **不可用**：`timing.py` DTW 需要 float64，MPS 不支持（`TypeError`），实测 20.2 s 后失败 | 已放弃 MPS，见下 |
| faster-whisper / CTranslate2 `medium` int8，CPU，词级时间 | **7.4 s / 24 s 音频（RTF 0.31）**，模型加载 1.6 s（缓存后） | ✅ **选定方案**（`pyav` 解码在现装版本不兼容，改为 ffmpeg 预解码 numpy 输入） |
| 人脸检测 YuNet（1080p, 5 Hz 采样） | 待测（G1） | 预计 < 5 s/分钟视频 |
| 磁盘 | 单条 1080p 15–60 s 成片 2–25 MB；中间帧序列 < 500 MB/条 | 307 GB 可用，充裕 |
| 子智能体并发 | ≤ 3 | 硬约束 |

**阻塞条件（当前无）：** 若后续需要真人中文口播素材（HUMAN 真值）、或需要闭源分发、或需要自定义代码执行沙箱，则对应验收维度记为 `NOT RUN/BLOCKED`，不伪造通过。

## 7. 复现命令

```bash
cd /Volumes/ZHITAI/SemanticMotion
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json
cat artifacts/env/env_probe.json
```

## 8. G0 基线实测证据

```bash
PYTHONPATH=src .venv/bin/python -c "
from pathlib import Path
from semotion.baseline import run_g0_baseline
print(run_g0_baseline(Path('tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4'), Path('artifacts/g0'))['verdict'])"
```

* 输入：`tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4`（1080×1920, 25 fps, 24.2 s, h264+aac）
* 输出：`artifacts/g0/baseline_remux.mp4`、`artifacts/g0/baseline_reencode.mp4`
* 报告：`artifacts/g0/baseline_report.json` → `"verdict": "PASS"`
  * remux：605/605 帧哈希一致、PCM 音频哈希一致、时长/流/分辨率一致
  * re-encode：605 帧逐帧最大像素差 0.0118（阈值 12），时长差 0.000 s
