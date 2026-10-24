# Gate G0 — 预检与可执行基线

**判定：PASS**（2026-10-24，Asia/Shanghai）
依据 `docs/semantic-motion-spec.md` §2。本报告只写实际运行产物；未运行项已标 `NOT RUN`。

---

## 1. 验收条件对照

| G0 要求 | 结果 | 证据 |
|---|---|---|
| 工程初始化完成 | ✅ | `pyproject.toml`、`.venv`、`src/semotion/`、`docs/INTERFACES.md`（contracts-1.0.0） |
| 环境信息写入 `ENVIRONMENT.md` | ✅ | `ENVIRONMENT.md` + 机器可读 `artifacts/env/env_probe.json` |
| 基础视频解码可运行 | ✅ | `ffprobe`/`ffmpeg` 实测（见 §2） |
| 音视频透传/重新封装可运行 | ✅ | `artifacts/g0/baseline_remux.mp4` |
| 可复现命令 | ✅ | `PYTHONPATH=src .venv/bin/python -m semotion.cli baseline run --input <in> --out artifacts/g0` |
| **实际生成可播放 baseline MP4 与日志** | ✅ | remux + re-encode 两条 MP4，`artifacts/g0/logs/` |
| 无法运行则 BLOCKED | 不适用（无阻断） | — |

## 2. 输入与输出（真实文件）

| 项 | 路径 | 事实 |
|---|---|---|
| 输入 | `tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4` | 1080×1920, 25 fps, **24.200 s**, h264 + aac 48 kHz 立体声, sha256 `67ea5476d154eb34…` |
| 透传输出 | `artifacts/g0/baseline_remux.mp4` | 522,917 B, sha256 `087463e8c436f1af…` |
| 重编码输出 | `artifacts/g0/baseline_reencode.mp4` | 553,936 B, sha256 `661f8be7294fd5ea…` |
| 运行日志 | `artifacts/g0/logs/{remux,reencode,audio_src,audio_dst,frames_src,frames_dst}.log` | 含完整 argv |
| 机器判定 | `artifacts/g0/baseline_report.json` | `"verdict": "PASS"` |

两者均可用 `ffprobe` 正常解析、可逐帧解码（解码帧数 605 = 24.2 s × 25 fps）。

## 3. 验证方法与结果（不使用字节一致作为标准）

### 3.1 流拷贝（remux）
| 检查 | 结果 |
|---|---|
| 时长 | 24.200 s vs 24.200 s，delta 0.000 |
| 流数量 | video 1/1，audio 1/1 |
| 几何/帧率 | 1080×1920 / 25 fps 一致 |
| **解码音频 PCM 哈希** | `ec89e650b7821ef9…` **完全一致** |
| **逐帧解码哈希** | 605/605 帧**完全一致** |

### 3.2 重编码（re-encode，渲染目标编码）
| 检查 | 结果 | 阈值 |
|---|---|---|
| 时长 | delta 0.000 s | ≤ 1 帧（0.040 s） |
| 分辨率 | 1080×1920 | 一致 |
| 编码 | h264 + aac | 契约要求 |
| **逐帧像素差** | max 0.0118 / mean 0.0058（320×180 灰度） | ≤ 12.0 |
| 音频存在 | aac 48 kHz | — |

> 说明：MP4 字节不一致是正常的（编码器元数据/时间基），因此按规格 §3.5 采用**解码后帧比较**。

## 4. 环境结论摘要

- macOS 26.2 / Apple M4 10 核 / 16 GB / 工作盘 307 GB 可用（系统盘仅 3 GB，禁止写大文件）。
- ffmpeg 8.1：480 个滤镜；**无 libass/freetype ⇒ 无 `drawtext`/`subtitles`/`ass`**。架构决定：文字与图形由 Python/Pillow 栅格化为 RGBA PNG，再用 `overlay` 合成（已写入契约）。
- ASR：openai-whisper CLI 可用但 `--device mps` 在 `word_timestamps` 下崩溃（float64 不支持，实测 20.2 s 失败）；**改用 faster-whisper / CTranslate2 `medium` int8 CPU**，24 s 音频转录 7.4 s（RTF 0.31），词级时间戳可用。
- 中文语音素材：`say` 的 Tingting/Meijia 可用；Flo/Reed/Rocko 等个人音色无本地数据（输出 0.016 s 空音频），不可用。
- 字体：项目内置 `assets/fonts/NotoSansSC-Regular.otf`（SIL OFL 1.1），避免系统字体分发风险。

## 5. 许可证初审（详见 `reports/licenses.md`，G4 前最终复核）

- openai-whisper / 权重：MIT ✅；PyTorch BSD-3 ✅；OpenCV Apache-2.0 ✅；Pillow HPND ✅；NumPy BSD-3 ✅。
- YuNet ONNX：opencv_zoo Apache-2.0（训练集 WIDER FACE 研究用途）⚠️ 仅本地检测使用。
- Noto Sans CJK SC：SIL OFL 1.1 ✅（随产品分发需保留许可证）。
- ffmpeg homebrew 构建为 **GPL**（`--enable-gpl`）：作为**外部依赖调用**，不随 Skill 分发；若未来闭源分发需替换编码路径（videotoolbox / LGPL 构建）。⚠️
- 系统字体（Hiragino/STHeiti）仅本机可用，**不可分发**；产品默认字体固定 Noto。

## 6. 预算（实测）

| 操作 | 实测 |
|---|---|
| 1080×1920 20 s x264 `medium crf20` 编码 | 2.8 s |
| 1080×1920 24.2 s 重编码 | 1.1 s |
| whisper medium（faster-whisper int8, CPU）24 s 音频 | 7.4 s（模型加载 1.6 s） |
| 磁盘占用（不含模型） | < 2 GB |

## 7. 未完成 / 未运行项（G0 范围内）

- 真人中文口播素材（HUMAN 真值）：`NOT RUN`（用户未提供；按规格 §5 允许用 SYNTH_KNOWN 校准，但不得冒充真人验证）。
- 公开 PD 素材引入：`NOT RUN`（G1 计划内）。
- 自定义动效代码沙箱：`NOT RUN`（规格 §3.6 允许"无法隔离则禁用"，本项目采用声明式 IR + 白名单原语，不执行生成代码）。

## 8. 复现命令

```bash
cd /Volumes/ZHITAI/SemanticMotion
export PYTHONPATH=src
.venv/bin/python scripts/env_probe.py artifacts/env/env_probe.json
.venv/bin/python -m semotion.cli baseline run \
  --input tests/fixtures/clips/fx_a_zh_portrait/fx_a_zh_portrait.mp4 \
  --out artifacts/g0
cat artifacts/g0/baseline_report.json | head -20
```

## 9. 结论

G0 **PASS**：环境可信、工具链可用、编解码链路实测通过、基线视频真实可播放且验证有据。
不宣称任何语义动效能力已完成——G1 尚在实现中（见 `PROGRESS.md` §2/§3）。
