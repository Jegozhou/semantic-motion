# 依赖与许可证清单（初稿，G4 前最终复核）

生成：G0/G1 阶段；复核要求：每个闸门结束前更新。

## 运行时依赖

| 组件 | 版本/来源 | 许可证 | 分发风险 | 处置 |
|---|---|---|---|---|
| openai-whisper（仅参考，未用于生产路径） | brew 20250625_3 | MIT | 低 | 未使用（改用 faster-whisper） |
| faster-whisper | 1.2.1 (pip) | MIT | 低 | ✅ 生产 ASR |
| CTranslate2 | 4.8.3 | MIT | 低 | ✅ |
| Whisper 权重 `Systran/faster-whisper-medium` | CTranslate2 转换自 OpenAI Whisper medium | MIT（OpenAI 权重） | 低 | 本地缓存于 `/Volumes/ZHITAI/cache/huggingface`，不随产品分发 |
| PyTorch（仅 openai-whisper venv） | 2.10.0 | BSD-3-Clause | 低 | 非生产路径 |
| OpenCV (headless) | 5.0.0 | Apache-2.0 | 低 | ✅ 人脸检测 |
| YuNet 模型 `face_detection_yunet_2023mar.onnx` | opencv_zoo | Apache-2.0（仓库）；训练集 WIDER FACE 为研究用途 | **中** | 仅本地检测使用；若对外分发模型文件须复核 WIDER FACE 条款；可替换为自训练/其他许可模型 |
| Pillow | 13.0.0 | HPND | 低 | ✅ 栅格化 |
| NumPy | 2.5.4 | BSD-3-Clause | 低 | ✅ |
| pytest | 9.1.1 | MIT | 低 | 开发期 |
| Noto Sans CJK SC (`assets/fonts/NotoSansSC-Regular.otf`) | notofonts/noto-cjk | SIL OFL 1.1 | 低 | ✅ 随产品分发需保留 OFL 文本与版权声明（待补 `assets/fonts/OFL.txt`） |
| ffmpeg | 8.1 homebrew，`--enable-gpl --enable-libx264 --enable-libx265` | **GPL-3.0**（构建） | **高** | 作为**外部依赖**调用，不随 Skill 分发；闭源分发时需改用 LGPL 构建或 videotoolbox 编码路径 |
| macOS `say` | 系统组件 | Apple 许可 | 中 | 仅用于生成本地测试素材，不随产品分发 |

## 素材许可

| 素材 | 来源 | 许可 | 用途 |
|---|---|---|---|
| `fx_a_zh_portrait` / `fx_b_zh_landscape` / `fx_c_zh_cuts` / `fx_test_cuts` | 本机 `say` TTS + 程序生成图形（本仓库原创） | 项目自有（合成，无第三方权利） | dev 校准，SYNTH_KNOWN |
| `fx_pd_real_16x9_a` / `fx_pd_real_16x9_b` | Wikimedia Commons, "Weekly Address (2009-01-24), small.ogv" | **Public domain**（美国政府作品，17 U.S.C. §105） | dev 真实素材，真实人脸；无人工词级标注 |
| 系统字体（Hiragino/STHeiti/Songti） | macOS | Apple 系统许可 | **不可分发**；产品默认字体固定 Noto |

## 待办（G4 前必须完成）

- [ ] 补 `assets/fonts/OFL.txt`（Noto OFL 原文）
- [ ] 复核 YuNet/WIDER FACE 的分发条件，或替换模型
- [ ] 明确 ffmpeg 依赖交付方式（推荐：要求用户系统自备，版本探测 + 明确报错）
- [ ] 生成 `requirements.lock`（`uv pip freeze`）并纳入报告
