# 依赖与许可证清单（G4 终审）

复核日期：2026-10-24｜锁定文件：`requirements.lock`（30 个包）
结论：**可用于内部生产与非商业交付**；对外分发若要闭源，需先处理 ffmpeg 的 GPL 与 YuNet 模型条款（见 §4）。

## 1. 运行时依赖（生产链路）

| 组件 | 版本 | 许可证 | 风险 | 处置 |
|---|---|---|---|---|
| faster-whisper | 1.2.1 | MIT | 低 | ✅ 生产 ASR（CPU int8） |
| CTranslate2 | 4.8.3 | MIT | 低 | ✅ 推理后端 |
| Whisper 权重 `Systran/faster-whisper-medium` | 本地快照 | MIT（OpenAI 发布权重） | 低 | 缓存在 `/Volumes/ZHITAI/cache/huggingface`，**不随产品分发** |
| PyAV (`av`) | 19.0.2 | BSD-3-Clause | 低 | 作为 faster-whisper 依赖安装，生产路径不走它解码 |
| OpenCV (headless) | 5.0.0 | Apache-2.0 | 低 | ✅ 人脸检测 / 图像处理 |
| YuNet ONNX `face_detection_yunet_2023mar.onnx` | opencv_zoo | Apache-2.0（仓库）；训练集 WIDER FACE 为研究用途 | **中** | 仅本地检测；对外分发模型文件前需复核 WIDER FACE 条款，或替换模型 |
| Pillow | 13.0.0 | HPND | 低 | ✅ 文字/图形栅格化 |
| NumPy | 2.5.4 | BSD-3-Clause | 低 | ✅ |
| opencc-python-reimplemented | 0.1.7 | Apache-2.0 | 低 | ✅ 繁简归一 |
| pytest | 9.1.1 | MIT | 低 | 仅开发 |
| Noto Sans CJK SC `assets/fonts/NotoSansSC-Regular.otf` | notofonts/noto-cjk | **SIL OFL 1.1**（原文见 `assets/fonts/OFL.txt`） | 低 | ✅ 可随产品分发，须保留许可证与版权声明 |
| ffmpeg / ffprobe | 8.1（homebrew，`--enable-gpl --enable-libx264 --enable-libx265`） | **GPL-3.0**（该构建） | **高** | 作为**外部依赖**调用，不随 Skill 分发；闭源分发需改用 LGPL 构建或 `videotoolbox` 编码路径 |
| macOS `say` | 系统 | Apple 许可 | 中 | 仅用于生成本地测试素材，不分发 |

## 2. 系统/开发工具（非分发）

| 工具 | 用途 | 许可证/说明 |
|---|---|---|
| ImageMagick 7.1.2 | 裁切/对比（现行数值校验为确定性复算，不依赖 OCR） | ImageMagick License（Apache-2.0 风格） |
| tesseract 5 | 旧版 OCR 校验（已被确定性复算取代） | Apache-2.0 |
| rsvg-convert / cairo / pango | 备用矢量栅格化（当前未使用） | LGPL / MPL |
| Python 3.12（uv 安装） | 运行时 | PSF License |
| macOS 系统字体（Hiragino/STHeiti/Songti） | 仅本机显示 | Apple 系统许可，**不可分发**；产品默认字体固定 Noto |

## 3. 素材许可

| 素材 | 来源 | 许可 | 用途 |
|---|---|---|---|
| fx_a / fx_b / fx_c / fx_d / fx_e / fx_t1–t3 / fx_test_cuts | 本仓库：macOS `say` TTS + 程序生成图形 | 项目自有（合成，无第三方权利） | dev / calibration（SYNTH_KNOWN） |
| fx_pd_real_16x9_a / _b | Wikimedia Commons「Weekly Address (2009-01-24)」 | **Public domain**（美国政府作品，17 U.S.C. §105） | 真实影像测试 |
| 肖像照片 `assets/media/portraits/obama_official.jpg` | Wikimedia Commons「Official portrait of Barack Obama」 | **CC BY 3.0**（Pete Souza） | 仅用于 fx_face_* 测试夹具（照片合成），**不得作为真实拍摄素材对外发布**；如需公开须署名 |

## 4. 分发前必须完成（当前未做）

- [ ] ffmpeg 交付方式二选一：要求用户自备（推荐，脚本已做版本探测与明确报错）或改为 LGPL/videotoolbox 构建。
- [ ] YuNet/WIDER FACE 条款复核，或替换为自训练/许可更清晰的检测模型。
- [ ] 若公开示例成片包含 `fx_face_*`（含 CC BY 肖像），须显著署名 Pete Souza / CC BY 3.0。
- [ ] 公开发布时附带 `assets/fonts/OFL.txt`。

## 5. 生成方式声明（干净室）

- 所有代码为本仓库原创实现，依据 `docs/semantic-motion-spec.md` 与公开通用技术知识编写。
- 未使用、未复制任何第三方动效产品的源码、内部提示词或工程结构。
- 运行时**不执行**用户文本/转录/LLM 生成的代码；只消费 schema 校验过的 Motion IR 与白名单原语。
