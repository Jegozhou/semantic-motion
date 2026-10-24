# Semantic Motion — 模块接口契约（INTERFACES.md）

契约版本：`contracts-1.0.0`（所有 JSON 带 `schema` 字段；不兼容变更必须升版本号）
最后更新：G0

本文件是并行开发的唯一接口基准。任何模块改动跨模块数据形状，必须先改本文件并同步所有实现。

---

## 1. 运行环境与入口

```bash
export PYTHONPATH=src            # 项目根目录下
.venv/bin/python -m semotion.cli <command>      # 主入口
.venv/bin/python -m semotion.fixtures ...       # fixture 构建（也可经 cli）
```

* Python：3.12（`.venv`），依赖 `numpy / pillow / opencv-python-headless`。
* ffmpeg / ffprobe：系统 `/opt/homebrew/bin`，版本记录在 `artifacts/env/env_probe.json`。
  **该构建无 libass/libfreetype**，因此不存在 `subtitles`/`ass`/`drawtext` 滤镜：
  所有文字/图形一律由 Python 栅格化为 RGBA PNG，再由 `overlay` 合成。
* ASR：`/opt/homebrew/Cellar/openai-whisper/20250625_3/libexec/bin/python`（独立解释器，
  含 torch 2.10 + MPS）。**不得**在主 venv 里导入 whisper；一律子进程调用。

### CLI 命令（稳定契约）

| 命令 | 作用 | 产物 |
|---|---|---|
| `fixtures build --specs DIR --out DIR [--only ID...]` | 合成 TTS 校准视频 | `clips/<id>/<id>.mp4` + `ground_truth.json` |
| `baseline run --input F --out DIR` | 透传/重编码基线 | `baseline_report.json` + MP4 + logs |
| `asr run --video F --out F.json [--model medium] [--device cpu\|mps]` | 转录 + 词级时间 | Transcript JSON |
| `vision analyze --video F --out F.json` | 人脸/镜头/安全区 | Vision JSON |
| `anchors build --transcript T --vision V --out A.json` | 语义锚点 | Anchors JSON |
| `plan --anchors A --vision V --canvas WxH --out P.json` | 导演决策 | Plan JSON |
| `ir build --transcript T --vision V --anchors A --plan P --source F --out IR.json` | Motion IR | IR JSON |
| `render --ir IR.json --source F --out MP4 [--preview] [--fps N]` | 渲染 | MP4 + `render_manifest.json` |
| `verify run --ir IR.json --render MP4 --truth G.json --out M.json` | 度量 | Metrics JSON |
| `run --fixture ID --outdir DIR` | 端到端（fixture 真值） | 全部产物 |
| `run --input F --outdir DIR` | 端到端（无真值时 timing=NOT RUN） | 全部产物 |

统一退出码：`0` 成功；`2` 参数错误；`3` 契约校验失败（E_IR_INVALID）；
`4` 媒体失败（E_FFMPEG/E_INPUT_MISSING/E_NO_VIDEO）；`5` 生成失败（E_ASR_FAILED）；
`6` 布局失败（E_LAYOUT_NO_SPACE）；`7` 渲染失败；`8` 度量失败。

---

## 2. 错误码

| code | 语义 | 责任模块 |
|---|---|---|
| `E_INPUT_MISSING` | 输入文件不存在 | media |
| `E_NO_VIDEO` | 无视频流 | media |
| `E_FFMPEG` | ffmpeg/ffprobe 非零退出 | media |
| `E_TTS` | 合成语音失败 | fixtures |
| `E_ASR_FAILED` | ASR 子进程失败/输出不可解析 | asr |
| `E_IR_INVALID` | IR 未通过 schema 校验 | ir |
| `E_LAYOUT_NO_SPACE` | 无可用安全区（应降级为静态布局而非崩溃） | design |
| `E_RENDER_FAILED` | 渲染失败 | render |
| `E_VERIFY_FAILED` | 度量不可计算（缺帧/缺真值） | verify |

失败必须写 `error.code` + `cmd` + `stderr` 尾部到日志，不允许静默降级。

---

## 3. 数据契约

### 3.1 Transcript — `semotion.transcript/1`

```json
{
  "schema": "semotion.transcript/1",
  "source": {"path": "...", "sha256": "...", "duration_s": 24.2, "width": 1080, "height": 1920, "fps": 25.0},
  "language": "zh",
  "model": {"name": "whisper", "size": "medium", "device": "cpu", "version": "20250625"},
  "segments": [{"id": "s0", "start_s": 0.0, "end_s": 3.1, "text": "大家好", "avg_logprob": -0.31, "no_speech_prob": 0.01}],
  "words": [
    {"id": "w0", "segment_id": "s0", "text": "大家", "start_s": 0.52, "end_s": 0.94,
     "probability": 0.91, "char_start": 0, "char_end": 2}
  ],
  "provenance": {"created_utc": "...", "command": "...", "truth_class": "MACHINE_WEAK"}
}
```

* `words[].probability` 必须原样保留；`truth_class` 恒为 `MACHINE_WEAK`（ASR 不得自证真值）。
* `start_s/end_s` 为秒（float，6 位小数）；帧号由消费方按 fps 换算，不写入本层。

### 3.2 Vision — `semotion.vision/1`

```json
{
  "schema": "semotion.vision/1",
  "source": {"path": "...", "sha256": "...", "width": 1080, "height": 1920, "fps": 25.0, "n_frames": 605},
  "detector": {"name": "yunet", "model": "face_detection_yunet_2023mar.onnx", "score_threshold": 0.6},
  "faces": [{"frame": 12, "t_s": 0.48, "box": [x, y, w, h], "score": 0.93}],
  "face_tracks": [{"id": "f0", "frames": [{"frame": 0, "t_s": 0.0, "box": [...], "score": 0.9}],
                   "bbox_union": [x, y, w, h], "coverage": 0.97}],
  "shots": [{"index": 0, "start_s": 0.0, "end_s": 24.2, "start_frame": 0, "end_frame": 605}],
  "safe_zones": {
    "title":   [0.08, 0.08, 0.84, 0.14],
    "caption": [0.08, 0.78, 0.84, 0.12],
    "card":    [0.06, 0.30, 0.88, 0.42]
  },
  "provenance": {"created_utc": "...", "command": "...", "scanned_fps": 25.0}
}
```

* 坐标为**像素**的 `[x, y, w, h]`，整数；`safe_zones` 为**归一化** `[x, y, w, h]`。
* 采样策略：默认每 `ceil(fps/5)` 帧检测一次（≥5 次/秒），人脸框按最近邻插值到帧。
* 无检测结果时 `faces=[]`、`face_tracks=[]`，消费方必须降级为静态安全布局。

### 3.3 Anchors — `semotion.anchors/1`

```json
{
  "schema": "semotion.anchors/1",
  "source_sha256": "...",
  "anchors": [{
    "id": "a03",
    "kind": "keyword",            // keyword | number | list_item | card | comparison | process | conclusion
    "text": "其实是节奏",
    "start_s": 6.336458, "end_s": 7.51425,
    "confidence": 0.82,
    "source": "ASR_ALIGNED",      // ASR_ALIGNED | SYNTH_KNOWN | HUMAN | MACHINE_WEAK
    "segment_id": "s3",
    "word_ids": ["w11", "w12"]
  }],
  "provenance": {"created_utc": "...", "method": "asr_word_spans+rule_classifier", "n_anchors": 1}
}
```

* 时间**只能**来自 Transcript 的词/段区间或 SYNTH_KNOWN 真值，禁止 LLM 编造时间戳。
* `confidence < 0.6` 的锚点不得驱动精确动效；调用方应降级（保守效果或不加动效）。

### 3.4 Plan — `semotion.plan/1`（导演决策）

```json
{
  "schema": "semotion.plan/1",
  "style": "smart_balanced",
  "canvas": {"width": 1080, "height": 1920, "fps": 25},
  "decisions": [{
    "id": "d03",
    "anchor_id": "a03",
    "effect": "keyword",          // none | keyword | number | list | card
    "emphasis": "high",           // low | medium | high
    "reason": "keyword_density+contrast_marker",
    "confidence": 0.82,
    "decision_model": "rules-v1" , // 或 "llm:<model>" —— 冻结进 IR
    "anchor_ids": ["a03"],       // list 动效为多个锚点
    "content": {"text": "其实是节奏", "items": null},
    "window": {"start_s": 6.336, "end_s": 7.514}   // 由决策层给出的时间窗（秒，含尾部停留）
  }],
  "notes": {"skipped": [{"anchor_id": "a07", "reason": "plain_speech_no_motion"}],
            "policy": "P1 effect set; plain speech may get zero motion",
            "budget": {"max_new_per_10s": 3, "min_gap_s": 0.9, "max_coverage": 0.58, "min_confidence": 0.6},
            "overlay_coverage": 0.41}
}
```

### 3.5 Motion IR — `semotion.motion-ir/1`（核心可编辑工程）

```json
{
  "schema": "semotion.motion-ir/1",
  "ir_version": "1.0.0",
  "source": {"path": "abs", "sha256": "...", "width": 1080, "height": 1920,
             "fps": 25.0, "duration_s": 24.2, "n_frames": 605,
             "timebase": {"num": 1, "den": 25}},
  "canvas": {"width": 1080, "height": 1920, "fps": 25.0},
  "style": "smart_balanced",
  "semantic": {
    "transcript_sha256": "...", "anchors_sha256": "...",
    "decisions_sha256": "...",
    "anchors": [ /* Anchors 数组原样冻结 */ ]
  },
  "tracks": [
    {"id": "t_content", "z": 20, "elements": ["el_d03"]},
    {"id": "t_caption", "z": 10, "elements": ["cap_0"]}
  ],
  "elements": [{
    "id": "el_d03", "type": "keyword",
    "anchor_ref": "a03",
    "window": {"start_s": 6.336, "end_s": 7.514, "start_frame": 158, "end_frame": 188},
    "layout": {"mode": "static_safe", "zone": "card", "box": [64, 576, 952, 300],
               "avoid_face": true, "face_safe_margin_px": 32},
    "style": {"font_family": "NotoSansSC", "font_path": "assets/fonts/NotoSansSC-Regular.otf",
              "font_size_px": 88, "weight": "bold", "color": "#F5F7FA",
              "accent": "#F2B233", "align": "center", "bg": "rgba(12,16,28,0.72)",
              "radius_px": 28, "padding_px": 36},
    "motion": {"enter": {"type": "pop_up", "dur_s": 0.26, "ease": "out_cubic", "scale_from": 0.86},
               "exit": {"type": "fade", "dur_s": 0.18, "ease": "in_quad"}},
    "content": {"text": "其实是节奏"},
    "provenance": {"decision_model": "rules-v1", "prompt_summary": "", "confidence": 0.82,
                   "effect_params_sha256": "..."}
  }],
  "output": {"width": 1080, "height": 1920, "fps": 25.0, "video_codec": "h264",
             "pix_fmt": "yuv420p", "audio_codec": "aac", "audio_bitrate": "192k",
             "loudness_target_db": null},
  "provenance": {"generated_utc": "...", "tool_versions": {...}, "host": "macOS-26.2-arm64",
                 "ir_sha256": "<对除本字段外内容规范化 JSON 的 sha256>"}
}
```

**IR 不变量（校验器必须强制）：**

1. `elements[].window.start_frame == round(start_s*fps)`，`end_frame == round(end_s*fps)`，且 `0 ≤ start < end ≤ n_frames`。
2. `elements[].anchor_ref` 必须存在于 `semantic.anchors`；时间窗必须落在锚点 ±250ms 容差内（除非 `type="caption"`）。
3. `layout.box` 完全位于 canvas 内（`x≥0, y≥0, x+w≤W, y+h≤H`）。
4. 任何 `avoid_face=true` 的元素，其 box 与人脸 `bbox_union` 交集面积必须 ≤ 人脸框面积 1%，或元素被标 `allow_face_overlap: true` 且记录理由。
5. `tracks[].elements` 必须引用存在的 element id；每个 element 恰好属于一个 track。
6. 元素时间窗互相重叠且 **非设计** 的，必须在 IR 中显式声明 `collision_ok: [idA,idB]`；否则校验器报 E_IR_INVALID。
7. `ir_sha256` 计算方式固定（见 `ir.py::compute_ir_hash`），用于可复现性核对。
8. 文本 `content.text` 必须能在 `semantic.anchors[].text` 或 transcript 文本中找到（防捏造事实）。

### 3.6 Render manifest — `semotion.render/1`

```json
{
  "schema": "semotion.render/1",
  "ir_sha256": "...",
  "output": {"path": "abs", "sha256": "...", "size_bytes": 123, "duration_s": 24.2,
             "n_frames": 605, "width": 1080, "height": 1920, "fps": 25.0},
  "audio": {"pcm_sha256": "...", "source": "passthrough_decode_encode", "delta_vs_source_s": 0.0},
  "overlay": {"fps": 25.0, "frames_dir": "...", "rendered_frames": 42, "empty_frames": 0},
  "commands": ["ffmpeg ..."],
  "tool_versions": {"ffmpeg": "...", "ffprobe": "..."},
  "created_utc": "..."
}
```

### 3.7 Metrics — `semotion.metrics/1`

```json
{
  "schema": "semotion.metrics/1",
  "gate": "G1",
  "input_sha256": "...", "ir_sha256": "...", "render_sha256": "...",
  "timing": {
    "samples": 14, "source": "SYNTH_KNOWN", "groups": {"keyword": 2, "list_item": 6},
    "p50_ms": 21, "p90_ms": 44, "max_ms": 80, "errors_ms": [21, 18, 44],
    "threshold_p90_ms": 100, "verdict": "PASS",
    "note": "independent assembly truth; ASR not used to define targets"
  },
  "position": {
    "face_overlap_rate": {"value": 0.0, "samples": 120, "threshold": 0.0, "verdict": "PASS"},
    "out_of_bounds_rate": {"value": 0.0, "samples": 168, "threshold": 0.0, "verdict": "PASS"},
    "unintended_collision_rate": {"value": 0.0, "samples": 40, "threshold": 0.01, "verdict": "PASS"}
  },
  "render": {
    "duration_delta_s": 0.0, "audio_identity": "decoded_pcm_equal",
    "determinism": {"runs": 2, "frame_hashes_equal": true, "max_pixel_diff": 0.0, "verdict": "PASS"}
  },
  "verdicts": {"timing": "PASS", "position": "PASS", "render": "PASS"},
  "evidence": {"commands": [], "logs": [], "artifacts": {}}
}
```

`verdict` ∈ `PASS | FAIL | NOT RUN | BLOCKED`（**只有这三种失败态**，禁止出现 "probably"）。

---

## 4. Python 模块接口

```python
# media.py
info(path) -> MediaInfo
probe(path) -> dict
run(argv, *, timeout=3600, log: Path|None=None) -> str
sha256_file(path) -> str
audio_pcm_sha256(path, *, rate=48000, channels=2, log=None) -> str
video_frame_hashes(path, *, width=320, height=180, log=None) -> list[str]
frame_diff(a, b, *, width=320, height=180) -> dict
extract_audio(path, out_wav, *, rate=16000, channels=1, log=None) -> Path

# asr.py
transcribe(video: Path, out_json: Path, *, model="medium", device="cpu",
           language="zh") -> dict           # schema semotion.transcript/1

# vision.py
analyze(video: Path, out_json: Path, *, fps_scan=5.0, model: Path|None=None) -> dict
face_tracks(vision: dict) -> list[dict]

# anchors.py
classify(transcript: dict) -> dict          # semotion.anchors/1
load_truth(ground_truth_json) -> list[dict] # SYNTH_KNOWN 事件

# semantic.py
decide(anchors: dict, vision: dict, canvas: dict, *, style="smart_balanced") -> dict  # plan

# design.py
layout(plan_element: dict, canvas: dict, vision: dict, occupied: list[dict]) -> dict  # box + zone
safe_area_boxes(canvas, vision) -> dict

# raster.py
render_element_png(element: dict, canvas: dict, out_dir: Path, *, font_dir="assets/fonts") -> list[Path]
     # 返回元素窗口内逐帧 RGBA PNG 列表（z 顺序由 render 组合）

# ir.py
build(transcript, vision, anchors, plan, source: Path, canvas: dict) -> dict
validate(ir: dict) -> list[str]             # 空列表 = 合法
compute_ir_hash(ir: dict) -> str

# render.py
render(ir: dict, source: Path, out: Path, *, preview=False, workers=1) -> dict  # semotion.render/1

# verify.py
metrics(ir, render_manifest, truth_json=None, *, gate="G1") -> dict             # semotion.metrics/1
determinism(ir, source, out_dir_a, out_dir_b) -> dict
```

## 5. 并行开发边界（避免覆盖）

| 负责人 | 独占写路径 |
|---|---|
| speech-vision | `src/semotion/asr.py`, `src/semotion/vision.py`, `src/semotion/anchors.py`, `tests/test_asr_vision.py`, `tests/fixtures/specs/*`, `assets/models/*` |
| motion-render | `src/semotion/ir.py`, `src/semotion/design.py`, `src/semotion/raster.py`, `src/semotion/effects/*`, `src/semotion/render.py`, `tests/test_ir_render.py`, `assets/fonts/*` |
| lead | `src/semotion/media.py`, `src/semotion/baseline.py`, `src/semotion/fixtures.py`, `src/semotion/semantic.py`, `src/semotion/verify.py`, `src/semotion/cli.py`, `docs/*`, `PROGRESS.md`, `reports/*`, `ENVIRONMENT.md`, `.dsh/skills/*` |

共享只读：`docs/INTERFACES.md`、`artifacts/env/env_probe.json`、`tests/fixtures/clips/*`。

## 6. 确定性策略

* 渲染命令固定参数（`-preset medium -crf 18 -pix_fmt yuv420p` 等），不依赖时间/随机数。
* 随机性仅允许在 fixture 背景生成，且必须固定 `seed`。
* 可复现性定义：**相同 IR + 相同素材 + 相同工具版本** → 解码帧一致（像素哈希或明确容差），
  不以 MP4 字节一致为唯一标准（见 `verify.determinism`）。
* LLM 输出只在其被冻结进 IR 后才有复现意义；IR 记录 `decision_model` 与 `prompt_summary`。
