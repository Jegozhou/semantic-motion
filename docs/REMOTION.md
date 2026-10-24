# Remotion 渲染后端 + LLM 导演：接口契约

契约版本：`remotion-1.0.0`｜状态：G5（进行中）
关联：`docs/INTERFACES.md`（IR 契约不变；本文件只定义新后端的边界）

---

## 1. 为什么加这个后端

现有后端 `pillow_ffmpeg` 把文字栅格化成 PNG 再 overlay，胜在确定性强、零依赖；
不足是动效表现力有限（只有淡入/缩放/位移），且动效选择由规则引擎决定。

新后端 `remotion` 用 React 组件描述动效，**动效类型由 LLM 选择**，时间与文案来自词级转录与字幕。
两条后端共用同一份 Motion IR 与同一套度量，可通过 `render.backend` 切换与对比。

**许可提示（必须记录）**：Remotion 为 [Remotion License](https://github.com/remotion-dev/remotion/blob/main/LICENSE.md)，
个人 / ≤3 员工公司 / 非营利可免费（含商用），更大规模的营利组织需购买 Company License。
本项目把它作为可选渲染后端，若不符合免费条件需替换（如 Revideo）或购证。

## 2. 渲染后端接口

```python
# src/semotion/remotion_bridge.py
props(ir: dict, *, work_dir: Path) -> dict          # semotion.remotion-props/1
render(ir: dict, source: Path, out: Path, *,
       preview: bool = False, chrome: str | None = None) -> dict   # 与 render.render 同形状的 manifest
```

* 输出：MP4（h264 + aac），分辨率/帧率/时长与 IR 一致；音频来自源视频。
* `preview=True`：`--scale 0.5` + crf 28（Remotion 侧参数），manifest 记录实际分辨率。
* 失败：`media.MediaError("E_RENDER_FAILED", ...)`，保留 Node/Chrome 日志尾部。
* 不执行 LLM 生成的代码：props 只含数据；组件来自固定白名单。

## 3. Props schema — `semotion.remotion-props/1`

```json
{
  "schema": "semotion.remotion-props/1",
  "canvas": {"width": 1080, "height": 1920, "fps": 25, "durationInFrames": 605},
  "source": {"file": "source.mp4", "audio": true},
  "captions": [
    {"startFrame": 12, "endFrame": 38, "lines": ["大家好"], "words": [{"text": "大家好", "startFrame": 12, "endFrame": 38}]}
  ],
  "elements": [
    {
      "id": "el_d01",
      "effect": "keyword",
      "startFrame": 83,
      "endFrame": 105,
      "box": [408, 1035, 264, 232],
      "content": {"text": "痛点", "value": null, "prefix": "", "suffix": "", "decimals": 0, "items": []},
      "style": {"accent": "#F2B233", "fg": "#F5F7FA", "bg": "rgba(12,16,28,0.88)",
                "fontPx": 88, "radius": 28, "align": "center", "weight": 700},
      "motion": {"enter": "spring_pop", "enterDurFrames": 8,
                 "exit": "fade", "exitDurFrames": 5}
    }
  ]
}
```

**约束**

1. 所有时间都是**帧号**（绝对帧），由 IR 的 `window.start_frame/end_frame` 直接映射，不由 LLM 生成。
2. `effect` ∈ `keyword | number | list | card`（白名单；新增须改契约版本）。
3. `content.value` 为数字真值：组件必须**精确收敛**到该值，不得四舍五入丢失语义。
4. 文本来自字幕/锚点；人工改写须在 IR 中 `content.edited=true`（沿用既有规则）。
5. `box` 与 IR 的 `layout.box` 一致；组件内部不得溢出（超出即判 `raster_containment` 失败）。

## 4. LLM 导演接口

```python
# src/semotion/llm_director.py
build_prompt(transcript: dict, anchors: dict, vision: dict, canvas: dict) -> str
validate_decisions(doc: dict, anchors: dict, transcript: dict) -> list[str]   # 空=合法
to_plan(doc: dict, anchors: dict, canvas: dict) -> dict                        # semotion.plan/1
caption_cues(transcript: dict, canvas: dict, *, max_chars: int = 22,
             max_lines: int = 2) -> list[dict]                                 # 字幕轨
```

### LLM 输出（冻结进 IR 的 `semantic.decisions`）

```json
{
  "schema": "semotion.llm-decisions/1",
  "decision_model": "deepseek-v4.1-flash",
  "prompt_summary": "选动效：只给重要信息；数字必须精确；每 10s ≤4 个；不得遮脸；不得跨切镜",
  "decisions": [
    {"anchor_id": "a03", "effect": "number", "emphasis": "high",
     "reason": "用户在讲可量化结论（九成）", "hold_s": 0.45, "confidence": 0.86,
     "params": {"accent": "#F2B233", "enter": "count_up"}}
  ],
  "skipped": [{"anchor_id": "a01", "reason": "普通叙述，加了会打断"}]
}
```

### 校验规则（违反即拒绝，不静默修补）

| # | 规则 |
|---|---|
| 1 | `effect` 必须在白名单内 |
| 2 | `anchor_id` 必须存在于 anchors |
| 3 | 时间窗由锚点推导：`start = anchor.start`，`end = min(anchor.end + hold_s, next_element_start - 0.04, duration)`；`hold_s ≤ 0.8` |
| 4 | 同一时间窗只能有一个元素（重叠报错，除非显式 `collision_ok`） |
| 5 | 元素数量：每 10 s ≤ 4；覆盖率 ≤ 58% |
| 6 | `number` 必须能解析出数值，且与 `content.numeric` 一致 |
| 7 | 不得引用 `kind=plain` 的锚点 |
| 8 | 有脸时必须给出 `params.avoid_face=true`（布局阶段强制） |

## 5. 字幕轨

* 由 `transcript.words` 生成：按标点/停顿切句 → 每行 ≤ `max_chars`（9:16 默认 22，16:9 默认 30）→ 每屏 ≤ `max_lines`。
* 每条 cue 记录 `startFrame/endFrame/lines/words`（词级高亮用）。
* 字幕位置固定在底部安全区（`safe_zones.caption`），不参与动效碰撞；若与元素重叠，元素让位（元素布局已避开 caption 区）。
* 字幕同样受"不可遮脸"约束：底部安全区与人脸框冲突时上移一行并记录 warning。

## 6. 度量复用

`verify.metrics()` 不需要改动：它以 IR 的元素框为测量区域，从渲染像素反查 onset / 遮脸 / 越界。
新增一项：

```json
"captions": {
  "cues": 42, "checked": 42,
  "onset_p50_ms": 18.0, "onset_p90_ms": 40.0,
  "verdict": "PASS"
}
```
其中 onset = 字幕条切换的首帧时间 vs 该 cue 首个词的 `start_s`。

## 7. 并行边界

| 负责人 | 写域 |
|---|---|
| lead | `remotion/src/**`（组件 + composition + Root）、`src/semotion/remotion_bridge.py`、`src/semotion/verify.py`（字幕度量）、`scripts/*`、`docs/*`、`reports/*` |
| llm-director | `src/semotion/llm_director.py`、`tests/test_llm_director.py`、`artifacts/llm/*.json`（LLM 决策样例） |

共享只读：`src/semotion/{media,ir,anchors,semantic}.py`、`tests/fixtures/clips/*`、`artifacts/g1/**`。
