# ALIGNMENT.md — 字符级强制对齐与字幕构建（P2）

> 模块：`src/semotion/align.py`（`semotion.alignment/1`）、`src/semotion/captions.py`（`semotion.captions/1`）
> 证据：`artifacts/p2/align/`；测试：`tests/test_align.py`（28 项，`pytest tests/test_align.py -q` → 全绿）
> 真值类别：**SYNTH_KNOWN（逐单元 TTS 拼接，非真人）**；输出自身标记 **MACHINE_WEAK**（不是人工标注）。

## 0. 一页结论

| 指标 | 数值 | 样本量 | 性质 |
|---|---|---|---|
| `fx_a_zh_portrait` 字符起音误差 P90 | **7.04 ms**（P50 4.47、max 7.96、bias +3.74 ms） | n=14 单元 | 验收数字，**n<20 仅作探索性** |
| 旧管线同一素材（ASR 词起点） | **181.1 ms**（P50 114、max 189） | n=14 | 对比基线 |
| 全部 10 个 SYNTH_KNOWN 素材合计 | **P50 2.75 / P90 7.56 / P95 8.13 ms**，max 108.17，bias +4.97 ms | **n=147 单元** | 开发集 P90（可宣称范围：仅合成集） |
| 旧管线合计 | P50 132.35 / **P90 226.43 ms**，max 556 | n=147 | 对比基线 |
| 无音频基线（按段均分） | P90 151.76 ms | n=147 | 负对照 |
| **打乱 onset 包络**（负对照） | **P90 98.88 ms** | n=147 | 证明指标会失败 |
| `zhengqinwen_25s`（真实中文）单调性违规 | **0**（123 字符 / 123 音节） | 123 | 可验证事实 |
| 真实语音音节时长 | **median 150 ms**，p10 120 / p90 200 / min 90 / max 280 ms | 123 | 合理性检查（无人工真值） |
| 预测起音与真实包络显著峰 | 中位距离 0 ms，97.6% 在 50 ms 内，point-biserial r=0.432 | 123 | 合理性检查 |
| 字幕 | 两片素材 `validate()` 均 0 违规；CJK 行宽 ≤20（fx_a 最大 15、真实素材 20），min display 2.12 s（fx_a）/ 1.32 s（真实素材）；**按 ASR 段分 cue**：真实素材 10 条、cue0=奥运会是我的梦想没错（task-3b） | — | 规则验收 |
| 攻击点细化（ablation，默认关闭） | P90 16.01 ms / bias −11.09 ms / max 20 ms | n=147 | 权衡记录，见 §6.4 |

**一句话**：在本仓库唯一可量化的真值（合成 TTS 拼接）上，字符起音 P90 从旧管线的 ~200 ms 降到 **7.56 ms（n=147）**；在真实中文上无法测误差（无人工标注），可报告的只有单调性 0、音节时长中位 150 ms、起音与包络峰高度吻合，以及一张供人工目检的时间轴图。

## 1. 问题：为什么 Whisper 的词时间不能直接用于字级动效

现有 `src/semotion/asr.py` 用 faster-whisper `word_timestamps=True`，再做一个"能量起音吸附"。问题有三层：

1. **段起点本身会偏**：`inputs/zhengqinwen_25s.mp4` 的 `s0` 报 `start_s=0.0`，但真实语音从 **0.50 s** 才开始（VAD 与波形一致）。旧管线把首字 `奥` 标成 `0.0–0.66 s`（一个音节被拉长到 660 ms），字幕/动效提前 ~0.5 s。
2. **中文"词"不是字符**：这段 25 s 音频转写出 **98 个"词"**，而文本是 **123 个汉字**（本题：一字≈一音节）。多字词内部没有任何时间信息，按字数均分（旧字幕做法）把误差摊到每个字上。
3. **能量吸附会吸到下一个音节**：词内能量峰并不总在起音处，粗吸附可能把锚点推到相邻音节，误差 100–300 ms 且不稳定。

`fx_a_zh_portrait` 上同样可见：旧管线第一段报 `start_s=0.64`、首个"词"`大家好,` 从 0.64 开始，而拼接真值是 **0.50**；14 个单元上旧管线 P90 = **181 ms**。这个误差会直接变成字幕漂移。

## 2. 方法（`semotion.align`）

确定性、纯 numpy、离线，不重新训练、不下载模型；**复用** `asr.py` 的解码与缓存（`asr.transcribe` 命中缓存时只读 JSON），不修改 `asr.py`/`anchors.py`/`cli.py`。

### 2.1 音频前端

- 16 kHz 单声道 PCM（经 `semotion.media.extract_audio`，不裸调 ffmpeg）。
- 25 ms 帧 / 10 ms 步长；Hann 窗；40 维 log-mel（80–7600 Hz，三角形滤波器，numpy 实现）。
- **起音强度 onset strength** = 0.62×log-mel 正向谱通量 + 0.22×低频（<800 Hz）通量 + 0.16×能量上升；三项各自在语音帧上做 5–95 分位鲁棒归一化，3 帧平滑（保峰、去单帧抖动）。
- **VAD**：能量 dB 双阈值滞回（高阈值建段、低阈值延伸），填 <90 ms 小孔、丢 <60 ms 碎片 → 语音岛（speech islands）。不依赖任何模型。

### 2.2 文本 → 音节单元（`tokenize_text`）

| 文本类型 | 规则 | 例 |
|---|---|---|
| 汉字 / 假名 / 谚文 | 1 字 = 1 音节 | `今` → 1 |
| 阿拉伯数字串 | 中文读法的音节数（文档化启发式） | `25`→二十五=3；`100`→一百=2；`1,000`→一千=2；`3.5`→三点五=3；`90%`→百分之九十=5；`-5`→负五=2 |
| 4 位年份 | 逐位读（`1900–2100`，或后接 年/月/日/号） | `2024`→二零二四=4；`2024年`→5 |
| 拉丁词（中文文本中） | 1 个起音（接口约定） | `AI` → 1 |
| 英文（`language=en`） | 元音组启发式音节数 | `align`→2，`every`→3 |
| 标点/空白/符号 | **不占时长**，挂到前一 token 的 `trailing_punct` | `，。！？` 等 |

### 2.3 约束：停顿切块 + 段边界软锚

1. 相邻语音岛之间 `gap < pause_min_s (140 ms)` 先合并（句内换气不应当作硬边界）；`≥140 ms` 的停顿成为**硬块边界**。
2. **按 ASR 段分配音节**（关键修复）：每个段的音节数由文本确定，只分配到该段时间范围覆盖的语音岛上，按重叠时长比例分配，并在标点处吸附。这样"某段少算一个音节"不会把下一个短语的首字吸到上一段末尾（早期版本正是这样在 `fx_face_pan` 上出现 145 ms 误差）。
3. ASR 段起止只作为**软锚**（容差 ±100/120 ms，权重 0.35），语音岛首尾作为强锚；因此 Whisper 的段偏 0.5 s 时（`s0`）以声学证据为准，不会被拉回 0.0。

### 2.4 单调 DP

对每个块（syllable run），在 `[island_start−30 ms, island_end+30 ms]` 的帧网格上求解 N+1 个边界（N 个音节起点 + 末音节终点）：

```
cost = −w_onset·onset[t_i]                      # 起音证据（末边界不加）
     + w_dur·(ln(d_i / median_syllable))²         # 时长先验，median 由岛内自适应
     + anchor 罚项（段起点/终点、岛首/岛尾）
约束：d_i ∈ [60 ms, 1.3 s]，t 严格递增，边界只能落在语音帧
```

动态规划逐层对帧做向量化，回溯得到全部边界；不可行（岛太短/音节太多）时退化为均匀分布并写 warning。`w_onset=1.0`、`w_dur=1.2`，是最小验证集上的固定值，不逐文件调参。**dp 保证输出严格单调**，这是 p90 误差之外最容易验收的硬约束。

### 2.5 字符 / 词输出

- 汉字 token：`chars[]` 一字一条（`start/end/conf/approx=false`）。
- 数字/拉丁 token：`words[]` 保留整体跨度；`chars[]` 按比例摊开并标 `approx=true`（下游字幕必须用词跨度，不能信近似字跨度）。
- 置信度 = 0.65×峰值强度分位 + 0.35×时长合理性（log-normal），`< 0.35` 计入 `low_confidence_syllables`。
- 每文件质量报告包含：单调性违规、音节时长分布、起音包络分位、与显著峰距离、point-biserial r、段锚偏差、低置信音节数、块数/可行块数。

### 2.6 起音细化（可选，默认关闭）

`syllable.peak_s` 记录 DP 选中的包络峰；`refine_onsets=1` 时再向回找能量谷→上升沿（辅音起音）。在合成集上它把整体起音提前 ~12 ms，P90 从 7.56 变差到 16.01 ms，因此**默认关闭**；它真正有价值的场景是长清辅音（本集合中的 `三`/`核` 起音被峰法估晚 76–108 ms，细化后 max 降到 20 ms）。取舍见 §6.4，参数保留以便在有真人标注后重选。

## 3. 合成真值上的实测误差

### 3.1 验收素材 `fx_a_zh_portrait`

| 方法 | P50 | **P90** | max | bias | n |
|---|---|---|---|---|---|
| **本模块（字符起音）** | 4.47 ms | **7.04 ms** | 7.96 ms | +3.74 ms | 14 |
| 旧管线（最近 ASR 词起点） | 114.0 ms | 181.1 ms | 189.5 ms | +108.7 ms | 14 |
| 旧管线（词内按字数均分，字幕旧做法） | 114.0 ms | 181.1 ms | 189.5 ms | +108.7 ms | 14 |

直接修好的例子：旧 `s0.start_s=0.64`、首词 `大家好,`@0.64；真值首单元 0.50；新首字 `大`@**0.50**。

> ⚠️ **n=14 ≠ 稳健 P90**。按仓库规则（`PROGRESS.md` §1.4：≥50 独立事件才可宣称 P90），下面 147 个单元才是可引用的数字；`fx_a` 的 7.04 ms 只是同一素材的验收对照。

### 3.2 全部 SYNTH_KNOWN 素材（10 个，147 个独立单元）

| 素材 | 单元数 | 本模块 P90 | 旧管线（词起点）P90 | 旧管线（词内均分）P90 |
|---|---|---|---|---|
| fx_a_zh_portrait | 14 | **7.04 ms** | 181.10 ms | 181.10 ms |
| fx_b_zh_landscape | 8 | 6.74 | 226.94 | 226.94 |
| fx_c_zh_cuts | 7 | 6.17 | 248.98 | 115.64 |
| fx_d_zh_portrait | 20 | 7.50 | 214.55 | 182.02 |
| fx_e_zh_landscape | 17 | 7.31 | 273.79 | 273.79 |
| fx_face_pan | 9 | 6.02 | 204.42 | 204.42 |
| fx_face_zh_portrait | 12 | 7.37 | 212.05 | 212.05 |
| fx_t1_zh_phrase | 20 | 7.61 | 209.33 | 181.41 |
| fx_t2_zh_phrase | 20 | 7.63 | 212.66 | 199.10 |
| fx_t3_zh_phrase | 20 | 7.89 | 247.87 | 247.87 |
| **合计** | **147** | **P50 2.75 / P90 7.56 / P95 8.13，max 108.17，bias +4.97** | P50 132.35 / **P90 226.43**，max 555.96 | P50 127.37 / **P90 199.58** |

逐单元明细（含每单元真值/预测/误差）在 `artifacts/p2/align/fixtures_score.json`。

误差的最大值 108 ms 来自 `fx_e` 的 `三条…`：`三`(sān) 的清擦音让谱通量峰落在元音上，属于辅音起音系统性偏晚；P90 不受影响。这就是 §6.4 的场景。

### 3.3 负对照：指标会失败，且差异显著

| 输入 | P90（n=147） | 说明 |
|---|---|---|
| 正常 onset 包络 | **7.56 ms** | 拟合 |
| **打乱 onset 包络帧序**（同一 VAD、同一 DP） | **98.88 ms** | 声学证据被破坏 → 误差放大 13 倍 |
| 均匀按段分摊（无音频） | 151.76 ms | 只有文本/时长先验 |
| 旧 ASR 词时间 | 226.43 ms | 现网基线 |

三者都远差于 7.56 ms，说明 P90 不是"空转指标"。

## 4. 真实中文语音的合理性检查（`inputs/zhengqinwen_25s.mp4`）

**没有人工标注**，因此不报误差；以下全部是可复核的事实/合理性指标（`artifacts/p2/align/evidence.json`）：

| 检查 | 结果 |
|---|---|
| 单调性违规 | **0**（123 字，123 音节，16 个语音块） |
| 音节时长 | min 90 / p10 120 / **median 150** / p90 200 / max 280 ms（Mandarin 预期 150–260 ms 的**下限**：该说话人 18.4 s 语音里 123 个音节，语速偏快） |
| 起音 vs 显著包络峰 | 中位 **0 ms**，p90 10 ms，**97.6% ≤50 ms**，98.4% ≤80 ms |
| 起音能量 vs 语音平均能量 | contrast **2.30×**；局部包络分位均值 0.861；point-biserial **r=0.432** |
| 低置信音节 | 0 |
| 段锚偏差（中位） | 起点 90 ms、终点 150 ms（Whisper 段本身的偏差量级） |
| 关键修复 | 旧：`s0` 起 0.0，`奥`@0.0；新：`奥`@**0.50**，`运`@0.66（与 Whisper 的第二个词巧合一致），`会`@0.82 |
| 字幕（task-3b） | **10 条 cue（每 ASR 段一条）**，`validate()=[]`，最大行宽 20，min display 1.32 s；cue0=`奥运会是我的梦想没错` |

**时间轴图（人工目检入口）**：`artifacts/p2/align/zhengqinwen_25s.timeline.png`
上排 = 波形 + 语音岛 + 段边界（蓝虚）+ 新起音（绿）+ Whisper 词起点（橙）；下排 = onset 包络 + 显著峰 + 起音竖线；底部 = 成片字幕 cue 条。`fx_a` 版本另有灰色菱形标出逐单元真值起音，可直接看到绿线与真值重合、橙线（旧管线）滞后。

**负对照（真实语音，无真值）**：把 onset 包络打乱后重新对齐，再用**原始**包络评估两者：拟合对齐 0.861 分位 / 2.30× 对比度；打乱后 0.480 分位 / 0.94×。说明对齐确实在跟踪真实包络，而不是只靠时长先验。（这是合理性对照，不是误差测量。）

## 5. 字幕（`semotion.captions`）

输入 `semotion.alignment/1`，输出 `semotion.captions/1` + SRT + ASS；所有时间**先行对齐到帧**（`start_s = start_frame / fps` 精确成立），SRT/ASS 由同一帧号导出（SRT 毫秒四舍五入 ≤0.5 ms）。

### 5.1 分组与断行规则（`DEFAULT_STYLE`）

| 项 | CJK | Latin |
|---|---|---|
| 行数 | ≤2 | ≤2 |
| 每行目标/上限 | 22 / **24** 字 | 38 / **42** 字符 |
| 短行下限 | 6 字（避免孤字行） | 8 字符 |
| cue 字符上限 | 44（=2×22） | 84 |
| 断行 | 优先短语/ASR 词边界，其次均衡 | 只在词边界断 |
| 结尾标点 | **不允许**（行尾标点剥离；不产生纯标点行） | 同上 |
| 最短显示 | **0.90 s**（按帧向上取整，25 fps 即 23 帧 = 0.92 s） | 同 |
| 小间隙 | `<0.18 s`：能并就并，不能并就延长前一条（不闪断） | 同 |
| 句末标点 | 不跨句合并；必要时延长显示而不是拼接 | — |
| 字体（ASS/Style 默认） | `Noto Sans SC` | `Inter` |

task-3b 新增两个分组策略（`build_captions(..., respect_segments=..., respect_words=...)`）：

* **`respect_segments`（默认自动开启：alignment 里的 words 带 segment id 即开）**：cue 绝不跨两个 ASR 段；段内不再用"停顿 ≥0.35 s"启发式断 cue（一个 ASR 段就是自然的字幕单位），但仍允许句末标点和 44 字硬上限切分。合并/最短显示也只在不跨段时合并，否则延长前一条的显示。
* **`respect_words`（默认开）**：cue/行绝不在一个 Whisper 词内部切开（`word.group_id`，由 `align.py` 依据 `transcript.words[]` 的字符跨度写入）。量化为 `quality.mid_word_cue_starts`，违反时 `validate()` 报错。

### 5.2 task-3b 前后对比（同一对齐、同一素材）

| 素材 | cue 数（前 → 后） | 跨词 cue 起点（前 → 后） | min display（前 → 后） | 最大行宽（前 → 后） |
|---|---|---|---|---|
| `fx_a_zh_portrait` | 10 → **7**（= ASR 段数） | 0 → **0** | 0.92 → **2.12 s** | 15 → 15 |
| `zhengqinwen_25s` | 6 → **10**（= ASR 段数） | 0 → **0** | 1.68 → **1.32 s** | 22 → **20** |

"前"= task-3 行为（`respect_segments=False, respect_words=False`），"后"= task-3b 默认。真实素材的首条 cue 从 20 字合并句 `奥运会是我的梦想没错但是我还有其他的梦想` 变为 `奥运会是我的梦想没错`；此前 `…没有结|束网球…` 的跨词 cue（Whisper 自己把「结」「束」分成两个词）也随之消失。跨词语句在真实素材上是 0/0 —— 不是守卫没生效，而是 Whisper 在这里本来就是单字词；守卫的有效性由合成负对照测试证明（构造一个跨 44 字硬上限的 3 字词：无守卫 1 条跨词 cue，有守卫 0 条，见 `tests/test_align.py::TestCaptionSegmentsAndWords`）。

### 5.3 zhengqinwen_25s 新 cue 列表（25 fps，见 `artifacts/p2/align/zhengqinwen_25s.captions.json`）

```
cue   start_frame  end_frame  duration  lines
cue0  12           48         1.44s     奥运会是我的梦想没错
cue1  48           138        3.60s     但是我还有其他的梦想等着我去实现我的目标
cue2  143          198        2.20s     我的梦想其实还远远没有结束
cue3  208          279        2.84s     网球最好的一点就是它永远都有比赛
cue4  279          319        1.60s     每一年我们都有比赛
cue5  326          393        2.68s     每一年都有更大的困难
cue6  393          441        1.92s     更大的山坡等着你去战胜
cue7  455          518        2.52s     因为如果我觉得太过于满足的话
cue8  518          567        1.96s     你可能就会停滞在一个位置
cue9  567          600        1.32s     没有办法向前进步
```

`validate(doc)` 检查：schema、帧精度、时长 ≥0.9 s、行数/行宽、行尾标点、纯标点行、cue 重叠、闪断间隙、跨 ASR 段混合、跨词 cue 起点；两片素材均为 `[]`。

## 6. 诚实的边界（必须连同数字一起引用）

1. **没有真人标注语料**。仓库里唯一真值是 SYNTH_KNOWN（逐单元 TTS 拼接，时间由 ffprobe 实测，非 ASR）。本报告所有 P90 都是**合成集**数字；真实中文只有合理性检查。不得把 7.56 ms 当成真人场景的精度。
2. **合成集偏容易**：单元之间有 0.2–0.4 s 停顿、TTS 吐字清晰、无背景人声/音乐。连续真人语音（无停顿、连读、轻声、口语填充词）预计更差；真实素材仅证明"不崩、单调、时长合理、与包络吻合"。
3. **真值是单元级，不是字级**：`score_unit_onsets` 只对每个单元的首字起音计分（每素材 n=7…20），所以"字符级精度"的严格含义是"每个单元首字定位精度"。内部字符的相对位置由 DP 约束保证单调与时长合理，但没有逐字真值可打分。
4. **辅音起音**：谱通量峰偏向元音，长清辅音（s/sh/x/h…）可能偏晚 50–100 ms（见 §3.2 的 108 ms 离群点）。可选 `refine_onsets=1` 修掉离群点（max 20 ms）但给全部音节带来 −11 ms 偏置（P90 16.0 ms）。无真人真值，无法决定哪个更贴近"人耳起点"，因此默认取合成集最优（关闭）。
5. **数字/年份读法**是启发式：`2024` 按年份逐位读（4 音节），`2024` 若被读成"两千零二十四"（6 音节）会错 2 个音节；括号、电话号、小数、百分比同理。文本与音频音节数不一致时 DP 会把误差摊到整个短语（置信度下降、`low_confidence_syllables` 增加）。
6. **VAD 依赖能量**：音乐底噪大、混响强、多人同时说话时语音岛可能粘连/断裂，停顿切块会退化；此时仍单调，但精度下降。本仓库真实素材已验证可用（16 块、0 低置信），但未做噪声压力测试。
7. **英文未端到端验证**：英文分支（词→音节加权 DP、≤42 字符断行）只有合成/单元测试，没有英文 SYNTH_KNOWN 真值素材；`fx_pd_real_16x9_*` 无 `ground_truth.json`，不能用来宣称精度。
8. **中文没有词典分词**：断行/cue 只依据标点、字数预算和 **ASR 词边界**（`word.group_id`）。因此可读性上限取决于 Whisper 的词切分：若 Whisper 本身把「结束」切成「结」「束」两个词，守卫无法知道它们是一个词（task-3b 的段模式已让这种跨 cue 切开消失，但同一 cue 内的换行仍以 ASR 词为界）。要更严格需要引入中文分词器（未做，避免新增依赖）。
9. **未接入现有流水线**：本轮只交付新模块（未改 `asr.py`/`anchors.py`/`cli.py`，按任务约定）。`anchors.py` 目前仍读 Whisper 词时间；要享受 7 ms 级对齐，需要在后续任务里把 `align.align_video(...)` 的输出接到锚点/字幕阶段（共享文件改动需 lead 统一安排）。
10. **确定性**：同一输入（相同音频、相同转录文本、相同参数）逐字节可复现；无随机数（负对照测试显式使用固定 seed 7）。DP 无时间依赖，`elapsed` 不进产物。

## 7. 复现

```bash
cd "/Volumes/ZHITAI/Semantic Motion"
export PYTHONPATH=src

# 1) 单元测试（28 项；不需要网络、不跑 ASR 模型，命中缓存即跳过）
.venv/bin/python -m pytest tests/test_align.py -q

# 2) 全量证据（对齐 JSON、字幕 JSON/SRT/ASS、时间轴 PNG、汇总 JSON）
.venv/bin/python artifacts/p2/align/evidence.py

# 3) 仅重算 147 单元误差表
.venv/bin/python artifacts/p2/align/eval_align.py

# 4) 可选：开启起音细化做消融
.venv/bin/python artifacts/p2/align/eval_align.py \
  --params '{"refine_onsets": 1}' --out artifacts/p2/align/work/refine.json
```

- Python 3.12（`.venv`），numpy/Pillow；**无新增依赖**。
- 真实素材对齐约 0.2 s / 25 s 音频；10 个合成素材 + 负对照全量证据约 6 s（缓存命中，无模型加载）。

## 8. 文件与契约

| 文件 | 内容 |
|---|---|
| `src/semotion/align.py` | `AudioFeatures`、`audio_features()`、`load_audio()`、`tokenize_text()`、`number_syllables()`、`english_syllables()`、`align_transcript()`/`align_video()`、`score_unit_onsets()`/`score_word_onsets()`、`validate()`、`save/load` |
| `src/semotion/captions.py` | `build_captions()`/`build_from_file()`、`export()`、`write_json/srt/ass()`、`validate()`、帧时间工具 |
| `tests/test_align.py` | 28 项：分词/数字读法、合成真值恢复、打乱包络负对照、均匀基线对照、VAD/包络、真实 fixture（命中缓存时）、10 项字幕规则/导出 |
| `artifacts/p2/align/fixtures_score.json` | 10 素材 × 147 单元的逐单元误差、旧管线对比、运行参数 |
| `artifacts/p2/align/evidence.json` | 一页汇总（本文件所有数字） |
| `artifacts/p2/align/old_pipeline_fx_a.json` | 旧管线逐单元误差（before/after 表） |
| `artifacts/p2/align/*.alignment.json` | 两片素材的 `semotion.alignment/1` 全文 |
| `artifacts/p2/align/*.captions.{json,srt,ass}` | 两片素材的字幕三件套 |
| `artifacts/p2/align/*.timeline.png` | 人工目检时间轴（真实语音 + fixture 真值） |
| `artifacts/p2/align/caption_samples.txt` | cue 列表 |
| `artifacts/p2/align/evidence.py` / `eval_align.py` | 证据生成脚本（可重跑） |

**schema 要点（`semotion.alignment/1`）**

```
schema, source{path,sha256,duration_s,fps,width,height,language,audio_rate},
method{name,truth_class:"MACHINE_WEAK",engine,params,notes},
segments[]{id,start_s,end_s,text,n_tokens,n_syllables,speech_start_s,speech_end_s},
syllables[]{index,token_index,segment_id,start_s,end_s,conf,island,peak_s},
words[]{id,segment_id,text,kind,n_syllables,start_s,end_s,conf,char_start,char_end,
        trailing_punct,group_id,syllable_ids},
chars[]{id,segment_id,token_id,text,kind,start_s,end_s,conf,char_start,char_end,
        group_id,approx},
quality{...}, warnings[]
```

`semotion.captions/1`：`source{fps,width,height,language}`、`style`（含 `font_cjk`/`font_latin`）、
`options{respect_segments,respect_words}`、
`cues[]{id,start_frame,end_frame,start_s,end_s,duration_s,lines,text,display_text,char_count,max_line_chars,word_ids,segment_ids}`、
`quality{n_cues,n_lines,max_line_chars,min_duration_s,max_duration_s,mid_word_cue_starts,segment_boundary_breaks,violations}`。
`group_id` 是 ASR 词分组（`<segment>:<word id>`），只用于**断行/断 cue 约束**；时间仍来自本模块的音节对齐。旧版 alignment（无 `group_id`）仍可生成字幕，只是词守卫退化为不生效。
