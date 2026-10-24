# 测试报告（Test Report）— Semantic Motion 0.1.0

> 日期：2026-10-24 · 环境：Linux 沙箱，**2 vCPU、无 GPU**，Python 3.13（venv），Claude Code 2.1.310。
> 依赖实测版本：numpy 2.5.4 · opencv 5.0.0 · skia-python 144.1 · mediapipe 1.1.1 · faster-whisper 1.2.1 · onnxruntime 1.31.0。
> 说明约定：**【真实】**= 真实人物口播素材上的完整运行；**【合成】**= TTS 生成的音频，仅用于有真值的时间精度测量，不代表真实口播效果。
> 视频等二进制证据不进仓库；素材来源、许可与复现命令见 [TEST_FOOTAGE.md](TEST_FOOTAGE.md)。

## 0. 结论摘要
| 项目 | 结果 |
|---|---|
| 真实素材端到端（4 条，横/竖、中/英、含高难度片段） | 全部 **QA PASS**，0 error；音频全部逐比特一致；帧数/时长与源一致；A/V 偏移 0 ms |
| 人脸遮挡 | 4 条全部 0 次（硬约束，渲染前 + 渲染后双重检查，集成测试覆盖） |
| 图形落点 vs 锚点 | 偏差 0 ms（OpenClaw 片段最大 13 ms，来自 30 fps 帧量化）；改稿中用户要求“早出现”的元素按要求提前 200 ms |
| 图形之外区域不被改动 | 透传帧 PSNR 最低 45.65 dB（竖屏）… 49.82 dB（英文） |
| 确定性 | 同一工程两次独立渲染：980/980 帧 framemd5 **完全一致** |
| 自然语言改稿 | 1 条组合改稿请求 → 计划 diff → 重建；只动了被点名的 3 个元素，其余元素时间与位置完全不变；`sm undo` 精确还原 |
| 插件打包 | `claude plugin validate` 通过；本地 marketplace 添加 + 安装成功（2 个技能、2 个子代理） |
| 干净环境安装 | `setup.sh` 在空 HOME 中 40 s 完成，`sm doctor` ok，并完成一次端到端 PASS |
| 自动化测试 | pytest：单元 + 真实工程集成测试，全部通过（见 §8） |
| 【合成】词级时间精度 | 精修后 ±2 帧（25 fps）内：中文 92.5%、英文 96.4%；英文最大误差 557.5 → 85 ms |

## 1. QA 检查了什么
每次 `sm build` 都会跑两级 QA（详见 `skills/semantic-motion/reference/qa.md`）：
- **渲染前**（编译后的时间线）：人脸/身体/安全区/烧录文字的遮挡、元素互相碰撞、屏幕密度（每分钟元素数、无图形时长占比 clean_ratio）、
  落点与锚点偏差、阅读时长、镜头推进是否裁掉字幕（CAMERA_CROPS_TEXT）。error 会触发自动修复（重新求解版式 / 缩小 / 跳过），再复检。
- **渲染后**（真实成片）：分辨率、帧数、时长、色彩元数据、音频是否为源流的逐比特拷贝、A/V 起点偏移、
  图形区域之外的透传帧 PSNR、每个图形框内是否真的画出了东西（graphics_box_mean_diff > 0）、联系表（contact sheet）与时间线图。
- 联系表由成片抽帧生成，每次都由导演（Claude）目检；本报告中的修复 D17 就是目检发现的。

## 2. 【真实】端到端运行
所有运行使用默认 ASR `large-v3-turbo`（CPU int8）、默认风格 `balanced`（智能均衡）。计划由导演（Claude）按 SKILL.md 流程编写。

| 工程 | 规格 | 元素 | 密度 | clean | 落点偏差 max | 帧 | 音频 | A/V | 透传 PSNR min | 警告 | 渲染 | 整体 build |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `zh_talk_16x9` | 1920×1080, 23.976 fps, 40.9 s | 6 + 1 推镜 | 8.81/min | 0.355 | 0 ms | 980/980 | 逐比特 | 0 ms | 48.13 dB | 无 | 84 s（11.6 fps） | 100 s（含确定性检查） |
| `zh_talk_9x16` | 1080×1920, 23.976 fps, 40.9 s | 6 | 8.81/min | 0.499 | 0 ms | 980/980 | 逐比特 | 0 ms | 45.65 dB | CLIPPED_TO_SHOT | 100 s（9.8 fps） | 109 s |
| `en_talk_16x9` v1 | 1280×720, 25 fps, 80 s | 6 + 1 推镜 | 4.5/min | 0.551 | 0 ms | 2000/2000 | 逐比特 | 0 ms | 49.82 dB | 无 | 70 s（28.5 fps） | 234 s（含 `--overlay` ProRes 4444 透明层导出） |
| `en_talk_16x9` v2（改稿） | 同上 | 5 + 1 推镜 | 3.75/min | 0.582 | 200 ms（用户要求提前） | 2000/2000 | 逐比特 | 0 ms | 49.74 dB | 无 | — | 75 s |
| `zh_openclaw_60s` | 1920×1080, 30 fps, 60 s | 3 | 3.0/min | 0.785 | 13 ms | 1800/1800 | 逐比特 | 0 ms | 47.32 dB | SHRUNK | 135 s（13.3 fps） | 142 s |

分析阶段耗时（同一 2 vCPU 机器）：ASR 41.5 s（40.9 s 中文）、64.5 s（80 s 英文）、75.8 s（60 s 中文快语速）；视觉分析约 43–130 s（与其他任务并行时偏慢）。

### 2.1 `zh_talk_16x9`（中文横屏，12 个跳切镜头，源片自带烧录字幕）
- 元素：`name` 姓名条（Zohran Mamdani / 纽约市长）、`tickets` 数字卡（2,000+ 张，锚定“两千多张”，数字在说出时计数到位）、`deadline` 日期关键词（8月17日前）、
  `steps` 三步流程（访问网站 → 了解规则 → 参加抽签，每一步各自锚定到原话）、`tdf` 列表卡（TDF 会员 · 免费：The Theatre Development Fund / 为期 2 个月的会员资格 / 百老汇演出折扣，每项锚定到原话）、
  `end` 结尾关键词（演出现场见！）+ `cam_end` 结尾缓推镜头。
- 源片已有烧录字幕 → 不加字幕，所有图形避让字幕带（D16）。
- 发现并修复：最初结尾推镜跨越了 37.12 s 的跳切而被跳过；改为锚定“8:现场见”并限制在最后一个镜头内后，推镜又会把底部烧录字幕推出画面 → 新增 CAMERA_CROPS_TEXT 规则，编译器自动把推镜焦点从 0.36 调到 0.76 以保住字幕（D17，CAMERA_ADJUSTED）。
- 最终：PASS，0 error / 0 warning；联系表目检：字幕完整、脸部无遮挡、图形落在桌面/胸前负空间。

### 2.2 `zh_talk_9x16`（中文竖屏，抖音安全区）
- 同一段讲话的**派生竖屏**（中心裁切，不是原生竖屏拍摄，见限制）。烧录字幕在竖屏中被裁成整行宽 → 推镜头无法在保住字幕的同时推进，
  编译器按规则跳过该推镜（CAMERA_CROPS_TEXT），导演据此在计划中关闭 `cam_tickets`；`tdf` 所在镜头之后是大特写，因此改为更紧凑的关键词卡；`discount` 关键词为控制均衡密度而关闭（TDF 卡已承载该信息）。
- 平台设为 `douyin`：版式自动落在胸前/桌面区域（BODY_OVERLAP 仅为 info：竖屏里脸以下几乎都是身体，图形压在衣服上是预期的），避开平台右侧按钮栏与底部文案安全区；
  `name` 与 `tdf` 在跳切处因新机位没有不遮脸的空间而提前收起（CLIPPED_TO_SHOT 警告，属预期：图形不跨越会遮脸的镜头）。
- 最终：PASS，0 error，1 类 warning（CLIPPED_TO_SHOT）。

### 2.3 `en_talk_16x9`（英文横屏，左下角已有烧录姓名条）
- 元素：`twenty`（20 years）、`spirit`（With you in spirit）、`since2001`（custom 程序化时间轴 2001 → 2021，English Wikipedia → today）、
  `versions`（+18 more Wikipedias turn 20）、`arc`（Anyone could edit → Idealistic vision → Something real）、`today`（55M+ articles / ~300 languages / 1.5B devices，
  每一项在说到时出现）+ 推镜。
- 已有姓名条被检测为烧录文字并避让；未重复添加姓名条（D20 克制）。
- v1：PASS，0 error / 0 warning。同一次构建用 `--overlay` 导出了带 alpha 的 ProRes 4444 叠加层（`renders/overlay.mov`，1280×720，80.0 s，可在剪辑软件中二次编辑）。

## 3. 【合成】词级时间精度基准
**这是合成测试**：用 edge-tts（zh-CN-YunxiNeural / en-US-GuyNeural）生成语音，以 TTS 的 WordBoundary 作为词起点真值，
测量引擎实际使用的时间（`tests/bench_timing.py`，结果 `timing_benchmark.json`）。TTS 语音比真人更“干净”，因此这些数字只说明时间精修算法的相对效果，
不能直接等同于真实口播的精度；真实素材上的精度由 §2 的落点偏差与联系表目检补充验证。

| 语言 | 方法 | n | 中位误差 | p90 | 最大 | ±1 帧 | ±2 帧 |
|---|---|---|---|---|---|---|---|
| 中文 | Whisper 原始词时间 | 53 | 27.5 ms | 79.5 ms | 147.5 ms | 64.2% | 92.5% |
| 中文 | **精修（soft_start + silence_end，默认）** | 53 | 25.0 ms | 77.5 ms | 147.5 ms | 69.8% | 92.5% |
| 英文 | Whisper 原始词时间 | 56 | 26.3 ms | 76.2 ms | 557.5 ms | 69.6% | 91.1% |
| 英文 | **精修（默认）** | 56 | 26.3 ms | 67.5 ms | 85.0 ms | 69.6% | 96.4% |

- 早期默认的 `pause_onset` 规则在同一基准上更差（±2 帧：中文 86.8%、英文 80.4%，p90 多 40–50 ms）→ 改为当前默认（D7）。
- 锚点探针（引擎把短语解析到的时间 vs 真值）：
  中文 百分之四十 +25 ms · 两万 −25 ms · 八万五千 −45 ms · 信任 +57.5 ms · 先定目标 +145 ms；
  英文 forty percent +22.5 ms · twenty thousand 0 ms · trust +5 ms · set a goal −27.5 ms。9/9 全部解析（口语数字 ↔ ASR 阿拉伯数字的回退见 D13）。

## 4. 确定性
- `zh_talk_16x9`：`sm build --determinism` 内置检查（24 帧，1.48–2.48 s 重复渲染比较）通过；另外用 `sm render` 做了一次**完全独立的第二次渲染**，
  两次成片逐帧 `ffmpeg -f framemd5`：**980/980 帧哈希完全一致**。
- 集成测试 `test_real_project_compiles_deterministically_and_never_covers_face`：每个真实工程编译两次，时间线哈希一致。

## 5. 【真实】自然语言改稿 → diff → 撤销（`en_talk_16x9`）
用户原话（组合请求）：
> 把 +18 那个数字放到左边，整体换成蓝色，删掉 "With you in spirit"，最后的列表早一点出来

导演将其翻译为计划修改，`sm diff` 输出（字段级）：
```
- spirit (keyword)
~ today.motion: (none) -> {"lead": 0.2}
~ versions.placement: (none) -> {"region": "left"}
~ accent: "auto" -> "sky"
（以及两条 intent 备注）
```
重建后编译时间线逐元素对比（v2 vs v1）：
```
= arc: unchanged (timing + box)
= since2001: unchanged (timing + box)
+ spirit            （v1 有、v2 删除）
~ today: t_in 69.007->69.207; t_land 69.2->69.4   （v2 提前 0.2 s）
= twenty: unchanged (timing + box)
~ versions: box [103, 258, …] -> [887, 194, …]     （v2 在左侧，v1 在右侧）
```
- 结果：只有被点名的元素发生变化；未被点名的 `arc`、`since2001`、`twenty` 时间和位置**完全不变**。
  这条性质是在本测试中发现缺陷后才成立的：修复前，`versions` 移到左边会通过“同侧一致”规则把相邻元素也带到左边（D18）。
- 联系表目检：+18 在左、强调色为蓝、spirit 已删除、列表 69.01 s 出现（v1 为 69.21 s）。v2 QA PASS，0 warning。
- `sm undo` → `restored plan.001.json -> plan.json`；随后 `sm diff` 精确显示反向差异；v1 重新构建 PASS（即 §2 表中 v1 行）。

## 6. 【真实】鲁棒性：`zh_openclaw_60s`
高难度素材：快语速（≈5.9 字/s）、17 个镜头、大量 B-roll/录屏/文档截图、源片自带烧录字幕与贴纸图形、ASR 把品牌名识别错。
- ASR 错误由计划中的 `corrections` 修正后再锚定：OpenCloud → OpenClaw、卖客 → 卖课、常哭了 → 笑哭了、又懒 → 又来（单元测试 `test_corrections_map_brand_names` 覆盖）。
- 导演初稿 5 个元素；预览图目检后主动关闭 2 个（`shells`、`u_turn`）：源片在那两处已经有自己的图形 + 字幕，再加会打架（D20 克制）。
- 保留：`risk`（啥权限都敢给，B-roll 黑板空白处）、`double_edge`（讲人话就能操作 → 使用门槛降低 → 当黑客也更简单，主讲镜头左侧负空间，三步各自锚定到原话）、
  `brokers`（证券公司：严禁安装，叠在文档截图上，glass 底板；因空白不足被自动缩小 → SHRUNK 警告）。
- 最终：PASS，0 error；1800/1800 帧，音频逐比特一致。
- 对照：同一片段上启发式 `auto-plan` 只给出了一个推镜（置信度过滤后），说明高质量的“决定加什么”依赖导演（Claude），auto-plan 只是基线（D12）。

## 7. 安装与打包
- `claude plugin validate .`（Claude Code 2.1.310）：**Validation passed**，仅 1 条可选提示 ICON_MISSING。
- 本地 marketplace：`/plugin marketplace add <repo>` + `/plugin install semantic-motion@semantic-motion` 成功；
  加载结果：Skills（`semantic-motion`、命令 `motion`）、Agents（`motion-director`、`motion-critic`），常驻上下文约 354 token。
- 干净环境：`env -i HOME=<空目录> … skills/semantic-motion/scripts/setup.sh --asr-model small` → 40 s 完成（新建 venv、安装全部依赖、下载视觉模型；沙箱网络较快，慢网络下会更久），`sm doctor` ok（ffmpeg/libx264、Noto 字体、4 个视觉模型均就绪）。
  在该环境中对一段 24 s 英文片段跑完整流程：analyze 48 s（ASR small 18.4 s、视觉 29.5 s）→ auto-plan → build 32 s → **QA PASS**，音频逐比特一致。

## 8. 自动化测试
`pytest tests/`（`SM_TEST_PROJECTS` 指向真实工程时运行集成测试，否则跳过）：
- 单元：数字/中文数字/英文数字词、`numeric_variants`、文本归一与宽度、锚点语法、短语 → 词起点、口语数字匹配、模糊匹配与错误、corrections、
  soft_start / silence_end 精修、不在虚词上断拍、版式求解器（硬障碍、avoid 框与偏好、无处可放时返回 None）、值解析、推镜不裁字幕。
- 集成（每个真实工程）：编译确定性 + 不遮脸/不碰撞/不越安全区；**渲染出的像素足迹与 QA 使用的框一致**（这条测试发现了 D15 的二次缩放缺陷）。
- 最终结果：见 §8.1。

### 8.1 最终运行
`SM_TEST_PROJECTS` = 4 个真实工程（zh_talk_16x9、zh_talk_9x16、en_talk_16x9、zh_openclaw_60s），在全部最终构建完成后运行：
**27 passed in 3.81 s**（19 个单元测试 + 2 个集成测试 × 4 个工程）。

## 9. 开发中由测试发现并修复的缺陷
| 编号 | 缺陷 | 发现方式 | 修复 |
|---|---|---|---|
| D7 | 早期 `pause_onset` 精修让停顿后的词起点偏早，±2 帧比例下降 | 【合成】基准 | 改为 soft_start + silence_end；英文最大误差 557.5 → 85 ms |
| D13 | “百分之四十 / twenty thousand” 等口语数字无法锚定到 ASR 输出的阿拉伯数字 | 基准锚点探针 | 数字变体 + 精确优先、模糊兜底 |
| D15 | QA 使用的元素框被二次缩放，与实际像素不符 | 新集成测试 | `element_box_at` 直接使用编译框 |
| D17 | 结尾推镜把烧录字幕推出画面 | 联系表目检 | 推镜必须保住叠加文字与人脸，否则调整焦点/降低倍率/跳过 |
| D18 | 改一个元素的位置会牵连相邻元素 | 改稿测试 | 同侧一致只在“前一元素 4 s 内结束且为自动放置”时继承 |
| — | 词结束时间取整后可能破坏单调性 | 精修代码复查，单元测试覆盖 | 单调取整 |
| — | auto-plan 对重复出现的相同数字重复出卡 | 检查 auto-plan 在真实素材上的输出 | 跳过重复值 |

## 10. 性能（2 vCPU、无 GPU）
- 渲染 9.8–28.5 fps（取决于分辨率与图形覆盖帧数；只有图形帧走 Skia 合成，其余帧透传）。
- 40 s 1080p 片段：分析约 1.5–3 min + 构建约 1.7 min；60 s 1080p 快语速片段：分析约 2.3 min + 构建 2.4 min。
- 首次运行需下载模型：视觉模型约 25 MB，Whisper large-v3-turbo 约 1.6 GB（可用 `--asr-model small` 换小模型）。

## 11. 已知限制 / 未验证
- 测试素材是已发布的公开许可**成片**（带烧录字幕、跳切），不是真正的“原始素材”；对原始素材通常更容易（无需避让既有文字），但未实测。
- 竖屏只在**派生竖屏**（横屏中心裁切）上测试，原生竖屏拍摄未实测。
- 最长实测 80 s；5–10 分钟长视频的性能与分段导演（`motion-director` 子代理）未实测。
- 时间精度的量化数字来自【合成】TTS 基准；真实口播上用落点偏差（相对 ASR 精修后的词时间）+ 目检验证，没有人工标注真值。
- ASR 对品牌/人名会出错，需要导演在计划里写 `corrections`（技能流程里已要求先通读转写稿）。
- 启发式 `auto-plan` 很保守，只是起点；“加什么、加在哪”的质量主要来自导演（Claude）阅读 brief 后写的计划。
- 单镜头长时间静止的背景文字也会被当作需保护的叠加文字（偏保守）。
- 插件无图标（validator 的可选提示）。

## 12. 复现
```bash
bash skills/semantic-motion/scripts/setup.sh      # 或按 README 安装（插件 / 技能）
sm analyze zh_talk_16x9.mp4 --language zh         # → zh_talk_16x9.smproj/analysis + brief.md
# 导演按 SKILL.md 写 plan.json（本次运行的计划在各工程的 plan.json / plans/ 快照中）
sm validate zh_talk_16x9.smproj && sm preview zh_talk_16x9.smproj   # 预览图目检
sm build zh_talk_16x9.smproj --determinism        # → renders/final.mp4 + qa/report.md + qa/contact_sheet.jpg
ffmpeg -i renders/final.mp4 -map 0:v -f framemd5 a.framemd5   # 再 sm render --out b.mp4 并比较
SM_TEST_PROJECTS=a.smproj:b.smproj pytest tests/
python tests/bench_timing.py --out /tmp/sm_bench  # 【合成】时间基准（需要 edge-tts 与网络）
```
