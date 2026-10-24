# ThreeUI 对抗性审查 — 是否吸收进 Semantic Motion

状态：**决策记录（2026-10-25）**。审查对象：https://github.com/MengTo/threeui
结论：**否决"运行时吸收"；条件批准"离线插入镜头"；主路径转向原生动效能力。**

> 本文只记录决策，不包含任何 ThreeUI 代码。仓库内不 vendor 其组件，不新增运行时依赖。

---

## 1. 被审查的计划

用户目标：把 MengTo/threeui 吸收进来，让口播视频动效更专业。

隐含的三步（本审查按最激进的理解评估）：

1. vendor 其 React/Three.js 组件；
2. 在我们的渲染链路里渲染这些 WebGL 场景，作为叠加图层或插入镜头；
3. 以此提升成片的"专业感"。

## 2. 事实核对（源码级，2026-10-25）

| 项 | 事实 | 对我们的含义 |
|---|---|---|
| 项目性质 | React + Three.js/WebGL **组件目录**：50 个父组件、111 路由、141 变体；核心是 ~46 个 shader 场景目录 | 是"网页 3D 展示组件库"，不是时间轴动效引擎 |
| 许可 | 代码 MIT；字体 OFL；**目录缩略图/预览视频与部分 CDN 素材不在 MIT 内**；Pro 源码不发布 | MIT 可用，但整包吸收会夹带非 MIT 远程素材 |
| 体积 | 仓库 42.6 MB；`public/source-code.json` **32.8 MB**；npm `@designcodeio/threeui@1.2.0` | 分发形态是生成物 + 大 JSON，不是可组合 SDK |
| 依赖 | React 18–19、Vite、`three@0.128` 与 `three@0.165` 双别名、peer `three>=0.149` | 运行时引入 Node/Chromium/React/Three 全套 |
| 时间模型 | `performance.now()` + `requestAnimationFrame` + `IntersectionObserver`（不可见即暂停）+ 鼠标交互；**无 `time` prop、无帧索引 API** | 无法确定性渲染第 N 帧，必须逐组件打补丁 |
| 像素确定性 | `window.devicePixelRatio`（≤1.5 / ≤2 因组件而异）、`antialias: true`、`alpha: true`（energy-orb）/`alpha: false`（liquid-form） | 各组件 alpha/DPR/AA 不统一，没有合成契约 |
| 导出能力 | scripts 仅 dev/build/preview/typecheck，无视频/离屏渲染入口 | "做成视频"需要自造浏览器渲染壳 |
| 健康度 | 6,825 stars / 646 forks，但 1 位主维护者（19 commits）+ 机器人；创建 2026-08-21，最后功能推送 2026-09-01；8 open issues | 热度 ≠ 生产成熟度；组件 API 非稳定契约 |

对照本项目约束：

| 约束 | 证据 |
|---|---|
| 渲染路径零 Node | v2 主链路 Python + ffmpeg；Remotion 仅历史可选后端 |
| 确定性是交付门 | 双跑 625/625 帧 `mean|Δ| = 0.0` |
| 逐像素可归因 | overlay-only + ink-only 双通道，QA 可定位到元素 |
| 离线可复现 | 无网络渲染；CTC 模型本地缓存 |
| 审美基线 | `docs/DESIGN.md`、`docs/BEST-PRACTICES.md §4`：不默认 3D/浏览器/粒子 |

## 3. Steelman（对用户计划的最强辩护）

1. **shader 工艺**：光晕、液态金属、颗粒、体积感，手写 Pillow 难以短期达到同等视觉密度；
2. **MIT + 高完成度**：48 个场景可读可改，省去从零写 shader 的时间；
3. **对品牌片/产品片确有效**：发布会、hero、章节转场，插入镜头能立刻抬"制作规格"。

问题不在"它没用"，而在**它服务的目标（3D 展示）与"口播叙事动效的专业"不是同一目标**。

## 4. 十个对抗性问题

1. **吸收什么？** 风格母题（免费）/ 离线素材（一次性，中成本）/ 运行时代码（vendor + 造壳，高成本且不可逆）。计划默认了第三种，却没有说明为何需要运行时可变性。
2. **是否解决我们的问题？** 否。口播动效的杠杆是语义相关性、时间精准、构图避让、排版层级、节奏克制；ThreeUI 的强项是全屏 3D 冲击。
3. **确定性守得住吗？** 需要虚拟时钟、固定 rAF 步进、固定 DPR、统一 AA/alpha、禁用可见性暂停与鼠标；逐组件改。同机双跑或许可过，**跨机器一致性丢失**。
4. **运行时依赖值得吗？** 当前安装 = venv + ffmpeg + 字体（+ 可选 1.3 GB CTC）；引入后 +Chromium/Node/React/Three（约 +200 MB）且渲染变慢。
5. **离线与可复现？** 部分场景引用 threeui.com/CDN 素材；目录媒体不在 MIT。离线需逐场景替换素材。
6. **QA 归因还成立吗？** WebGL 无 ink-only pass，元素级归因失效；作为插入镜头则需新增窗口/音频/避脸检查，质量体系要重写一块。
7. **许可干净吗？** 代码 MIT、字体 OFL 干净；目录媒体与远程 CDN 素材不干净。合规只能自托管全素材并逐项登记。
8. **维护风险？** 单维护者、生成式打包、33 MB 源码 JSON、社区版移除 Pro/Beta；组件 API 非稳定契约，长期 fork 不可避免。
9. **审美风险？** 3D shader 压真人口播最易呈现为"模板特效"，与我们已建立的克制、类型优先品牌冲突。
10. **机会成本？** 同样两周：关键帧/数据可视化/节奏模型/critic 可被现有 QA 验证；ThreeUI 集成提升的是不可测量的"炫技感"，且难回退。

## 5. 决策矩阵

| 方案 | 内容 | 成本 | 风险 | 专业感收益 | 结论 |
|---|---|---|---|---|---|
| A 运行时吸收 | vendor 场景 + 浏览器渲染壳，作为常规图层 | 高（2–4 周） | 确定性/归因/许可/依赖全破 | 低–中且风格冲突 | **否决** |
| B 离线插入镜头 | 固定环境预渲染 3–5 段带 alpha 的 1.5–3 s 插入/片头 | 中（2–3 天 spike + 1 周） | 低（可移除、可静态审查） | 中（仅品牌片/转场） | **条件批准，先 spike** |
| C 母题移植 | 借鉴光晕/颗粒/渐变/视差，Pillow+numpy 原生重画 | 低 | 低 | 中 | **推荐** |
| D 原生能力优先 | 关键帧、数据可视化、节奏模型、critic、排版 | 中 | 低 | **高** | **推荐（主线）** |

**决策：D 为主 + C 点缀；B 仅在用户确认"品牌片/允许插入镜头"后启动。** 仓库不引入 ThreeUI 代码或运行时依赖。

## 6. B 方案的唯一被 sanction 架构（离线素材工厂）

```
构建机（一次性，版本固定）
  Chrome headless + DPR=1 + 关闭 MSAA + 虚拟时钟（覆盖 performance.now/rAF）+ 本地素材
        │  逐帧截图（RGBA PNG，2× 超采样后降采样）
        ▼
  assets/inserts/<name>/frames/*.png + meta.json
    （帧号、alpha、来源 commit、许可证、素材哈希）
        │
运行时（现有 renderer，零浏览器零网络）
  render2 "insert track"：在声明窗口内替换底片，音频原样续接
  QA 新增 insert 检查：窗口/时长/音频连续/不压脸/素材哈希/许可清单
```

硬约束：

- 默认关闭（`--inserts` off），素材包独立目录、可单独卸载；
- 只允许全画幅插入/片头，**禁止把 WebGL 当叠加图层压在人脸上**；
- 素材自托管，逐项记录 MIT/OFL，禁止目录媒体；
- 不做跨机器像素一致性承诺（插入素材是固定文件，合成仍确定）。

## 7. Spike 门禁（仅在你明确说"做"之后执行，2 天）

| 门 | 标准 |
|---|---|
| G1 确定性 | 同机双跑解码帧 `mean|Δ| ≤ 0.05`、`max ≤ 32`（与现有 QA 一致） |
| G2 离线 | 渲染期零网络请求，全部素材本地 |
| G3 Alpha | 2× 超采样边缘无黑/白边；合成不触及人物掩码（插入镜头校验窗口边界） |
| G4 成本 | 1080p 插入 ≤3× 实时；插入总时长 ≤ 片长 8% |
| G5 品味 | 与原生标题卡盲 A/B，用户 + 至少一位真实观众更偏好插入镜头 |
| G6 许可 | 仅 MIT 代码 + OFL 字体；无目录媒体；`ASSET-LICENSES` 可审计 |

**Kill criteria**：G1 或 G2 失败 → 停止，退回 C；G5 失败 → 不进入产品；任一 gate 失败不得以"以后再说"放行。

## 8. 主路径：真正提升"专业"的 ROI 排序

1. 关键帧/逐属性缓动（借鉴 movis `Motion`）——当前只有 enter/exit 两段；M / 高。
2. 数据可视化组件（bar/line/comparison/timeline，数值来自转写）——M / 高。
3. 节奏与停留模型（字数、语速、句长 → hold/enter）——S / 高。
4. 视觉评审 critic（在 `layout.candidates` 合法候选内排序）——M / 高。
5. 排版微调（中英混排、tabular numerals、光学边距）——S / 中。
6. 背景自适应（亮背景加深 scrim / 调整 accent；可 GPU-free 移植光晕母题）——S / 中。

## 9. 待用户确认（只影响 B 是否启动）

1. "更专业"= A 纪录片/品牌片质感，还是 B 科技发布会/3D 炫技？
2. 是否允许插入镜头替换原画面 1–3 s（音频连续、不压脸）？
3. 是否接受 +~200 MB 安装体积、渲染变慢、跨机器一致性下降？
4. 目标平台：9:16 短视频，还是 16:9 品牌片/发布会？

在这 4 个问题回答前，不启动 B；主线按 D 推进。

## 10. 复现与证据

- 克隆与只读检查（不写入仓库）：
  ```bash
  gh api repos/MengTo/threeui --jq '{stars,forks,license,created_at,pushed_at,open_issues}'
  curl -sL https://raw.githubusercontent.com/MengTo/threeui/main/package.json
  curl -sL https://raw.githubusercontent.com/MengTo/threeui/main/ASSET-LICENSES.md
  curl -sL https://raw.githubusercontent.com/MengTo/threeui/main/src/shaders/energy-orb/EnergyOrb.tsx | grep -nE 'performance.now|requestAnimationFrame|devicePixelRatio|antialias|alpha:'
  curl -sL https://raw.githubusercontent.com/MengTo/threeui/main/src/shaders/liquid-form/LiquidFormBackground.tsx | grep -nE 'performance.now|alpha:|IntersectionObserver'
  ```
- 本仓库相关约束证据：`reports/FINAL-REPORT-V2.1.md`、`artifacts/v2/final_zq/qa/qa_metrics.json`、`docs/BEST-PRACTICES.md §4`。
