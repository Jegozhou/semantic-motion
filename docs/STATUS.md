# Semantic Motion — 工作状态（恢复入口）

> 恢复工作时：先读本文件，再读 `docs/DECISIONS.md`，然后 `docs/TEST_REPORT.md`。
> 开发环境（沙箱）：仓库 /data/semantic-motion · venv /data/venv · 模型 /data/models · 素材 /data/footage · 运行工程 /data/runs
> 远端：github.com/Jegozhou/semantic-motion（私有，main）。沙箱无 git 凭据，文本文件通过 GitHub MCP `push_files` 推送；
> 视频/图片等二进制证据不进仓库（见 TEST_FOOTAGE.md 的复现命令），在交付消息中以附件提供。

## 阶段
- [x] P0 环境与测试素材（docs/TEST_FOOTAGE.md）
- [x] P1 引擎核心：媒体 I/O、音频特征、ASR + 时间精修、视觉分析
- [x] P2 计划 DSL、锚点解析、时间编译、版式求解
- [x] P3 Skia 渲染器 + 元素库（balanced 风格）
- [x] P4 QA（渲染前/后）、联系表、时间线图、确定性、自动修复
- [x] P5 技能打包：SKILL.md、reference、agents、commands、插件清单（`claude plugin validate` 通过，本地 marketplace 安装成功）
- [x] P6 真实素材：zh 16:9、zh 9:16、en 16:9（含改稿→diff→撤销）、OpenClaw 鲁棒性片段
- [x] P7 合成时间基准、单元/集成测试（4 个真实工程下 27 项全部通过）、干净环境安装 + 端到端
- [x] P8 文档、推送 GitHub、交付成片

## 当前版本：0.1.0（首个完整可用版本）
所有结论与数据见 TEST_REPORT.md。下一步候选（未开始）：
1. 原生竖屏/手机原始素材（非派生裁切）的实测；更长视频（5–10 min）的性能与分段导演（motion-director 子代理）实测。
2. auto-plan 基线的质量提升（目前保守，只作起点）。
3. 插件图标（validator 的 ICON_MISSING 提示，可选）。
4. 背景静态文字与烧录叠加文字的区分（目前单镜头视频里长期静止的背景文字也会被当作需要保护的叠加文字，偏保守）。

## 日志
- 2026-10-24 首轮：zh 16:9 PASS；随后完成竖屏、英文、基准、测试、文档。
- 时间精修默认改为 soft_start + silence_end（D7）；锚点支持口语数字（D13）。
- 最终构建（全部 PASS）+ 逐帧确定性（980/980 帧 framemd5 一致）+ 改稿局部性时间线对比，写入 TEST_REPORT.md。
- 修复：QA 框二次缩放（D15，由新集成测试发现）、推镜裁字幕（D17，由联系表目检发现）、改稿牵连其他元素（D18，由改稿测试发现）。
