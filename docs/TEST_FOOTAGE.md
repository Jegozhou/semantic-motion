# 测试素材（Test Footage）

所有真实素材均来自 Wikimedia Commons，许可允许再分发与改编。**这些都是已发布的成片，不是“原始素材”**：
有的带烧录字幕/台标、有的有剪辑跳切，这对引擎更难（需要避让既有文字、处理跨镜头），因此是更严格的测试。
素材本身不放进 Git 仓库（体积 + 再分发说明），下表给出下载地址与可复现的转换命令。

| 项目名 | 来源 | 许可 / 作者 | 转换 | 用途 |
|---|---|---|---|---|
| `zh_talk_16x9` | [我们将向纽约市高中生赠送2000张百老汇免费门票…（Mamdani announces Broadway ticket give away in Mandarin Chinese）](https://commons.wikimedia.org/wiki/File:%E6%88%91%E4%BB%AC%E5%B0%86%E5%90%91%E7%BA%BD%E7%BA%A6%E5%B8%82%E9%AB%98%E4%B8%AD%E7%94%9F%E8%B5%A0%E9%80%812000%E5%BC%A0%E7%99%BE%E8%80%81%E6%B1%87%E5%85%8D%E8%B4%B9%E9%97%A8%E7%A5%A8%E3%80%82%E7%8E%B0%E5%9C%A8%E5%B0%B1%E5%8F%82%E5%8A%A0%E6%8A%BD%E5%A5%96%E5%90%A7%EF%BC%81(Mamdani_announces_Broadway_ticket_give_away_in_Mandarin_Chinese).webm) | CC BY 4.0 · NYC Mayor's Office | 3840×2160 → 1920×1080（lanczos），23.976 fps，40.9 s | 中文口播、横屏、数字/日期/步骤/专有名词（百老汇、TDF） |
| `zh_talk_9x16` | 同上 | CC BY 4.0 · NYC Mayor's Office | 中心裁切 1216×2160 → 1080×1920（**派生竖屏，非原生竖屏拍摄**） | 竖屏版式（抖音安全区、胸前文字） |
| `en_talk_16x9` | [Katherine Maher's message on the occasion of Wikipedia 20](https://commons.wikimedia.org/wiki/File:Katherine_Maher%27s_message_on_the_occasion_of_Wikipedia_20.webm) | CC0 · Wikimedia Foundation | 前 80 s，1280×720，25 fps | 英文口播、左下角已有烧录姓名条、数字/年份/列举 |
| `zh_openclaw_60s` | [Commentary on OpenClaw hype (Chinese)](https://commons.wikimedia.org/wiki/File:Commentary_on_OpenClaw_hype_(Chinese).webm) | CC BY 4.0 · 林亦LYi | 截取 20–80 s，3840×2160 → 1920×1080，30 fps | 鲁棒性：快语速（≈5.9 字/s）、大量 B-roll/录屏、烧录字幕、品牌名误识别 |
| 合成 `tts_zh` / `tts_en` | edge-tts 生成（zh-CN-YunxiNeural / en-US-GuyNeural） | 自生成 | 静态画面 + TTS 音轨 | **合成**时间基准：TTS 的 WordBoundary 作为词起点真值 |

## 复现命令
```bash
# 下载（UA 按 Wikimedia 规范填写）
curl -L -o mamdani_zh.webm  "<上表文件页中的原始文件链接>"
ffmpeg -i mamdani_zh.webm -vf "scale=1920:1080:flags=lanczos" -c:v libx264 -crf 17 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 192k zh_talk_16x9.mp4
ffmpeg -i mamdani_zh.webm -vf "crop=1216:2160:(iw-1216)/2:0,scale=1080:1920:flags=lanczos" -c:v libx264 -crf 17 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 192k zh_talk_9x16.mp4
ffmpeg -t 80 -i maher_wp20.webm -c:v libx264 -crf 17 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 192k en_talk_16x9.mp4
ffmpeg -ss 20 -t 60 -i openclaw_commentary.webm -vf "scale=1920:1080:flags=lanczos" -c:v libx264 -crf 17 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 192k zh_openclaw_60s.mp4
# 合成基准
python tests/bench_timing.py --out /tmp/sm_bench
```

## 署名（输出视频再分发时需要）
- “Mamdani announces Broadway ticket give away in Mandarin Chinese” — NYC Mayor's Office, CC BY 4.0，经 Semantic Motion 添加动态图形（并有缩放/裁切）。
- “Katherine Maher's message on the occasion of Wikipedia 20” — Wikimedia Foundation, CC0。
- “Commentary on OpenClaw hype (Chinese)” — 林亦LYi, CC BY 4.0（仅用于分析/鲁棒性测试）。
