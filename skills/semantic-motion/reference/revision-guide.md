# Revisions in natural language

Workflow for any change request:
1. `sm status` (which plan/snapshot/build is current) and re-read the relevant part of `plan.json`.
2. Map the request to the smallest plan edit (table below). Identify the element by its words or time
   ("the 2,000 number", "0:12 那个") → match `id` via `build/timeline.json` (`t_land`) or `sm validate` output.
3. Edit `plan.json`; `sm validate`; `sm preview --elements <id>` and look at it; `sm build`.
4. Tell the user what changed (`sm diff`) and show the new contact sheet frame(s).
5. `sm undo` restores the previous snapshot if the user prefers the old version (then `sm build`).

| request (zh / en) | edit |
|---|---|
| 早一点 / 晚一点出现 · earlier / later | `motion.lead: 0.2` (earlier) or anchor another word; later: `"at": "3:门票"` or `"3:两千多张+0.3"` |
| 多停留一会 / 早点消失 · stay longer / shorter | `until: {"hold": 4}` or a later/earlier phrase anchor |
| 放左边 / 右边 / 上面 · move left/right/top | `placement.region: "left"` … (still face-safe) |
| 大一点 / 小一点 · bigger / smaller | `placement.size: "l"` / `"s"` |
| 换个颜色 / 用蓝色 · blue accent | `"accent": "sky"` (amber, mint, sky, coral, violet or `#RRGGBB`) |
| 太花了 / 更克制 · too busy / more subtle | remove lowest-importance elements; or `"style": "minimal"` |
| 再丰富一点 / 更有冲击力 · more punchy | add 1–2 elements on spine moments; or `"style": "bold"` |
| 这个字错了 · wrong word | fix `content.text`; if ASR misheard a name, add to `corrections` |
| 换成列表 / 图表 · make it a list/chart | change `type` and restructure `content` (see elements.md) |
| 跟着说话一个字一个字出来 · word by word as spoken | `motion.reveal: "spoken"` |
| 去掉这个 · remove it | delete the element or `"enabled": false` (keeps it for later) |
| 加个推镜头 · add a zoom | camera element `punch_in`/`push_in` anchored inside one shot |
| 加字幕 · add captions | `"captions": {"enabled": true}` (refused automatically if subtitles are burned in) |
| 别挡住手/产品 · don't cover the product | `placement.region` to the other side, or `callout` to point at it |
| 导出给剪辑师 · deliver to an editor | `sm build --overlay` → `renders/overlay.mov` (ProRes 4444 alpha) + `sm export` |
| 回到上一版 · undo | `sm undo` then `sm build` |

Keep `intent` up to date when you change an element; it explains the choice to the next session.
Never re-run `sm analyze --force` for a revision — analysis is cached and independent of the plan.
