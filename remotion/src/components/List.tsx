import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { ElementStyle } from "../theme";

export type ListItem = { text: string; startFrame: number };

/**
 * Step list. The panel is content-sized and grows as rows reveal, so a 3-step list
 * that is still on step 1 does not render a large empty card. Rows are anchored to
 * their own spoken frames (karaoke-like reveal), the whole panel fades at the end.
 */
export const ListEffect: React.FC<{
  items: ListItem[];
  box: [number, number, number, number];
  style: ElementStyle;
  startFrame: number;
  endFrame: number;
  enterDurFrames: number;
  exitDurFrames: number;
}> = ({ items, box, style, startFrame, endFrame, enterDurFrames, exitDurFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = frame - startFrame;
  const outStart = Math.max(0, endFrame - startFrame - exitDurFrames);
  const panelOpacity = interpolate(local, [outStart, Math.max(outStart + 1, endFrame - startFrame)], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const [x, y, w, h] = box;
  const rowFont = Math.round(style.fontPx * 0.82);
  const gap = Math.round(rowFont * 0.42);
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: w,
        height: h,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        opacity: panelOpacity,
      }}
    >
      <div
        style={{
          width: "100%",
          boxSizing: "border-box",
          background: style.bg,
          borderRadius: style.radius,
          padding: "18px 24px",
          display: "flex",
          flexDirection: "column",
          gap,
        }}
      >
        {items.map((item, i) => {
          const at = Math.max(0, item.startFrame - startFrame);
          const s = spring({ frame: local - at, fps, durationInFrames: Math.max(2, enterDurFrames), config: { damping: 200, mass: 0.6 } });
          const opacity = interpolate(s, [0, 1], [0, 1]);
          const dx = interpolate(s, [0, 1], [-18, 0]);
          if (local < at) {
            return null;
          }
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 14, opacity, transform: `translateX(${dx}px)` }}>
              <div
                style={{
                  width: Math.round(rowFont * 0.72),
                  height: Math.round(rowFont * 0.72),
                  borderRadius: 99,
                  background: style.accent,
                  color: "#101420",
                  fontSize: Math.round(rowFont * 0.5),
                  fontWeight: 800,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flex: "0 0 auto",
                }}
              >
                {i + 1}
              </div>
              <div style={{ color: style.fg, fontSize: rowFont, fontWeight: 600, lineHeight: 1.2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {item.text}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
