import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Box } from "./Box";
import { clampText, ElementStyle } from "../theme";

export const Keyword: React.FC<{
  text: string;
  box: [number, number, number, number];
  style: ElementStyle;
  startFrame: number;
  endFrame: number;
  enterDurFrames: number;
  exitDurFrames: number;
}> = ({ text, box, style, startFrame, endFrame, enterDurFrames, exitDurFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const local = frame - startFrame;
  const s = spring({ frame: local, fps, durationInFrames: Math.max(2, enterDurFrames), config: { damping: 200, mass: 0.6 } });
  const scale = interpolate(s, [0, 1], [0.86, 1]);
  const opacityIn = interpolate(local, [0, Math.max(1, enterDurFrames * 0.6)], [0.55, 1], { extrapolateRight: "clamp" });
  const outStart = Math.max(0, endFrame - startFrame - exitDurFrames);
  const opacityOut = interpolate(local, [outStart, Math.max(outStart + 1, endFrame - startFrame)], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const underline = interpolate(local, [enterDurFrames * 0.4, enterDurFrames * 1.4], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <Box box={box} style={style} scale={scale} opacity={Math.min(opacityIn, opacityOut)}>
      <div style={{ width: "100%", textAlign: style.align === "left" ? "left" : "center" }}>
        <div style={{ color: style.fg, fontSize: style.fontPx, fontWeight: style.weight, lineHeight: 1.22, whiteSpace: "pre-wrap" }}>
          {clampText(text, 26)}
        </div>
        <div style={{ height: Math.max(4, Math.round(style.fontPx * 0.06)), width: `${underline * 78}%`, background: style.accent, borderRadius: 99, marginTop: 10, marginLeft: style.align === "center" ? "11%" : 0 }} />
      </div>
    </Box>
  );
};
