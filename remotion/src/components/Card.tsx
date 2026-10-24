import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { Box } from "./Box";
import { clampText, ElementStyle } from "../theme";

export const Card: React.FC<{
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
  const s = spring({ frame: local, fps, durationInFrames: Math.max(2, enterDurFrames), config: { damping: 200, mass: 0.7 } });
  const scale = interpolate(s, [0, 1], [0.96, 1]);
  const opacityIn = interpolate(s, [0, 1], [0.35, 1]);
  const outStart = Math.max(0, endFrame - startFrame - exitDurFrames);
  const opacityOut = interpolate(local, [outStart, Math.max(outStart + 1, endFrame - startFrame)], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const bar = interpolate(s, [0, 1], [0, 1]);
  return (
    <Box box={box} style={style} scale={scale} opacity={Math.min(opacityIn, opacityOut)}>
      <div style={{ width: "100%", display: "flex", gap: 18, alignItems: "stretch" }}>
        <div style={{ width: Math.max(5, Math.round(style.fontPx * 0.08)), borderRadius: 99, background: style.accent, transform: `scaleY(${bar})`, transformOrigin: "top" }} />
        <div style={{ color: style.fg, fontSize: style.fontPx, fontWeight: 700, lineHeight: 1.3, whiteSpace: "pre-wrap" }}>
          {clampText(text, 34)}
        </div>
      </div>
    </Box>
  );
};
