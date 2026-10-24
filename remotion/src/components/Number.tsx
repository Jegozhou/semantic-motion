import React from "react";
import { Easing, interpolate, useCurrentFrame } from "remotion";
import { Box } from "./Box";
import { ElementStyle } from "../theme";

const fmt = (v: number, decimals: number, prefix: string, suffix: string) => {
  const body = decimals > 0 ? v.toFixed(decimals) : String(Math.round(v));
  return `${prefix}${body}${suffix}`;
};

export const NumberEffect: React.FC<{
  value: number;
  decimals: number;
  prefix: string;
  suffix: string;
  box: [number, number, number, number];
  style: ElementStyle;
  startFrame: number;
  endFrame: number;
  enterDurFrames: number;
  exitDurFrames: number;
}> = ({ value, decimals, prefix, suffix, box, style, startFrame, endFrame, enterDurFrames, exitDurFrames }) => {
  const frame = useCurrentFrame();
  const local = frame - startFrame;
  const countFrames = Math.max(6, enterDurFrames + 6);
  const p = interpolate(local, [0, countFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const shown = local >= countFrames ? value : Math.round(value * p * 10) / 10;
  const outStart = Math.max(0, endFrame - startFrame - exitDurFrames);
  const opacity = interpolate(local, [outStart, Math.max(outStart + 1, endFrame - startFrame)], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <Box box={box} style={style} opacity={opacity}>
      <div
        style={{
          width: "100%",
          textAlign: "center",
          color: style.accent,
          fontSize: Math.round(style.fontPx * 1.35),
          fontWeight: 800,
          lineHeight: 1.05,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {fmt(shown, decimals, prefix, suffix)}
      </div>
    </Box>
  );
};
