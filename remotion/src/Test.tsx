import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

export const TestClip: React.FC<{ text: string }> = ({ text }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 200 } });
  const y = interpolate(s, [0, 1], [80, 0]);
  const opacity = interpolate(s, [0, 1], [0.5, 1]);
  return (
    <AbsoluteFill style={{ background: "linear-gradient(#10141f,#1b2130)", justifyContent: "flex-end", alignItems: "center", paddingBottom: 260 }}>
      <div style={{ transform: `translateY(${y}px)`, opacity, background: "rgba(10,14,24,0.86)", border: "2px solid rgba(242,178,51,0.5)", borderRadius: 28, padding: "28px 40px", color: "#F5F7FA", fontFamily: "NotoSansSC", fontSize: 88, fontWeight: 700 }}>
        {text}
      </div>
    </AbsoluteFill>
  );
};
