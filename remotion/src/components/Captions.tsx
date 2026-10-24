import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import { FONT_FAMILY } from "../theme";

export type CaptionCue = {
  startFrame: number;
  endFrame: number;
  lines: string[];
  words?: { text: string; startFrame: number; endFrame: number }[];
};

export const Captions: React.FC<{
  cues: CaptionCue[];
  canvas: { width: number; height: number };
  safeBottomPx?: number;
}> = ({ cues, canvas, safeBottomPx = 0 }) => {
  const frame = useCurrentFrame();
  const cue = cues.find((c) => frame >= c.startFrame && frame < c.endFrame);
  if (!cue) return null;
  const basePx = canvas.height > canvas.width ? Math.round(canvas.height * 0.030) : Math.round(canvas.height * 0.040);
  const active = cue.words?.find((w) => frame >= w.startFrame && frame < w.endFrame);
  const text = cue.lines.join("\n");
  const lines = cue.lines;
  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", pointerEvents: "none" }}>
      <div
        style={{
          marginBottom: Math.round(canvas.height * 0.055) + safeBottomPx,
          maxWidth: canvas.width * 0.86,
          padding: `${Math.round(basePx * 0.35)}px ${Math.round(basePx * 0.7)}px`,
          borderRadius: Math.round(basePx * 0.5),
          background: "rgba(8,10,16,0.72)",
          textAlign: "center",
        }}
      >
        {lines.map((line, i) => (
          <div
            key={i}
            style={{
              color: "#F7F8FA",
              fontFamily: FONT_FAMILY,
              fontSize: basePx,
              fontWeight: 700,
              lineHeight: 1.34,
              textShadow: "0 2px 8px rgba(0,0,0,0.55)",
            }}
          >
            {line}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
