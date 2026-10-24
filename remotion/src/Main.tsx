import React, { useEffect, useState } from "react";
import { AbsoluteFill, Audio, OffthreadVideo, useCurrentFrame, staticFile } from "remotion";
import { Keyword } from "./components/Keyword";
import { NumberEffect } from "./components/Number";
import { ListEffect } from "./components/List";
import { Card } from "./components/Card";
import { Captions, CaptionCue } from "./components/Captions";
import { FONT_FAMILY } from "./theme";
import { loadSemanticFont } from "./fonts";

export type Element = {
  id: string;
  effect: "keyword" | "number" | "list" | "card";
  startFrame: number;
  endFrame: number;
  box: [number, number, number, number];
  content: {
    text: string;
    value?: number | null;
    prefix?: string;
    suffix?: string;
    decimals?: number;
    items?: { text: string; startFrame: number }[];
  };
  style: {
    accent: string;
    fg: string;
    bg: string;
    fontPx: number;
    radius: number;
    align: "left" | "center";
    weight: number;
  };
  motion: { enterDurFrames: number; exitDurFrames: number };
};

export type Props = {
  schema: string;
  canvas: { width: number; height: number; fps: number; durationInFrames: number };
  source: { file: string; audio?: boolean };
  captions: CaptionCue[];
  elements: Element[];
  captionSafeBottomPx?: number;
};

const ElementView: React.FC<{ el: Element }> = ({ el }) => {
  const common = {
    box: el.box,
    style: el.style,
    startFrame: el.startFrame,
    endFrame: el.endFrame,
    enterDurFrames: el.motion.enterDurFrames,
    exitDurFrames: el.motion.exitDurFrames,
  };
  if (el.effect === "number") {
    return (
      <NumberEffect
        {...common}
        value={el.content.value ?? 0}
        decimals={el.content.decimals ?? 0}
        prefix={el.content.prefix ?? ""}
        suffix={el.content.suffix ?? ""}
      />
    );
  }
  if (el.effect === "list") {
    return <ListEffect {...common} items={el.content.items ?? []} />;
  }
  if (el.effect === "card") {
    return <Card {...common} text={el.content.text} />;
  }
  return <Keyword {...common} text={el.content.text} />;
};

export const Main: React.FC<Props> = (props) => {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    loadSemanticFont().then(() => setReady(true));
  }, []);
  const { canvas, source, elements, captions } = props;
  const src = staticFile(source.file);
  const frame = useCurrentFrame();
  const hidden = (source as { hidden?: boolean }).hidden === true;
  return (
    <AbsoluteFill style={{ backgroundColor: "#000", fontFamily: FONT_FAMILY }}>
      {hidden ? null : (
        <>
          <OffthreadVideo src={src} muted />
          {source.audio !== false ? <Audio src={src} /> : null}
        </>
      )}
      <Captions cues={captions ?? []} canvas={canvas} safeBottomPx={props.captionSafeBottomPx ?? 0} />
      {/* Elements are gated on the absolute composition frame: `useCurrentFrame`
          stays identical inside a Sequence-offset tree only by accident, and mixing
          both offset conventions silently shifted every animation (number stuck at 0). */}
      {elements
        .filter((el) => frame >= el.startFrame && frame < el.endFrame)
        .map((el) => (
          <ElementView key={el.id} el={el} />
        ))}
      {ready ? null : null}
    </AbsoluteFill>
  );
};
