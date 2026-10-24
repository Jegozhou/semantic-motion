import React from "react";
import { Composition, CalculateMetadataFunction } from "remotion";
import { Main, Props } from "./Main";

const calculateMetadata: CalculateMetadataFunction<Props> = async ({ props }) => {
  return {
    durationInFrames: Math.max(1, props.canvas?.durationInFrames ?? 150),
    fps: props.canvas?.fps ?? 25,
    width: props.canvas?.width ?? 1080,
    height: props.canvas?.height ?? 1920,
  };
};

export const RemotionRoot: React.FC = () => (
  <Composition
    id="Main"
    component={Main}
    durationInFrames={150}
    fps={25}
    width={1080}
    height={1920}
    defaultProps={{ schema: "semotion.remotion-props/1", canvas: { width: 1080, height: 1920, fps: 25, durationInFrames: 150 }, source: { file: "source.mp4" }, captions: [], elements: [] } as Props}
    calculateMetadata={calculateMetadata}
  />
);
