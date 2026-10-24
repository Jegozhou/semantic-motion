import React from "react";
import { AbsoluteFill } from "remotion";
import { ElementStyle } from "../theme";

export const Box: React.FC<{
  box: [number, number, number, number];
  style: ElementStyle;
  scale?: number;
  opacity?: number;
  translateY?: number;
  children: React.ReactNode;
}> = ({ box, style, scale = 1, opacity = 1, translateY = 0, children }) => {
  const [x, y, w, h] = box;
  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        width: w,
        height: h,
        transform: `translateY(${translateY}px) scale(${scale})`,
        transformOrigin: "center center",
        opacity,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: style.radius,
          background: style.bg,
          display: "flex",
          alignItems: "center",
          justifyContent: style.align === "left" ? "flex-start" : "center",
          padding: "18px 24px",
          boxSizing: "border-box",
        }}
      >
        {children}
      </div>
    </div>
  );
};
