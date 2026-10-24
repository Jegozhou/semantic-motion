import React from "react";
import { ElementStyle, PANELS } from "../theme";

/**
 * Panel surface + geometry. The box is exactly the IR layout box (never larger), so
 * containment metrics stay meaningful; the visual depth comes from shadow/border.
 */
export const Box: React.FC<{
  box: [number, number, number, number];
  style: ElementStyle;
  scale?: number;
  opacity?: number;
  translateY?: number;
  children: React.ReactNode;
  padded?: boolean;
}> = ({ box, style, scale = 1, opacity = 1, translateY = 0, children, padded = true }) => {
  const [x, y, w, h] = box;
  const variant = style.variant ?? "glass";
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
        borderRadius: style.radius,
        ...PANELS[variant],
      }}
    >
      {/* warm corner glow — cheap depth cue that stays inside the box */}
      <div
        style={{
          position: "absolute",
          right: -80,
          top: -80,
          width: 220,
          height: 220,
          borderRadius: "50%",
          background: `radial-gradient(circle, ${style.accent}33 0%, transparent 70%)`,
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: style.align === "left" ? "flex-start" : "center",
          padding: padded ? "18px 26px" : 0,
          boxSizing: "border-box",
        }}
      >
        {children}
      </div>
    </div>
  );
};
