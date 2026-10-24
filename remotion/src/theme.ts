export const FONT_FAMILY = "SemanticMotion, 'PingFang SC', 'Noto Sans SC', sans-serif";

export type ElementStyle = {
  accent: string;
  fg: string;
  bg: string;
  fontPx: number;
  radius: number;
  align: "left" | "center";
  weight: number;
};

export const clampText = (s: string, max = 40) => (s.length > max ? s.slice(0, max - 1) + "…" : s);
