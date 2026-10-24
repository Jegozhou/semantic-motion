export const FONT_FAMILY = "SemanticMotion, 'PingFang SC', 'Noto Sans SC', sans-serif";

export type Variant = "glass" | "solid" | "outline";

export type ElementStyle = {
  accent: string;
  fg: string;
  bg: string;
  fontPx: number;
  radius: number;
  align: "left" | "center";
  weight: number;
  variant?: Variant;
  kicker?: string;
};

/** Panel treatments: one design language, three weights of emphasis. */
export const PANELS: Record<Variant, React.CSSProperties> = {
  glass: {
    background:
      "linear-gradient(160deg, rgba(24,29,42,0.90) 0%, rgba(12,15,24,0.94) 55%, rgba(9,11,18,0.96) 100%)",
    border: "1px solid rgba(242,178,51,0.32)",
    boxShadow:
      "0 26px 70px rgba(0,0,0,0.52), inset 0 1px 0 rgba(255,255,255,0.10), inset 0 -1px 0 rgba(0,0,0,0.35)",
    backdropFilter: "blur(14px) saturate(1.15)",
  },
  solid: {
    background: "linear-gradient(180deg, rgba(16,19,28,0.97), rgba(10,12,19,0.99))",
    border: "1px solid rgba(255,255,255,0.08)",
    boxShadow: "0 22px 60px rgba(0,0,0,0.55)",
  },
  outline: {
    background: "rgba(8,10,16,0.30)",
    border: "2px solid rgba(242,178,51,0.78)",
    boxShadow: "0 18px 50px rgba(0,0,0,0.35)",
    backdropFilter: "blur(6px)",
  },
};

export const KICKER_LETTER_SPACING = "0.16em";
export const clampText = (s: string, max = 40) => (s.length > max ? s.slice(0, max - 1) + "…" : s);

/** Per-character reveal: characters rise out of a mask, one after another. */
export const staggerDelay = (index: number, total: number, totalFrames: number) =>
  total <= 1 ? 0 : (index / Math.max(1, total - 1)) * totalFrames * 0.55;
