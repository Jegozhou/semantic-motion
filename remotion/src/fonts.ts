import { continueRender, delayRender, staticFile } from "remotion";

let loaded = false;

export const loadSemanticFont = async (): Promise<void> => {
  if (loaded) return;
  const handle = delayRender("loading SemanticMotion font");
  try {
    const face = new FontFace("SemanticMotion", `url(${staticFile("NotoSansSC-Regular.otf")})`, {
      weight: "100 900",
    } as FontFaceDescriptors);
    await face.load();
    (document.fonts as FontFaceSet).add(face);
    loaded = true;
  } catch (err) {
    // A missing font must not abort the render; the CSS fallback chain still works.
    console.warn("font load failed", err);
  } finally {
    continueRender(handle);
  }
};
