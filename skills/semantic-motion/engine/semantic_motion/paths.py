from __future__ import annotations

import os
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
SKILL_DIR = ENGINE_DIR.parent


def sm_home() -> Path:
    p = Path(os.environ.get("SM_HOME", Path.home() / ".semantic-motion"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = Path(os.environ.get("SM_MODELS", sm_home() / "models"))
    p.mkdir(parents=True, exist_ok=True)
    return p


MODEL_URLS = {
    "selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "text_detection_cn_ppocrv3_2023may.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/text_detection_ppocr/text_detection_cn_ppocrv3_2023may.onnx",
}


def model_path(name: str, download: bool = True) -> Path:
    """Return a local path for a small vision model, downloading it on first use."""
    p = models_dir() / name
    if p.exists() and p.stat().st_size > 1000:
        return p
    if not download:
        raise FileNotFoundError(p)
    import urllib.request

    from .util import log

    url = MODEL_URLS[name]
    log(f"downloading model {name} ...")
    tmp = p.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "semantic-motion/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, p)
    return p
