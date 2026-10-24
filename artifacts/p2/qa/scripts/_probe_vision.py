"""Probe: Apple Vision person segmentation + face detection on one frame.

Demo support only. This is NOT the project perception module
(src/semotion/perception.py); masks produced here are labelled
"qa-demo-vision-helper" everywhere they are used.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import Quartz
import Vision
from Foundation import NSURL


def load_cgimage(path: Path):
    url = NSURL.fileURLWithPath_(str(path))
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        raise RuntimeError(f"cannot open image {path}")
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        raise RuntimeError(f"cannot decode image {path}")
    return img


def person_mask(path: Path) -> np.ndarray:
    img = load_cgimage(path)
    req = Vision.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
    req.setOutputPixelFormat_(0x4C303038)  # kCVPixelFormatType_OneComponent8
    req.setQualityLevel_(1)  # accurate
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, err = handler.performRequests_error_([req], None)
    res = req.results()
    obs = res[0] if res else None
    if obs is None:
        raise RuntimeError(f"no segmentation result (ok={ok}, err={err})")
    pb = obs.pixelBuffer()
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        w = Quartz.CVPixelBufferGetWidth(pb)
        h = Quartz.CVPixelBufferGetHeight(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        buf = bytes(base[: bpr * h])
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr)[:, :w]
        return arr.copy()
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)


def faces(path: Path) -> list[list[float]]:
    img = load_cgimage(path)
    req = Vision.VNDetectFaceRectanglesRequest.alloc().initWithCompletionHandler_(None)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    handler.performRequests_error_([req], None)
    out = []
    for obs in req.results() or []:
        bb = obs.boundingBox()
        out.append([bb.origin.x, bb.origin.y, bb.size.width, bb.size.height])
    return out


if __name__ == "__main__":
    p = Path(sys.argv[1])
    m = person_mask(p)
    f = faces(p)
    print("mask", m.shape, "coverage", float((m > 127).mean()), "dtype", m.dtype)
    print("faces(normalized)", json.dumps(f))
    Image.fromarray(m, mode="L").save(sys.argv[2])
