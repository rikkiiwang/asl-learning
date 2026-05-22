"""Square anchor grid for the from-scratch detector (BlazePalm-style: hands are
rigid blobs, so square anchors only)."""
import numpy as np

def make_anchors(img=128, stride=16, scales=(32, 64, 96)) -> np.ndarray:
    n = img // stride
    cs = (np.arange(n) + 0.5) * stride
    cx, cy = np.meshgrid(cs, cs)
    cx, cy = cx.reshape(-1), cy.reshape(-1)
    out = []
    for cxi, cyi in zip(cx, cy):
        for s in scales:
            out.append([cxi - s / 2, cyi - s / 2, cxi + s / 2, cyi + s / 2])
    return np.asarray(out, dtype=np.float32)
