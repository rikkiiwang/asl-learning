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


# ---------------------------------------------------------------------------
# Resolution-aware helpers: anchor scales are absolute pixels, so they must
# track the input size. Deriving them from `img` here makes this the *single
# source of truth* — train and inference both call anchors_for(img), so they
# can never drift to mismatched scales (which would silently wreck matching).
# ---------------------------------------------------------------------------

BASE_IMG = 128
BASE_SCALES = (32, 64, 96)   # tuned for the 128² baseline


def scales_for(img: int) -> tuple[float, ...]:
    """Anchor scales (px) for a given input size, scaled from the 128² baseline.

    A 1.5× larger input ⇒ 1.5× larger anchors, so a hand that was ~32 px at 128²
    is still matched by the smallest anchor at 192².
    """
    f = img / BASE_IMG
    return tuple(s * f for s in BASE_SCALES)


def anchors_for(img: int, stride: int = 16) -> np.ndarray:
    """`make_anchors` with scales auto-derived from `img` (see scales_for)."""
    return make_anchors(img=img, stride=stride, scales=scales_for(img))
