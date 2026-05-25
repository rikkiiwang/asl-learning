"""Strip the heavy `crops` array from a recognizer cache for a small Colab upload.

RecognizerA (variant a) is geometry-only — it never loads `crops`. But the crops
(N,16,2,3,64,64 uint8) are ~99% of the .npz size (~3 GB for ASL Citizen). This writes
a geometry-only twin (geom/y/participant) so the val/test + WLASL caches upload in
seconds instead of an hour.

    python scripts/slim_recog_cache.py artifacts/cache/constellation_clips.npz
    # -> artifacts/cache/constellation_clips.slim.npz
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="path to a *_clips.npz cache")
    ap.add_argument("--out", default=None, help="default: <src>.slim.npz")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out) if args.out else src.with_suffix(".slim.npz")
    z = np.load(src, allow_pickle=True)
    np.savez(out, geom=np.asarray(z["geom"]), y=np.asarray(z["y"]),
             participant=np.asarray(z["participant"]))
    mb = lambda p: Path(p).stat().st_size / 1e6
    print(f"{src.name} ({mb(src):.0f} MB) -> {out.name} ({mb(out):.1f} MB), "
          f"{len(z['y'])} clips, crops dropped")


if __name__ == "__main__":
    main()
