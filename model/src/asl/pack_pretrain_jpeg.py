"""Compress the 7.1 GB raw pretrain cache to a ~1 GB JPEG archive for upload,
and decode it back to a byte-identical-shape frames.dat on the other side.

The training code (asl.pretrain + MemmapClipDataset) is untouched: we just make
the big cache cheap to move to Colab. JPEG q92 is lossless-enough for training a
from-scratch encoder (the source mp4s are already lossy-compressed).

Pack (local):
    python -m asl.pack_pretrain_jpeg --pack \
        --cache artifacts/cache/pretrain --out artifacts/cache/pretrain_jpeg.npz

Unpack (Colab):
    python -m asl.pack_pretrain_jpeg --unpack \
        --in artifacts/cache/pretrain_jpeg.npz --cache artifacts/cache/pretrain
"""
import argparse
import json
import os

import cv2
import numpy as np


def _load_memmap(cache):
    meta = np.load(os.path.join(cache, "meta.npz"), allow_pickle=True)
    shape = tuple(int(s) for s in meta["shape"])
    X = np.memmap(os.path.join(cache, "frames.dat"), dtype=np.uint8,
                  mode="r", shape=shape)
    return X, meta


def pack(cache, out, quality):
    X, meta = _load_memmap(cache)
    n, f = X.shape[0], X.shape[1]
    enc = np.empty((n, f), dtype=object)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    for i in range(n):
        for j in range(f):
            # store BGR so cv2.imdecode round-trips exactly (frames are RGB in
            # the cache; swap to BGR for encode, swap back on decode)
            bgr = cv2.cvtColor(X[i, j], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, params)
            if not ok:
                raise RuntimeError(f"encode failed at {i},{j}")
            enc[i, j] = buf.tobytes()
        if (i + 1) % 1000 == 0:
            print(f"  packed {i + 1}/{n}", flush=True)
    np.savez(out, frames=enc, y=meta["y"], participant=meta["participant"],
             files=meta["files"], shape=np.array(X.shape),
             size=X.shape[2], num_classes=int(meta["y"].max()) + 1)
    mb = os.path.getsize(out) / 1e6
    print(f"\npacked {n} clips -> {out} ({mb:.0f} MB, from "
          f"{os.path.getsize(os.path.join(cache, 'frames.dat'))/1e6:.0f} MB raw)")


def unpack(in_path, cache):
    d = np.load(in_path, allow_pickle=True)
    enc, y = d["frames"], d["y"]
    shape = tuple(int(s) for s in d["shape"])
    n, f, size = shape[0], shape[1], shape[2]
    os.makedirs(cache, exist_ok=True)
    X = np.memmap(os.path.join(cache, "frames.dat"), dtype=np.uint8,
                  mode="w+", shape=shape)
    for i in range(n):
        for j in range(f):
            bgr = cv2.imdecode(np.frombuffer(enc[i, j], np.uint8), cv2.IMREAD_COLOR)
            X[i, j] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (i + 1) % 1000 == 0:
            print(f"  unpacked {i + 1}/{n}", flush=True)
    X.flush()
    np.savez(os.path.join(cache, "meta.npz"), y=y, participant=d["participant"],
             files=d["files"], shape=np.array(shape))
    json.dump({"n": n, "frames": f, "size": size,
               "num_classes": int(d["num_classes"])},
              open(os.path.join(cache, "meta.json"), "w"))
    print(f"\nunpacked {n} clips -> {cache}/frames.dat ({shape})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--unpack", action="store_true")
    ap.add_argument("--cache", default="artifacts/cache/pretrain")
    ap.add_argument("--out", default="artifacts/cache/pretrain_jpeg.npz")
    ap.add_argument("--in", dest="in_path", default="artifacts/cache/pretrain_jpeg.npz")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()
    if args.pack:
        pack(args.cache, args.out, args.quality)
    elif args.unpack:
        unpack(args.in_path, args.cache)
    else:
        ap.error("pass --pack or --unpack")


if __name__ == "__main__":
    main()
