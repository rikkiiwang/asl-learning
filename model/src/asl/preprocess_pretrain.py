"""Preprocess the pretraining clips into a disk-backed np.memmap.

Unlike the 75-task npz (held in RAM), the ~500-sign set is too large for RAM, so
frames are written to a memmap on disk and read lazily at train time. Videos are
extracted from the zip on the fly if not already present.

Usage:
    python -m asl.preprocess_pretrain \
        --manifest artifacts/manifest/pretrain_manifest.json \
        --zip data/ASL_Citizen.zip --videos data/ASL_Citizen/videos \
        --out artifacts/cache/pretrain
"""
import argparse
import json
import os
import zipfile

import numpy as np

from asl.preprocess import load_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifest/pretrain_manifest.json")
    ap.add_argument("--zip", default="data/ASL_Citizen.zip")
    ap.add_argument("--videos", default="data/ASL_Citizen/videos")
    ap.add_argument("--out", default="artifacts/cache/pretrain")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--prefix", default="ASL_Citizen/videos/")
    args = ap.parse_args()

    clips = json.load(open(args.manifest))["clips"]
    n = len(clips)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.videos, exist_ok=True)
    dat_path = os.path.join(args.out, "frames.dat")
    X = np.memmap(dat_path, dtype=np.uint8, mode="w+",
                  shape=(n, args.frames, args.size, args.size, 3))
    y = np.empty(n, dtype=np.int64)
    participant = np.empty(n, dtype=object)
    files = np.empty(n, dtype=object)

    z = zipfile.ZipFile(args.zip)
    znames = set(z.namelist())
    ok = extracted = 0
    for c in clips:
        fname = c["file"]
        dest = os.path.join(args.videos, fname)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            member = args.prefix + fname
            if member in znames:
                with z.open(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
                extracted += 1
            else:
                continue
        clip = load_clip(dest, args.frames, args.size)
        if clip is None:
            continue
        X[ok] = clip
        y[ok] = c["label_idx"]
        participant[ok] = c["participant"]
        files[ok] = fname
        ok += 1
        if ok % 1000 == 0:
            print(f"  processed {ok}/{n}  (extracted {extracted} new videos) ...",
                  flush=True)
    z.close()
    X.flush()

    # trim if any failed
    meta = {"n": ok, "frames": args.frames, "size": args.size,
            "num_classes": json.load(open(args.manifest))["num_classes"]}
    np.savez(os.path.join(args.out, "meta.npz"),
             y=y[:ok], participant=participant[:ok].astype(str),
             files=files[:ok].astype(str), shape=np.array([ok, args.frames,
                                                           args.size, args.size, 3]))
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))
    print(f"\nWrote {ok} clips to {dat_path} "
          f"({os.path.getsize(dat_path)/1e9:.1f} GB) + meta. extracted {extracted} videos.")


if __name__ == "__main__":
    main()
