"""Bundle the CODE + small metadata Colab needs into one small zip for Drive.

The big data caches are NOT in this zip — upload them to the same Drive folder
separately (they're already compressed, so re-zipping is pointless):
    artifacts/cache/pretrain_jpeg.npz   (~0.9 GB, the 500-class pretrain set)
    artifacts/cache/clips.npz           (~1.1 GB, the 75-class fine-tune set)

Usage:
    python -m asl.package_for_colab --out artifacts/code_bundle.zip
"""
import argparse
import os
import zipfile

INCLUDE = [
    "src/asl",
    "configs",
    "requirements.txt",
    "artifacts/manifest/manifest.json",
    "artifacts/manifest/proposed_vocab.json",
    "artifacts/manifest/norm.json",
    "artifacts/manifest/norm_roi.json",
    "artifacts/manifest/signer_splits.json",
]


def add(z, path):
    if os.path.isdir(path):
        for root, _, fs in os.walk(path):
            if "__pycache__" in root:
                continue
            for f in fs:
                fp = os.path.join(root, f)
                z.write(fp, fp)
    elif os.path.exists(path):
        z.write(path, path)
    else:
        print(f"[skip missing] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/code_bundle.zip")
    args = ap.parse_args()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            add(z, p)
    print(f"Wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB).")
    print("Upload these THREE files to Drive folder MyDrive/asl-model/ :")
    print(f"  - {args.out}")
    print("  - artifacts/cache/pretrain_jpeg.npz")
    print("  - artifacts/cache/clips.npz")


if __name__ == "__main__":
    main()
