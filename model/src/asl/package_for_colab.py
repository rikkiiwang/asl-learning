"""Bundle everything Colab needs into one zip for Google Drive upload.

Includes the source package, configs, manifest + splits + norm, and the
preprocessed cache (clips.npz) — but NOT the 46 GB raw dataset. Upload the
resulting zip to Drive; the Colab notebook unzips and trains from it.

Usage:
    python -m asl.package_for_colab --out artifacts/colab_bundle.zip
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
    "artifacts/manifest/signer_splits.json",
    "artifacts/cache/clips.npz",
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
    ap.add_argument("--out", default="artifacts/colab_bundle.zip")
    args = ap.parse_args()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            add(z, p)
    print(f"Wrote {args.out} ({os.path.getsize(args.out)/1e9:.2f} GB). "
          "Upload to Drive: MyDrive/asl-model/colab_bundle.zip")


if __name__ == "__main__":
    main()
