"""Bundle code + manifests into one small zip for Colab upload.

Supports two modes via --landmark flag:
  Detector (default): packages code + detect manifests → colab_detector_code.zip
  Landmark:           packages code + landmark manifests → colab_landmark_code.zip

The image zip is NOT included here — upload it separately (already compressed).

Usage (from model-v2/ directory, using the project venv):
    # Detector (default):
    python scripts/package_for_colab.py
    python scripts/package_for_colab.py --out artifacts/colab_detector_code.zip

    # Landmark (run AFTER build_landmark_manifests.py + shrink_landmark_for_colab.py):
    python scripts/package_for_colab.py --landmark
    python scripts/package_for_colab.py --landmark --out artifacts/colab_landmark_code.zip
"""
import argparse
import os
import zipfile

# Paths relative to the model-v2/ root (where this script is run from).
INCLUDE_DETECTOR = [
    "src/aslv2",
    "configs",
    "pyproject.toml",
    "artifacts/detect/train.json",
    "artifacts/detect/val.json",
    "artifacts/detect/train_small.json",
    "artifacts/detect/val_small.json",
]

INCLUDE_LANDMARK = [
    "src/aslv2",
    "configs",
    "pyproject.toml",
    "artifacts/landmark/train_small.json",
    "artifacts/landmark/val_small.json",
]

# Recognizer (MS-ASL) bundle: code + splits + slim val/test & WLASL caches +
# the trained front-end (detector+landmark) used to cache MS-ASL on Colab.
INCLUDE_RECOG = [
    "src/aslv2",
    "configs",
    "pyproject.toml",
    "scripts/build_msasl_subset.py",   # the notebook runs this to fetch MS-ASL clips
    "artifacts/manifest/manifest.json",       # 75-sign gloss->label_idx map (adapter)
    "artifacts/manifest/signer_splits.json",
    "artifacts/cache/constellation_clips.slim.npz",
    "artifacts/cache/wlasl_clips.slim.npz",
    "artifacts/checkpoints/detector/best.pt",
    "artifacts/checkpoints/landmark/best.pt",
]


def add(z: zipfile.ZipFile, path: str) -> None:
    """Add a file or directory tree (skipping __pycache__) to the zip."""
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            # Skip bytecode caches and egg-info directories
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "*.egg-info")]
            if "__pycache__" in root or ".egg-info" in root:
                continue
            for fname in files:
                if fname.endswith(".pyc"):
                    continue
                fp = os.path.join(root, fname)
                z.write(fp, fp)
    elif os.path.exists(path):
        z.write(path, path)
    else:
        print(f"[skip missing] {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Package code + manifests for Colab (detector or landmark)."
    )
    ap.add_argument(
        "--landmark",
        action="store_true",
        help="Package landmark code + manifests instead of detector.",
    )
    ap.add_argument(
        "--recog",
        action="store_true",
        help="Package recognizer code + splits + slim caches + front-end checkpoints.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output zip path. "
            "Defaults to artifacts/colab_detector_code.zip (or colab_landmark_code.zip "
            "when --landmark is set)."
        ),
    )
    args = ap.parse_args()

    if args.recog:
        include  = INCLUDE_RECOG
        out_path = args.out or "artifacts/colab_recog_code.zip"
        img_zip  = None
        img_note = "MS-ASL videos are downloaded ON Colab (yt-dlp) — nothing to upload"
        drive    = "MyDrive/asl-recognizer/"
    elif args.landmark:
        include  = INCLUDE_LANDMARK
        out_path = args.out or "artifacts/colab_landmark_code.zip"
        img_zip  = "artifacts/landmark_small.zip"
        img_note = "shrunk FreiHAND images (~500 MB estimate)"
        drive    = "MyDrive/asl-landmark/"
    else:
        include  = INCLUDE_DETECTOR
        out_path = args.out or "artifacts/colab_detector_code.zip"
        img_zip  = "artifacts/detect_small.zip"
        img_note = "shrunk images (~1 GB)"
        drive    = "MyDrive/asl-detector/"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in include:
            add(z, p)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path}  ({size_mb:.1f} MB)\n")

    # Print contents summary
    with zipfile.ZipFile(out_path) as z:
        names = z.namelist()
    groups: dict[str, int] = {}
    for name in names:
        top = name.split("/")[0] if "/" in name else name
        groups[top] = groups.get(top, 0) + 1
    print("Contents summary:")
    for g, count in sorted(groups.items()):
        print(f"  {g}/  ({count} file{'s' if count != 1 else ''})")

    if img_zip:
        print(f"\nUpload these 2 files to your Drive folder (e.g. {drive}):")
        print(f"  {out_path}    <- code + manifests (this file)")
        print(f"  {img_zip}     <- {img_note}")
    else:
        print(f"\nUpload this 1 file to your Drive folder (e.g. {drive}):")
        print(f"  {out_path}    <- code + splits + slim caches + front-end checkpoints")
        print(f"  ({img_note})")
    print("\nSee model-v2/COLAB.md for details.")


if __name__ == "__main__":
    main()
