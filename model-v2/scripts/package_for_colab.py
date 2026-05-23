"""Bundle the detector code + manifests into one small zip for Colab upload.

The image zip (detect_small.zip) is NOT included here — upload it to the same
Drive folder separately (it is already compressed, so re-zipping would be
wasteful).

Usage (from model-v2/ directory, using the project venv):
    python scripts/package_for_colab.py
    # or with an explicit output path:
    python scripts/package_for_colab.py --out artifacts/colab_detector_code.zip
"""
import argparse
import os
import zipfile

# Paths are relative to the model-v2/ root (where this script is run from).
INCLUDE = [
    "src/aslv2",
    "configs",
    "pyproject.toml",
    "artifacts/detect/train.json",
    "artifacts/detect/val.json",
    "artifacts/detect/train_small.json",
    "artifacts/detect/val_small.json",
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
        description="Package detector code + manifests for Colab."
    )
    ap.add_argument(
        "--out",
        default="artifacts/colab_detector_code.zip",
        help="Output zip path (default: artifacts/colab_detector_code.zip)",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in INCLUDE:
            add(z, p)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nWrote {args.out}  ({size_mb:.1f} MB)\n")

    # Print contents summary
    with zipfile.ZipFile(args.out) as z:
        names = z.namelist()
    # Group by top-level entry
    groups: dict[str, int] = {}
    for name in names:
        top = name.split("/")[0] if "/" in name else name
        groups[top] = groups.get(top, 0) + 1
    print("Contents summary:")
    for g, count in sorted(groups.items()):
        print(f"  {g}/  ({count} file{'s' if count != 1 else ''})")

    print("\nUpload these 2 files to your Drive folder (e.g. MyDrive/asl-detector/):")
    print(f"  {args.out}           <- code + manifests (this file)")
    print("  artifacts/detect_small.zip              <- shrunk images (~1 GB)")
    print("\nTotal upload ~1 GB (replaces the old ~10.5 GB upload).")
    print("See model-v2/COLAB.md for details.")


if __name__ == "__main__":
    main()
