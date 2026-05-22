"""Extract only the manifest's video files from the ASL Citizen zip.

Random-access extraction via zipfile (no need to inflate the whole 46 GB archive).

Usage:
    python -m asl.extract_subset --zip data/ASL_Citizen.zip \
        --manifest artifacts/manifest/manifest.json \
        --out data/ASL_Citizen/videos
"""
import argparse
import json
import os
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/ASL_Citizen.zip")
    ap.add_argument("--manifest", default="artifacts/manifest/manifest.json")
    ap.add_argument("--out", default="data/ASL_Citizen/videos")
    ap.add_argument("--prefix", default="ASL_Citizen/videos/")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    files = {c["file"] for s in manifest["signs"] for c in s["clips"]}
    os.makedirs(args.out, exist_ok=True)

    extracted = skipped = missing = 0
    with zipfile.ZipFile(args.zip) as z:
        names = set(z.namelist())
        for fname in sorted(files):
            member = args.prefix + fname
            dest = os.path.join(args.out, fname)
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                skipped += 1
                continue
            if member not in names:
                missing += 1
                print(f"[MISSING] {member}")
                continue
            with z.open(member) as src, open(dest, "wb") as out:
                out.write(src.read())
            extracted += 1
            if extracted % 250 == 0:
                print(f"  extracted {extracted} ...")

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out)) / 1e9
    print(f"\nDone. extracted={extracted} skipped={skipped} missing={missing} "
          f"({len(files)} wanted). On disk: {total:.2f} GB in {args.out}")


if __name__ == "__main__":
    main()
