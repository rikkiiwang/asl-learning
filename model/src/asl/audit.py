"""Phase 0 — ASL Citizen data audit.

Counts videos-per-sign and unique-signers-per-sign from the dataset's split
CSVs (no video decoding needed), intersects with the beginner candidate vocab,
and reports which signs clear the per-class bar.

Usage:
    python -m asl.audit --data-root "<path to ASL_Citizen>" \
        --min-videos 25 --min-signers 4 --out ../artifacts/audit
"""
import argparse
import glob
import os
import sys

import pandas as pd

# allow running as a file or a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asl.beginner_vocab import all_candidates  # noqa: E402


def _find_col(cols, *keywords):
    """Pick the first column whose lowercased name contains all keywords."""
    low = {c: c.lower() for c in cols}
    for c, lc in low.items():
        if all(k in lc for k in keywords):
            return c
    return None


def load_splits(data_root):
    csvs = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    if not csvs:
        raise SystemExit(f"No CSVs found under {data_root!r}. Check extraction.")
    frames = []
    for path in csvs:
        name = os.path.splitext(os.path.basename(path))[0].lower()
        split = next((s for s in ("train", "val", "test") if s in name), "unknown")
        df = pd.read_csv(path)
        df["__split__"] = split
        df["__csv__"] = os.path.basename(path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csvs)} CSV(s); columns seen: {list(combined.columns)}")
    return combined


def audit(data_root, min_videos, min_signers, out_dir):
    df = load_splits(data_root)
    gloss_col = _find_col(df.columns, "gloss") or _find_col(df.columns, "label")
    signer_col = (_find_col(df.columns, "participant")
                  or _find_col(df.columns, "signer")
                  or _find_col(df.columns, "user"))
    file_col = (_find_col(df.columns, "video", "file")
                or _find_col(df.columns, "file")
                or _find_col(df.columns, "video"))
    if gloss_col is None:
        raise SystemExit(f"Could not find a gloss/label column in {list(df.columns)}")
    print(f"Detected -> gloss={gloss_col!r} signer={signer_col!r} file={file_col!r}")

    df["_gloss"] = df[gloss_col].astype(str).str.strip().str.upper()

    agg = {"videos": (gloss_col, "size")}
    if signer_col:
        agg["signers"] = (signer_col, "nunique")
    stats = df.groupby("_gloss").agg(**agg).reset_index()
    if "signers" not in stats:
        stats["signers"] = pd.NA

    cands = set(all_candidates())
    stats["is_candidate"] = stats["_gloss"].isin(cands)
    qualifies = (stats["videos"] >= min_videos)
    if signer_col:
        qualifies &= (stats["signers"] >= min_signers)
    stats["clears_bar"] = qualifies

    os.makedirs(out_dir, exist_ok=True)
    stats.sort_values(["is_candidate", "videos"], ascending=False).to_csv(
        os.path.join(out_dir, "audit_all_glosses.csv"), index=False)

    cand_stats = stats[stats["is_candidate"]].sort_values("videos", ascending=False)
    cand_stats.to_csv(os.path.join(out_dir, "audit_candidates.csv"), index=False)

    eligible = cand_stats[cand_stats["clears_bar"]]
    missing = sorted(cands - set(stats["_gloss"]))

    print("\n=== AUDIT SUMMARY ===")
    print(f"Total distinct glosses in dataset: {len(stats)}")
    print(f"Total videos: {len(df)}")
    print(f"Candidate beginner signs defined: {len(cands)}")
    print(f"  - present in dataset:        {len(cands) - len(missing)}")
    print(f"  - clear bar (>= {min_videos} vids"
          f"{f', >= {min_signers} signers' if signer_col else ''}): {len(eligible)}")
    print(f"\nTop eligible candidates (gloss, videos, signers):")
    for _, r in eligible.head(80).iterrows():
        print(f"  {r['_gloss']:<16} {int(r['videos']):>4}  "
              f"{'' if pd.isna(r['signers']) else int(r['signers'])}")
    if missing:
        print(f"\nCandidates NOT in dataset ({len(missing)}): {', '.join(missing)}")
    if len(eligible) < 75:
        print(f"\n[WARN] Only {len(eligible)} candidates clear the bar (<75). "
              "Loosen thresholds, broaden the candidate pool, or revisit vocab size.")
    print(f"\nReports written to {out_dir}/")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--min-videos", type=int, default=25)
    ap.add_argument("--min-signers", type=int, default=4)
    ap.add_argument("--out", default="artifacts/audit")
    args = ap.parse_args()
    audit(args.data_root, args.min_videos, args.min_signers, args.out)


if __name__ == "__main__":
    main()
