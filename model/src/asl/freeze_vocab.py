"""Phase 0 — resolve candidate concepts to canonical ASL Citizen glosses and
propose a frozen pilot vocabulary.

Variant policy: ASL Citizen disambiguates senses/citation forms with numeric
suffixes (EAT1, EAT2, DOG1-4). For a beginner teaching pilot we keep ONE
canonical form per concept -- the variant with the most videos that also clears
the per-class bar -- rather than merging visually different forms into one class.

A concept's candidate glosses are those matching ^CONCEPT[0-9]*$ (so CAT matches
CAT, CAT1, CAT2 -- but NOT CATEGORY/CATCH/CATHOLIC).

Usage:
    python -m asl.freeze_vocab --data-root data/ASL_Citizen \
        --target 75 --min-videos 25 --min-signers 4 \
        --out artifacts/manifest
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asl.beginner_vocab import CANDIDATE_VOCAB  # noqa: E402


def load_all(data_root):
    frames = []
    for path in glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True):
        split = next((s for s in ("train", "val", "test")
                      if s in os.path.basename(path).lower()), "unknown")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["__split__"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Gloss"] = df["Gloss"].astype(str).str.strip().str.upper()
    return df


def _suffix_rank(gloss, concept):
    """0 for the bare concept (most standard), else the numeric variant index."""
    tail = gloss[len(concept):]
    return int(tail) if tail.isdigit() else 0


def resolve_concept(df, concept, min_videos, min_signers):
    """Return the canonical gloss for a concept, or None.

    Pins the STANDARD citation form: prefer the bare concept, then the lowest
    numeric variant (`...1`); tiebreak by most videos. (Matches what an ASL-1
    textbook teaches; the variant-count difference is negligible.)
    """
    pat = re.compile(rf"^{re.escape(concept)}[0-9]*$")
    sub = df[df["Gloss"].str.match(pat)]
    if sub.empty:
        return None
    best, max_videos = None, 0
    for gloss, g in sub.groupby("Gloss"):
        videos = len(g)
        signers = g["Participant ID"].nunique()
        if videos >= min_videos and signers >= min_signers:
            max_videos = max(max_videos, videos)
            # minimize: standard form first (low suffix), then most videos
            key = (_suffix_rank(gloss, concept), -videos)
            if best is None or key < best["_key"]:
                best = {"gloss": gloss, "videos": videos, "signers": signers,
                        "_key": key}
    if best:
        # rank membership by concept-level richness (max over variants) so the
        # standard-form pick does NOT change which concepts make the cut
        best["rank_videos"] = max_videos
    return best


def freeze(data_root, target, min_videos, min_signers, out_dir):
    df = load_all(data_root)
    rows = []
    for category, words in CANDIDATE_VOCAB.items():
        for concept in words:
            res = resolve_concept(df, concept, min_videos, min_signers)
            if res:
                rows.append({"concept": concept, "gloss": res["gloss"],
                             "category": category, "videos": res["videos"],
                             "signers": res["signers"],
                             "rank_videos": res["rank_videos"]})
    rows.sort(key=lambda r: (r["category"], -r["rank_videos"]))

    os.makedirs(out_dir, exist_ok=True)
    full = pd.DataFrame(rows)
    full.to_csv(os.path.join(out_dir, "resolved_candidates.csv"), index=False)

    # Balanced selection: round-robin across categories (richest-first within
    # each) so every category is represented rather than a few crowding out.
    by_cat_all = {}
    for r in rows:
        by_cat_all.setdefault(r["category"], []).append(r)
    for items in by_cat_all.values():
        items.sort(key=lambda r: -r["rank_videos"])
    proposed, i = [], 0
    while len(proposed) < min(target, len(rows)):
        progressed = False
        for cat in by_cat_all:
            if i < len(by_cat_all[cat]):
                proposed.append(by_cat_all[cat][i])
                progressed = True
                if len(proposed) >= target:
                    break
        if not progressed:
            break
        i += 1
    with open(os.path.join(out_dir, "proposed_vocab.json"), "w") as f:
        json.dump({"target": target, "min_videos": min_videos,
                   "min_signers": min_signers,
                   "count": len(proposed),
                   "signs": [{k: r[k] for k in ("concept", "gloss", "category",
                                                "videos", "signers")}
                             for r in proposed]}, f, indent=2)

    print(f"Resolved {len(rows)} concepts clearing the bar "
          f"(>= {min_videos} vids, >= {min_signers} signers).")
    print(f"Proposing {len(proposed)} (target {target}).\n")
    by_cat = {}
    for r in proposed:
        by_cat.setdefault(r["category"], []).append(
            f"{r['concept']}({r['gloss']},{r['videos']}v/{r['signers']}s)")
    for cat, items in by_cat.items():
        print(f"[{cat}] ({len(items)})")
        print("  " + ", ".join(items))
    print(f"\nWrote resolved_candidates.csv and proposed_vocab.json to {out_dir}/")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--target", type=int, default=75)
    ap.add_argument("--min-videos", type=int, default=25)
    ap.add_argument("--min-signers", type=int, default=4)
    ap.add_argument("--out", default="artifacts/manifest")
    args = ap.parse_args()
    freeze(args.data_root, args.target, args.min_videos, args.min_signers, args.out)


if __name__ == "__main__":
    main()
