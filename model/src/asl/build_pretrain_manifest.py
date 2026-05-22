"""Build the encoder-pretraining manifest.

Selects up to N signs (always including our 75) with the most clips from signers
that are NOT in our 75-task val/test sets — so pretraining never sees a held-out
signer (no leakage into the learned features). Emits pretrain_manifest.json.

Usage:
    python -m asl.build_pretrain_manifest --data-root data/ASL_Citizen \
        --signer-splits artifacts/manifest/signer_splits.json \
        --vocab artifacts/manifest/proposed_vocab.json \
        --max-signs 500 --min-clips 12 --out artifacts/manifest/pretrain_manifest.json
"""
import argparse
import glob
import json
import os

import pandas as pd


def load_all(data_root):
    frames = []
    for path in glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Gloss"] = df["Gloss"].astype(str).str.strip().str.upper()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/ASL_Citizen")
    ap.add_argument("--signer-splits", default="artifacts/manifest/signer_splits.json")
    ap.add_argument("--vocab", default="artifacts/manifest/proposed_vocab.json")
    ap.add_argument("--max-signs", type=int, default=500)
    ap.add_argument("--min-clips", type=int, default=12)
    ap.add_argument("--out", default="artifacts/manifest/pretrain_manifest.json")
    args = ap.parse_args()

    df = load_all(args.data_root)
    splits = json.load(open(args.signer_splits))
    heldout = {s for s, sp in splits.items() if sp in ("val", "test")}
    # keep only clips from non-held-out signers
    usable = df[~df["Participant ID"].isin(heldout)].copy()

    our75 = {s["gloss"].upper() for s in json.load(open(args.vocab))["signs"]}

    counts = usable.groupby("Gloss").size().sort_values(ascending=False)
    counts = counts[counts >= args.min_clips]

    # always include our 75 (that clear min), then fill with the richest others
    chosen = [g for g in our75 if g in counts.index]
    for g in counts.index:
        if len(chosen) >= args.max_signs:
            break
        if g not in our75:
            chosen.append(g)
    chosen = sorted(set(chosen))

    label_idx = {g: i for i, g in enumerate(chosen)}
    sub = usable[usable["Gloss"].isin(set(chosen))]
    clips = [{"file": r["Video file"], "participant": r["Participant ID"],
              "gloss": r["Gloss"], "label_idx": label_idx[r["Gloss"]]}
             for _, r in sub.iterrows()]

    manifest = {"num_classes": len(chosen), "labels": chosen,
                "our75_included": sum(g in label_idx for g in our75),
                "clips": clips}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(manifest, open(args.out, "w"))
    print(f"Pretraining set: {len(chosen)} signs, {len(clips)} clips "
          f"(held-out signers excluded: {len(heldout)}).")
    print(f"  our-75 covered: {manifest['our75_included']}/75")
    print(f"  clips/sign: min {sub.groupby('Gloss').size().min()}, "
          f"median {int(sub.groupby('Gloss').size().median())}, "
          f"max {sub.groupby('Gloss').size().max()}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
