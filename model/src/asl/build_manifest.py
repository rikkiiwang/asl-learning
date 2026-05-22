"""Phase 1 — build the frozen training manifest.

Takes the proposed vocab (75 glosses) + ASL Citizen split CSVs and emits a single
manifest: fixed label→index map, clip spec, and every clip (video file, signer,
split) for each sign. Splits reuse ASL Citizen's official signer-disjoint
train/val/test, filtered to our glosses (so they stay signer-disjoint).

Usage:
    python -m asl.build_manifest --data-root data/ASL_Citizen \
        --vocab artifacts/manifest/proposed_vocab.json \
        --out artifacts/manifest/manifest.json
"""
import argparse
import glob
import json
import os

import pandas as pd

CLIP_SPEC = {"frames": 16, "window_seconds": 3.0, "size": 112,
             "resample": "uniform", "layout": "NFCHW"}


def load_all(data_root):
    frames = []
    for path in glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True):
        split = next((s for s in ("train", "val", "test")
                      if s in os.path.basename(path).lower()), "unknown")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Gloss"] = df["Gloss"].astype(str).str.strip().str.upper()
    return df


def build(data_root, vocab_path, out_path):
    vocab = json.load(open(vocab_path))["signs"]
    # gloss -> (label, category); label = English concept the app prompts
    by_gloss = {s["gloss"].upper(): s for s in vocab}
    df = load_all(data_root)

    labels = sorted(s["concept"] for s in vocab)        # fixed index order
    label_idx = {lab: i for i, lab in enumerate(labels)}

    signs, problems = [], []
    for s in sorted(vocab, key=lambda x: x["concept"]):
        gloss = s["gloss"].upper()
        rows = df[df["Gloss"] == gloss]
        clips = [{"file": r["Video file"], "participant": r["Participant ID"],
                  "split": r["split"]} for _, r in rows.iterrows()]
        counts = {sp: sum(c["split"] == sp for c in clips)
                  for sp in ("train", "val", "test")}
        if counts["train"] == 0 or counts["val"] == 0 or counts["test"] == 0:
            problems.append((s["concept"], gloss, counts))
        signs.append({"label": s["concept"], "gloss": gloss,
                      "label_idx": label_idx[s["concept"]],
                      "category": s["category"], "counts": counts,
                      "clips": clips})

    manifest = {"version": "v0.1", "input": CLIP_SPEC,
                "num_classes": len(labels), "labels": labels, "signs": signs}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(manifest, open(out_path, "w"), indent=2)

    tot = sum(len(s["clips"]) for s in signs)
    tr = sum(s["counts"]["train"] for s in signs)
    va = sum(s["counts"]["val"] for s in signs)
    te = sum(s["counts"]["test"] for s in signs)
    print(f"Manifest: {len(labels)} classes, {tot} clips "
          f"(train {tr} / val {va} / test {te}).")
    print(f"Per-class avg: train {tr/len(labels):.1f}, val {va/len(labels):.1f}, "
          f"test {te/len(labels):.1f}")
    # signer-disjointness check
    sp_signers = {sp: set() for sp in ("train", "val", "test")}
    for s in signs:
        for c in s["clips"]:
            sp_signers[c["split"]].add(c["participant"])
    print("signer overlap train∩val:", len(sp_signers["train"] & sp_signers["val"]),
          "train∩test:", len(sp_signers["train"] & sp_signers["test"]),
          "val∩test:", len(sp_signers["val"] & sp_signers["test"]))
    if problems:
        print(f"\n[WARN] {len(problems)} classes missing a split:")
        for c, g, ct in problems:
            print(f"  {c} ({g}): {ct}")
    else:
        print("\nAll 75 classes have train+val+test clips.")
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--vocab", default="artifacts/manifest/proposed_vocab.json")
    ap.add_argument("--out", default="artifacts/manifest/manifest.json")
    args = ap.parse_args()
    build(args.data_root, args.vocab, args.out)


if __name__ == "__main__":
    main()
