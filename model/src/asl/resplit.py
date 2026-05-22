"""Re-split the 52 ASL Citizen signers into train/val/test favoring training,
keeping splits signer-disjoint (no signer in two splits).

The official ASL Citizen split is test-heavy (built for retrieval). For our
from-scratch classifier we want more training data, so we re-partition signers.
Writes a participant->split policy file; the cache stays untouched and the
Dataset derives split from this policy.

Usage:
    python -m asl.resplit --cache artifacts/cache/clips.npz \
        --n-val 6 --n-test 8 --seed 1337 \
        --out artifacts/manifest/signer_splits.json
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="artifacts/cache/clips.npz")
    ap.add_argument("--manifest", default="artifacts/manifest/manifest.json")
    ap.add_argument("--n-val", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="artifacts/manifest/signer_splits.json")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    part = d["participant"].astype(str)
    y = d["y"]
    signers = sorted(set(part))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(signers)

    test = set(signers[:args.n_test])
    val = set(signers[args.n_test:args.n_test + args.n_val])
    train = set(signers[args.n_test + args.n_val:])
    policy = {s: ("test" if s in test else "val" if s in val else "train")
              for s in signers}
    json.dump(policy, open(args.out, "w"), indent=2)

    split = np.array([policy[p] for p in part])
    n_classes = len(json.load(open(args.manifest))["labels"])
    print(f"signers: train {len(train)} / val {len(val)} / test {len(test)}")
    for s in ("train", "val", "test"):
        m = split == s
        per = np.bincount(y[m], minlength=n_classes)
        print(f"{s}: {m.sum()} clips  per-class min/med/max "
              f"{per.min()}/{int(np.median(per))}/{per.max()}  "
              f"classes-with-0: {(per == 0).sum()}")
    # signer-disjointness is guaranteed by construction; assert anyway
    assert not (train & val) and not (train & test) and not (val & test)
    print(f"\nWrote {args.out} (signer-disjoint verified)")


if __name__ == "__main__":
    main()
