# Constellation v2 — Dataset Download Guide

All datasets go under `model-v2/data/` (**gitignored** — never committed). The
adapters (Plan 2 Task 3, Plan 3 Task 1) convert each into a unified manifest;
you only need to download + extract into the right folder.

> **Before downloading:** confirm each dataset's license permits a
> non-commercial research/educational pilot, and we record provenance + license
> in `artifacts/{detect,landmark}/PROVENANCE.md` (PRD Req 7/15).

## Folder layout (already created)
```
model-v2/data/
  detect/
    100doh/         # hand boxes — PRIMARY (frontal/varied internet video)
    egohands/       # hand boxes — optional, egocentric (lower domain match)
    widerface/      # face/head boxes — for the "head" class (face ≈ head anchor)
  landmark/
    freihand/       # 21 hand keypoints — PRIMARY
    coco_wholebody/ # hand keypoints in-the-wild — optional
```
ASL Citizen is already on disk at `../model/data/` (the v1 workspace); the audit-
slice frames (Plan 2 Task 7) are sampled from there — nothing to download.

## What to download

| Dataset | For | Source | Approx size | Access |
|---|---|---|---|---|
| **100DOH** (100 Days Of Hands) | hand boxes (Stage-1) | https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html | several–tens of GB (start with the detection subset) | download page; check terms |
| **WIDER FACE** | head class (face boxes) | http://shuoyang1213.me/WIDERFACE/ | ~3.5 GB (train+val) | Google Drive links on page |
| **FreiHAND** | 21 keypoints (Stage-1.5) | https://lmb.informatik.uni-freiburg.de/projects/freihand/ | ~4 GB | direct download |
| EgoHands *(optional)* | extra hand boxes | https://public.roboflow.com/object-detection/hands (COCO/VOC export) | ~1 GB | direct / Roboflow |
| COCO-WholeBody *(optional)* | extra frontal hand keypoints | https://github.com/jin-s13/COCO-WholeBody | annotations small; needs COCO train2017 images (~19 GB) | annotations + COCO images |

**Minimum to start:** 100DOH (hands) + WIDER FACE (head) for the detector;
FreiHAND for the landmark model. Add the optional ones only if the ASL-audit
gates (Plan 2 Task 9 / Plan 3 Task 7) come in weak.

## Where to put files
Extract each dataset's images + annotation files directly into its folder above,
keeping the dataset's own internal structure. Examples:
- `data/detect/100doh/` → the 100DOH images dir + its annotation file(s).
- `data/detect/widerface/` → `WIDER_train/`, `WIDER_val/`, `wider_face_split/`.
- `data/landmark/freihand/` → `training/`, `evaluation/`, the `*_K.json` / `*_xyz.json` label files.

## After the files land
Tell me and I'll:
1. Write the per-dataset **adapters** that convert to the unified manifests
   (`artifacts/detect/{train,val}.json`, `artifacts/landmark/{train,val}.json`).
2. Verify counts + that both classes / keypoints are present.
3. Record provenance + license.
Then we resume the detector training (Plan 2 Task 8) — which uses MPS, so we'll
want the v1 pretrain finished first to avoid GPU contention.

## Notes
- **Domain match matters:** 100DOH (frontal-ish) > EgoHands (egocentric top-down)
  for our front-facing webcam signer. The ASL-audit gates + optional self-labeled
  fine-tune (spec §9 item 5) handle residual domain shift.
- **Partial annotation:** 100DOH images label hands (not faces); WIDER FACE labels
  faces (not hands). Training across both needs per-image class masking so the
  detector isn't penalized for an unlabeled class — handled in the loss/training task.
- **Detector input resolution** is still an open choice (train at 112² to match the
  app's frozen capture, vs internal resize) — Plan 5 Task 1 / Plan 2 config.
