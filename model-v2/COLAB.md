# Training Constellation v2 on Google Colab

The training code is device-agnostic (`device()` picks CUDA → MPS → CPU), so the
same `aslv2.detect.train` runs locally or on Colab. Only the data lives elsewhere.

## 1. Make the code bundle (local, one command)
```bash
cd model-v2 && .venv/bin/python scripts/package_for_colab.py
# -> model-v2/artifacts/colab_detector_code.zip  (code + manifests, ~8 MB)
```

## 2. Upload to one Drive folder (default `MyDrive/asl-detector/`)
| File | From (local) | Size |
|---|---|---|
| `colab_detector_code.zip` | `model-v2/artifacts/` | ~8 MB |
| `raw.zip` | `model-v2/data/detect/100doh/` | ~8.8 GB |
| `WIDER_train.zip` | `model-v2/data/detect/widerface/` | ~1.4 GB |
| `WIDER_val.zip` | `model-v2/data/detect/widerface/` | ~0.35 GB |

Total ~10.5 GB, **one-time** (it persists in Drive). You do **not** need the WIDER
annotations or the 100DOH `file.json` — the manifests already contain the boxes.

## 3. Run the notebook
Open `model-v2/notebooks/train_detector_colab.ipynb` in Colab →
**Runtime → Change runtime type → GPU** (A100/L4/T4) → Run all.

It mounts Drive, copies the zips to local Colab disk, extracts them into the
layout the manifests expect (`100doh/raw/...`, `widerface/WIDER_train|val/...`),
`pip install -e`s the code, runs training, and copies `best.pt` + `history.json`
back to the Drive folder.

**Gate:** val detection-rate — head ≥ 0.85, hand ≥ 0.70. If below, increase
`width`/`epochs` in `configs/detector.yaml` and re-run.

## 4. Bring the result back
Download `best.pt` (+ `history.json`) from the Drive folder into
`model-v2/artifacts/checkpoints/detector/` locally. The ASL-audit domain gate and
the recognizer (Plan 4) then run against it.

> Note: `configs/detector.yaml` uses `data_root: data/detect` by default; the
> notebook overrides it with `--data-root /content/data/detect`.
