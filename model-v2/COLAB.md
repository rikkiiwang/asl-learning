# Training Constellation v2 on Google Colab

The training code is device-agnostic (`device()` picks CUDA → MPS → CPU), so the
same `aslv2.detect.train` runs locally or on Colab. Only the data lives elsewhere.

## 1. Make the code bundle (local, one command)
```bash
cd model-v2 && .venv/bin/python scripts/package_for_colab.py
# -> model-v2/artifacts/colab_detector_code.zip  (code + manifests, ~15 MB)
```

## 2. Make the shrunk image zip (local, one command)
```bash
cd model-v2 && .venv/bin/python scripts/shrink_for_colab.py
# -> model-v2/data/detect_small/   (max_side=256, JPEG q=82, ~1.1 GB on disk)
cd model-v2 && zip -q -r artifacts/detect_small.zip data/detect_small
# -> model-v2/artifacts/detect_small.zip  (~945 MB)
```

## 3. Upload to one Drive folder (default `MyDrive/asl-detector/`)

| File | From (local) | Size |
|---|---|---|
| `colab_detector_code.zip` | `model-v2/artifacts/` | ~15 MB |
| `detect_small.zip` | `model-v2/artifacts/` | ~945 MB |

**Total ~1 GB, one-time** (replaces the old ~10.5 GB upload). It persists in Drive.
You do **not** need the original full-res WIDER or 100DOH zips.

## 4. Run the notebook
Open `model-v2/notebooks/train_detector_colab.ipynb` in Colab →
**Runtime → Change runtime type → GPU** (A100/L4/T4) → Run all.

It mounts Drive, copies the 2 zips to local Colab disk, extracts
`detect_small.zip` into `/content/data/detect_small` (which already contains
`100doh/raw/...` and `widerface/WIDER_train|val/...`), `pip install -e`s the code,
runs training with `configs/detector_colab.yaml` and
`--data-root /content/data/detect_small`, and copies `best.pt` + `history.json`
back to the Drive folder.

**Gate:** val detection-rate — head ≥ 0.85, hand ≥ 0.70. If below, increase
`width`/`epochs` in `configs/detector_colab.yaml` and re-run.

## 5. Bring the result back
Download `best.pt` (+ `history.json`) from the Drive folder into
`model-v2/artifacts/checkpoints/detector/` locally. The ASL-audit domain gate and
the recognizer (Plan 4) then run against it.

> Note: `configs/detector.yaml` uses the original full-res data (`data_root: data/detect`).
> `configs/detector_colab.yaml` uses the shrunk data (`data_root: data/detect_small`)
> and is what the notebook passes to `aslv2.detect.train`.

---

## Landmark (Stage-1.5)

Train the 21-keypoint hand landmark regressor on FreiHAND v2.

### Prerequisites

Download FreiHAND v2 from https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
and place it under `model-v2/data/landmark/freihand/`:

```
data/landmark/freihand/
    training/rgb/%08d.jpg   (130,240 images)
    training_xyz.json
    training_K.json
```

### 1. Build manifests (local, one command, requires FreiHAND)
```bash
cd model-v2 && .venv/bin/python scripts/build_landmark_manifests.py
# -> artifacts/landmark/train.json  (~117,216 records)
# -> artifacts/landmark/val.json    (~13,024 records)
```

### 2. Shrink images (local, one command, requires FreiHAND)
```bash
cd model-v2 && .venv/bin/python scripts/shrink_landmark_for_colab.py
# -> data/landmark_small/   (max_side=128, JPEG q=85)
# -> artifacts/landmark/train_small.json
# -> artifacts/landmark/val_small.json
cd model-v2 && zip -q -r artifacts/landmark_small.zip data/landmark_small
# -> artifacts/landmark_small.zip
```

### 3. Package code (local, one command)
```bash
cd model-v2 && .venv/bin/python scripts/package_for_colab.py --landmark
# -> artifacts/colab_landmark_code.zip  (code + landmark manifests, ~15 MB)
```

### 4. Upload to one Drive folder (default `MyDrive/asl-landmark/`)

| File | From (local) | Est. size |
|---|---|---|
| `colab_landmark_code.zip` | `model-v2/artifacts/` | ~15 MB |
| `landmark_small.zip` | `model-v2/artifacts/` | ~500 MB |

**Total ~500 MB, one-time** — upload persists in Drive.

### 5. Run the notebook
Open `model-v2/notebooks/train_landmark_colab.ipynb` in Colab →
**Runtime → Change runtime type → GPU (A100/L4/T4)** → Run all.

It mounts Drive, copies the 2 zips to local Colab disk, extracts
`landmark_small.zip` into `/content/data/landmark_small`,
`pip install -e`s the code, runs training with
`configs/landmark_colab.yaml --data-root /content/data/landmark_small`,
and copies `best.pt` + `history.json` back to the Drive folder.

**Gate:** val PCK@0.2 ≥ 0.80. If below, increase `width`/`epochs` in
`configs/landmark_colab.yaml` and re-run.

### 6. Bring the result back
Download `best.pt` (+ `history.json`) from the Drive folder into
`model-v2/artifacts/checkpoints/landmark/` locally.
