"""Emit notebooks/train_recognizer_colab.ipynb — the MS-ASL recognizer pipeline.

The recognizer is tiny (trains in minutes), but caching MS-ASL needs the GPU
front-end (detector+landmark) over freshly-downloaded YouTube clips, so the whole
thing runs on Colab. Run locally to (re)generate the notebook:

    cd model-v2 && .venv/bin/python notebooks/build_recog_colab_nb.py
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parent / "train_recognizer_colab.ipynb"
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = [
md("""# Constellation v2 — Recognizer training on Colab (+ MS-ASL)

Adds **MS-ASL** to the recognizer's TRAIN set (ASL Citizen + WLASL + MS-ASL). 74/75
of our signs are in MS-ASL (~3.7k instances) — roughly doubling the training data.
**val/test stay pure ASL Citizen signer-held-out**, so the result is directly
comparable to the current **test 47.9% / top-3 ≈ 69%** baseline.

Pipeline: download MS-ASL YouTube clips → cache geometry through the trained
detector+landmark → train RecognizerA (3 seeds) → save best to Drive.

## Before running — upload **1 file** to `MyDrive/asl-recognizer/`:
- `colab_recog_code.zip` — make it locally with
  `cd model-v2 && python scripts/package_for_colab.py --recog`
  (bundles code + signer splits + slim val/test & WLASL caches + the detector &
  landmark checkpoints). MS-ASL videos are fetched here on Colab — nothing else to upload.

> Use a **GPU runtime** (Runtime → Change runtime type → T4). The download step is
> the long pole (YouTube + link-rot); training itself is minutes.
"""),

code("""from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = '/content/drive/MyDrive/asl-recognizer'   # folder holding colab_recog_code.zip
CODE_DIR  = '/content/model-v2'
assert os.path.isdir(DRIVE_DIR), f'Upload colab_recog_code.zip to {DRIVE_DIR} first'
print('Drive folder:', os.listdir(DRIVE_DIR))"""),

code("""# Extract the bundle to fast local disk.
import shutil
os.makedirs(CODE_DIR, exist_ok=True)
shutil.copy(f'{DRIVE_DIR}/colab_recog_code.zip', '/content/colab_recog_code.zip')
!unzip -q -o /content/colab_recog_code.zip -d $CODE_DIR
print('extracted. checkpoints present:')
!ls -la $CODE_DIR/artifacts/checkpoints/detector/best.pt $CODE_DIR/artifacts/checkpoints/landmark/best.pt
!ls -la $CODE_DIR/artifacts/cache/*.slim.npz"""),

code("""# Install the package + yt-dlp (for MS-ASL download). ffmpeg is preinstalled on Colab.
!pip install -q -e $CODE_DIR
!pip install -q -U yt-dlp
import torch
print('CUDA:', torch.cuda.is_available(), '|',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU — switch to T4')"""),

md("""## 1. Get MS-ASL annotations (YouTube URLs + time ranges)

MS-ASL is distributed as annotations only (not videos). We clone a public mirror
that carries `MSASL_{train,val,test}.json`, then our adapter matches glosses to our
75 signs and downloads just the matched clip sections.
"""),

code("""# Clone MS-ASL annotation JSONs (URLs + timestamps; small).
!rm -rf /content/msasl_ann && git clone -q --depth 1 https://github.com/simasrazinskas/ASL-Dataset /content/msasl_ann
!ls -la /content/msasl_ann/MSASL_*.json"""),

md("""## 2. Download the matched MS-ASL clips

`--max-per-sign 40` caps per-class clips for balance and a bounded download
(~74×40 ≈ 3k attempts; expect 30–50% link-rot). Resumable: re-run to retry only the
missing ones. Bump/remove the cap for more data (slower).
"""),

code("""# Long-running: fetches only the [start,end] section of each YouTube video.
!cd $CODE_DIR && python scripts/build_msasl_subset.py \\
    --ann-dir /content/msasl_ann --max-per-sign 40

import json
clips = json.load(open(f'{CODE_DIR}/data/msasl/clips.json'))
from collections import Counter
print(f'\\ndownloaded {len(clips)} MS-ASL clips for {len(set(c[\"label_idx\"] for c in clips))}/75 signs')"""),

md("""## 3. Cache MS-ASL geometry through the trained front-end

Runs the detector + landmark over each clip and stores the 93-d per-frame geometry
(same front-end as the 47.9% baseline). SANITY gates: head-present ≥0.90, ≥1-hand
frame ≥0.95 — if these are low the boxes/keypoints aren't landing on the signer.
"""),

code("""!cd $CODE_DIR && python -m aslv2.recog.cache \\
    --clips-json data/msasl/clips.json --video-dir data/msasl/videos \\
    --detector artifacts/checkpoints/detector/best.pt \\
    --landmark artifacts/checkpoints/landmark/best.pt \\
    --out artifacts/cache/msasl_clips.npz"""),

md("""## 4. Train RecognizerA — 3 seeds (honest mean on a small dataset)

Single-run variance is ±5 pts here, so we run 3 seeds and report mean±range. Config
appends WLASL + MS-ASL to TRAIN only; val/test stay pure ASL Citizen.
"""),

code("""import json, numpy as np
seeds = [1337, 1, 2]
results = []
for s in seeds:
    out = f'artifacts/checkpoints/recog_msasl_seed{s}'
    !cd $CODE_DIR && python -m aslv2.recog.train --config configs/recog_colab.yaml \\
        --seed {s} --out-dir {out}
    h = json.load(open(f'{CODE_DIR}/{out}/history.json'))
    results.append((s, h['best_val_top1'], h['test_top1']))
    print(f'  seed {s}: val {h[\"best_val_top1\"]:.3f}  test {h[\"test_top1\"]:.3f}')

val = np.array([r[1] for r in results]); test = np.array([r[2] for r in results])
print('\\n=== MS-ASL recognizer (ASL Citizen + WLASL + MS-ASL) ===')
print(f'val  top-1: {val.mean():.3f} ± {val.std():.3f}  (per-seed {[f\"{v:.3f}\" for v in val]})')
print(f'test top-1: {test.mean():.3f} ± {test.std():.3f}  (per-seed {[f\"{v:.3f}\" for v in test]})')
print('baseline to beat: test 0.479 (ASL Citizen + WLASL)')"""),

md("""## 5. Save the best seed back to Drive"""),

code("""best_seed = max(results, key=lambda r: r[1])[0]      # pick by val top-1
src = f'{CODE_DIR}/artifacts/checkpoints/recog_msasl_seed{best_seed}'
dst = f'{DRIVE_DIR}/recog_msasl_best'
os.makedirs(dst, exist_ok=True)
for f in ['best.pt', 'history.json']:
    shutil.copy(f'{src}/{f}', f'{dst}/{f}')
# also keep the freshly-cached MS-ASL geometry (so we never re-download)
shutil.copy(f'{CODE_DIR}/artifacts/cache/msasl_clips.npz', f'{DRIVE_DIR}/msasl_clips.npz')
print(f'saved best seed {best_seed} + msasl_clips.npz to {dst}')
print('Bring recog_msasl_best/ and msasl_clips.npz back to model-v2/ to compare locally.')"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
})
nbf.write(nb, NB)
print("wrote", NB, "|", len(cells), "cells")
