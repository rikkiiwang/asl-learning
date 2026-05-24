# ASL v1 — Data Scaling + Optical-Flow Motion Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push held-out test top-1 past 46% by strengthening augmentation, growing the encoder pretraining set to ~1500 glosses, and adding a classical optical-flow second stream — all from scratch, ONNX <10MB, signer-held-out.

**Architecture:** Three sequential phases. (A) strengthen the existing train-time augmentation; (B) re-pretrain the CNN encoder on a ~1500-gloss ASL Citizen slice with checkpoint/resume; (C) add a Farneback optical-flow stream as a second `FrameEncoder` with late fusion, warm-started from the pretrained RGB encoder body.

**Tech Stack:** PyTorch 2.7 (MPS local / CUDA Colab T4), OpenCV (classical ROI crop + Farneback flow), NumPy, ONNX Runtime, pytest.

**Spec:** `docs/superpowers/specs/2026-05-24-asl-v1-data-scaling-and-motion-stream-design.md`

**Conventions in this repo:**
- Run modules with `PYTHONPATH=src python -m asl.<module>`. The interpreter is anaconda `python` (NOT `python3`).
- Tests are new (`tests/` does not exist yet); run with `PYTHONPATH=src python -m pytest tests/ -v`.
- Eval for accuracy claims always via `asl.analysis.run_analysis` (no DataLoader workers).
- Commit messages end with the Co-Authored-By trailer (see each commit step).

---

## File structure

| File | Responsibility | Change |
|---|---|---|
| `src/asl/dataset.py` | clip loading + train aug; new two-stream (rgb+flow) loading | Modify |
| `src/asl/preprocess.py` | decode → ROI crop → cache; **new** `compute_flow`; flow + flow-norm output | Modify |
| `src/asl/preprocess_pretrain.py` | memmap pretrain cache (RGB only — no flow) | No change (B reuses as-is) |
| `src/asl/build_pretrain_manifest.py` | pretrain manifest (already `--max-signs`) | No code change; run with 1500 |
| `src/asl/pretrain.py` | encoder pretrain; **new** checkpoint/resume | Modify |
| `src/asl/model.py` | `FrameEncoder(in_ch)`; **new** `TwoStreamClassifier`; `build(two_stream=...)` | Modify |
| `src/asl/train.py` | two-stream batches + dual-encoder warm start | Modify |
| `src/asl/export.py` | two-input ONNX + flow in meta.json | Modify |
| `configs/finetune_twostream.yaml` | two-stream fine-tune config | Create |
| `tests/test_augment.py` | augment_clip invariants | Create |
| `tests/test_flow.py` | compute_flow correctness | Create |
| `tests/test_model_twostream.py` | encoder in_ch + two-stream forward | Create |
| `MODEL_WORKSTREAM.md` | flow inference-contract (already added) | Verify only |

---

# PHASE A — Strengthen augmentation (no re-cache)

Ships independently: a measurable A/B on the existing `clips_roi.npz`.

### Task A1: Make `augment_clip` testable and add rotation

**Files:**
- Modify: `src/asl/dataset.py:1-36` (imports + `augment_clip`)
- Test: `tests/test_augment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import numpy as np
from asl.dataset import augment_clip


def test_augment_preserves_shape_dtype_range():
    rng = np.random.default_rng(0)
    clip = rng.random((16, 100, 100, 3)).astype(np.float32)
    out = augment_clip(clip, size=112, rng=np.random.default_rng(0))
    assert out.shape == (16, 112, 112, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_augment_is_deterministic_under_seed():
    clip = np.random.default_rng(1).random((16, 100, 100, 3)).astype(np.float32)
    a = augment_clip(clip, 112, rng=np.random.default_rng(42))
    b = augment_clip(clip, 112, rng=np.random.default_rng(42))
    assert np.allclose(a, b)


def test_augment_same_geometric_transform_across_frames():
    # A static clip (all frames identical) must stay identical across frames
    # after augmentation (one transform applied to the whole clip).
    frame = np.random.default_rng(2).random((100, 100, 3)).astype(np.float32)
    clip = np.repeat(frame[None], 16, axis=0)
    out = augment_clip(clip, 112, rng=np.random.default_rng(7))
    assert np.allclose(out[0], out[1]) and np.allclose(out[0], out[15])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "$(git rev-parse --show-toplevel)" && PYTHONPATH=src python -m pytest tests/test_augment.py -v`
Expected: FAIL — `augment_clip()` got an unexpected keyword argument `rng` (current signature is `(clip, size)`).

- [ ] **Step 3: Rewrite `augment_clip` with an `rng` param and rotation**

Add `import cv2` to the imports at the top of `dataset.py` (after `import os`). Replace the existing `augment_clip` (lines 21-36) with:

```python
def augment_clip(clip, size, rng=None):
    """clip: (F,H,W,3) float32 in 0-1. ONE shared geometric+photometric transform
    across all frames. No horizontal flip (can change a sign's meaning)."""
    rng = rng if rng is not None else np.random.default_rng()
    # photometric: brightness + contrast
    b = rng.uniform(-0.12, 0.12)
    c = rng.uniform(0.85, 1.15)
    clip = np.clip((clip - 0.5) * c + 0.5 + b, 0, 1).astype(np.float32)
    F, H, W, _ = clip.shape
    # geometric: small rotation, same angle for every frame
    ang = float(rng.uniform(-10.0, 10.0))
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
    clip = np.stack([cv2.warpAffine(clip[i], M, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT)
                     for i in range(F)], 0)
    # geometric: random resized crop (zoom + translate)
    scale = float(rng.uniform(0.85, 1.0))
    ch, cw = int(H * scale), int(W * scale)
    y0 = int(rng.integers(0, H - ch + 1))
    x0 = int(rng.integers(0, W - cw + 1))
    clip = clip[:, y0:y0 + ch, x0:x0 + cw, :]
    t = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)
    t = torch.nn.functional.interpolate(t, size=(size, size),
                                        mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).numpy().astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_augment.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_augment.py src/asl/dataset.py
git commit -m "feat(aug): add rotation + seedable rng to augment_clip

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task A2: A/B measure augmentation strength on the stacked model

**Files:**
- No code change. This is a measurement task producing a number for the model story.

- [ ] **Step 1: Back up the current best, THEN re-fine-tune with the strengthened aug**

Training writes to `artifacts/checkpoints/finetune_roi/`, so back up the current 0.458 checkpoint FIRST:
```bash
cp artifacts/checkpoints/finetune_roi/best.pt artifacts/checkpoints/finetune_roi/best_preaug.pt
PYTHONPATH=src python -u -m asl.train --config configs/finetune_roi.yaml
```
Expected: a `BEST val_top1=… TEST top1=…` line. Compare TEST to the prior 0.458.

- [ ] **Step 2: Run the full analysis**

Run:
```bash
PYTHONPATH=src python -c "
import numpy as np; from asl import analysis
r = analysis.run_analysis(ckpt='artifacts/checkpoints/finetune_roi/best.pt',
    cache='artifacts/cache/clips_roi.npz', norm='artifacts/manifest/norm_roi.json')
pc = r['per_class'][~np.isnan(r['per_class'])]
print('test', round(r['acc']['test'],3), 'dead', int((pc==0).sum()))
"
```
Expected: prints test acc + dead-sign count. **Decision gate:** if strengthened aug does not beat 0.458 / 7-dead, keep `best_preaug.pt` as the Phase-A result and record that rotation didn't help (a valid finding). Either way, record the number.

- [ ] **Step 3: Commit the kept checkpoint pointer (no code)**

```bash
git add -A && git commit -m "chore(aug): record strengthened-aug A/B result on 75-class fine-tune

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# PHASE B — Bigger pretrain (~1500 glosses) with checkpoint/resume

### Task B1: Add checkpoint/resume to `pretrain.py`

**Files:**
- Modify: `src/asl/pretrain.py:42-106`
- Test: `tests/test_pretrain_resume.py`

- [ ] **Step 1: Write the failing test for the resume state helper**

```python
# tests/test_pretrain_resume.py
import os
import torch
from asl.pretrain import save_resume, load_resume
from asl.model import build


def test_resume_roundtrip(tmp_path):
    model = build(10, emb=384, head="attn", dropout=0.2, width=48)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = os.path.join(tmp_path, "resume.pt")
    save_resume(path, model, opt, epoch=5, best=0.42)
    model2 = build(10, emb=384, head="attn", dropout=0.2, width=48)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    ep, best = load_resume(path, model2, opt2, map_location="cpu")
    assert ep == 5 and abs(best - 0.42) < 1e-9
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_pretrain_resume.py -v`
Expected: FAIL — `cannot import name 'save_resume'`.

- [ ] **Step 3: Add the helpers and wire `--resume`**

Add these two functions to `pretrain.py` (above `main`):

```python
def save_resume(path, model, opt, epoch, best):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "best": best}, path)


def load_resume(path, model, opt, map_location):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    return ck["epoch"], ck["best"]
```

In `main`, add the arg (next to the other `ap.add_argument` calls):
```python
    ap.add_argument("--resume", default=None,
                    help="path to a resume.pt to continue from")
```

Replace the training-loop preamble `os.makedirs(args.out, exist_ok=True)` / `best = 0.0` / `for ep in range(args.epochs):` block (lines 83-85) with:

```python
    os.makedirs(args.out, exist_ok=True)
    resume_path = os.path.join(args.out, "resume.pt")
    best, start_ep = 0.0, 0
    if args.resume and os.path.exists(args.resume):
        start_ep, best = load_resume(args.resume, model, opt, dev)
        start_ep += 1
        print(f"resumed from {args.resume} at ep {start_ep} (best {best:.3f})")
    for ep in range(start_ep, args.epochs):
```

At the end of each epoch (after the `if vacc >= best:` save block, still inside the loop), add a resume snapshot every epoch:
```python
        save_resume(resume_path, model, opt, ep, best)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_pretrain_resume.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_pretrain_resume.py src/asl/pretrain.py
git commit -m "feat(pretrain): checkpoint/resume for long Colab runs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task B2: Build the ~1500-gloss pretrain manifest

**Files:** none (run existing module).

- [ ] **Step 1: Build the manifest**

Run:
```bash
PYTHONPATH=src python -m asl.build_pretrain_manifest \
  --max-signs 1500 --min-clips 12 \
  --out artifacts/manifest/pretrain_manifest_1500.json
```
Expected: prints `Pretraining set: <~1500> signs, <N> clips`, `our-75 covered: 75/75`, and a clips/sign summary. Verify the sign count is ~1500 and our-75 is fully covered.

- [ ] **Step 2: Commit the manifest**

```bash
git add artifacts/manifest/pretrain_manifest_1500.json
git commit -m "data(pretrain): 1500-gloss manifest (held-out signers excluded)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task B3: Preprocess the 1500-gloss ROI memmap and pack for Colab

**Files:** none (run existing modules). This is heavy/long (local M4).

- [ ] **Step 1: Preprocess to memmap (ROI-cropped)**

Run:
```bash
PYTHONPATH=src python -m asl.preprocess_pretrain \
  --manifest artifacts/manifest/pretrain_manifest_1500.json \
  --zip data/ASL_Citizen.zip --videos data/ASL_Citizen/videos \
  --out artifacts/cache/pretrain_1500 --roi
```
Expected: ends with `Wrote <N> clips to artifacts/cache/pretrain_1500/frames.dat (~23 GB) + meta.`

- [ ] **Step 2: JPEG-pack for upload**

Run:
```bash
PYTHONPATH=src python -m asl.pack_pretrain_jpeg --pack \
  --cache artifacts/cache/pretrain_1500 \
  --out artifacts/cache/pretrain_1500_jpeg.npz
```
Expected: writes a compact npz (~3 GB) and prints integrity info.

- [ ] **Step 3: Verify pack/unpack integrity (sanity, no commit of caches)**

Run:
```bash
PYTHONPATH=src python -m asl.pack_pretrain_jpeg --unpack \
  --in artifacts/cache/pretrain_1500_jpeg.npz \
  --cache "$CLAUDE_JOB_DIR/verify_1500" 2>&1 | tail -5
```
Expected: prints `y match True` and a small decode error. (Do not commit caches — they're large artifacts, not source.)

### Task B4: Re-pretrain the encoder on Colab T4 (1500-way)

**Files:** `model/notebooks/pretrain_colab.ipynb` (reuse; update cache name + add `--resume`).

- [ ] **Step 1: Update the Colab notebook cells**

In `pretrain_colab.ipynb`, the copy cell should fetch `pretrain_1500_jpeg.npz` and `clips_roi.npz`; the unpack cell targets `artifacts/cache/pretrain_1500`; the pretrain cell becomes:
```python
!PYTHONPATH=src python -u -m asl.pretrain --cache artifacts/cache/pretrain_1500 \
    --norm artifacts/manifest/norm_roi.json \
    --epochs 45 --warmup 4 --batch-size 64 --lr 0.004 \
    --out artifacts/checkpoints/pretrain \
    --resume artifacts/checkpoints/pretrain/resume.pt
```
(`--resume` is harmless on a fresh run — the file won't exist yet — and lets a reconnect continue.)

- [ ] **Step 2: Run on Colab, then fine-tune**

Upload the three files (`code_bundle.zip`, `pretrain_1500_jpeg.npz`, `clips_roi.npz`), run pretrain → fine-tune `configs/finetune_roi.yaml`. Bring back `encoder.pt`, `best.pt`, `history.json`.

- [ ] **Step 3: Place results locally and run analysis**

```bash
# after copying encoder.pt -> artifacts/checkpoints/pretrain/, best.pt -> artifacts/checkpoints/finetune_roi/
PYTHONPATH=src python -c "
import numpy as np; from asl import analysis
r = analysis.run_analysis(ckpt='artifacts/checkpoints/finetune_roi/best.pt',
    cache='artifacts/cache/clips_roi.npz', norm='artifacts/manifest/norm_roi.json')
pc = r['per_class'][~np.isnan(r['per_class'])]
print('1500-pretrain  test', round(r['acc']['test'],3), 'dead', int((pc==0).sum()))
"
```
Expected: TEST top-1; compare to 0.458 (500-class pretrain). Record the number.

---

# PHASE C — Optical-flow motion stream

### Task C1: Add `compute_flow` to `preprocess.py`

**Files:**
- Modify: `src/asl/preprocess.py` (add function after `motion_roi_box`)
- Test: `tests/test_flow.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flow.py
import numpy as np
from asl.preprocess import compute_flow


def test_flow_shape_and_zero_first_frame():
    frames = (np.random.default_rng(0).random((16, 64, 64, 3)) * 255).astype(np.uint8)
    flow = compute_flow(frames)
    assert flow.shape == (16, 64, 64, 2)
    assert flow.dtype == np.float32
    assert np.allclose(flow[0], 0.0)            # first frame has no predecessor


def test_flow_zero_for_static_clip():
    frame = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    frames = np.repeat(frame[None], 8, axis=0)
    flow = compute_flow(frames)
    assert np.abs(flow).max() < 0.5            # no motion -> ~zero flow


def test_flow_detects_horizontal_shift():
    # A vertical bar shifted right by 4 px between frames -> positive dx where the bar is.
    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[:, 20:24] = 255
    shifted = np.zeros((64, 64, 3), dtype=np.uint8)
    shifted[:, 24:28] = 255
    frames = np.stack([base, shifted], 0)
    flow = compute_flow(frames)
    # mean dx over the central region should be clearly positive
    assert flow[1, 20:44, 18:30, 0].mean() > 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_flow.py -v`
Expected: FAIL — `cannot import name 'compute_flow'`.

- [ ] **Step 3: Implement `compute_flow`**

Add to `preprocess.py` (after `motion_roi_box`, before `load_clip`):

```python
def compute_flow(frames_rgb):
    """frames_rgb: (k,H,W,3) uint8 RGB. Returns (k,H,W,2) float32 Farneback flow
    (dx,dy). flow[0] is zeros; flow[t] is motion from frame t-1 -> t.

    Classical algorithm, no learned weights (from-scratch-compliant)."""
    k, H, W, _ = frames_rgb.shape
    flow = np.zeros((k, H, W, 2), dtype=np.float32)
    prev = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for t in range(1, k):
        cur = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        flow[t] = cv2.calcOpticalFlowFarneback(
            prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev = cur
    return flow
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_flow.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_flow.py src/asl/preprocess.py
git commit -m "feat(flow): classical Farneback optical-flow computation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task C2: Extend the 75-class preprocessing to cache flow (float16) + flow norm

**Files:**
- Modify: `src/asl/preprocess.py` (`load_clip`, `main`)

- [ ] **Step 1: Add a `with_flow` return to `load_clip`**

Change the `load_clip` signature to `def load_clip(path, k, size, roi=False, with_flow=False):` and, just before `return out`, insert:

```python
    if with_flow:
        flow = compute_flow(out)                       # (k,size,size,2) float32
        return out, flow.astype(np.float16)
    return out
```

(Callers that pass `with_flow=False` are unaffected — they still get `out` only.)

- [ ] **Step 2: Add `--with-flow` to `main` and store `Xflow` + flow norm**

In `main`, add the arg:
```python
    ap.add_argument("--with-flow", action="store_true",
                    help="also compute+store Farneback flow (2ch float16)")
```
Allocate a flow buffer alongside `X` (after the `X = np.empty(...)` line):
```python
    Xflow = (np.empty((len(clips), args.frames, args.size, args.size, 2),
                      dtype=np.float16) if args.with_flow else None)
```
In the loop, replace the `clip = load_clip(...)` + assignment block with:
```python
        res = load_clip(os.path.join(args.videos, fname), args.frames, args.size,
                        roi=args.roi, with_flow=args.with_flow)
        if res is None:
            print(f"[SKIP] could not decode {fname}")
            continue
        if args.with_flow:
            clip, fl = res
        else:
            clip = res
        X[ok], y[ok], split[ok], participant[ok], files[ok] = clip, lab, sp, pid, fname
        if args.with_flow:
            Xflow[ok] = fl
        ok += 1
```
After the trim line (`X, y, split, participant, files = ...`), add:
```python
    if args.with_flow:
        Xflow = Xflow[:ok]
```
Change the `np.savez_compressed(...)` call to include flow when present:
```python
    save_kw = dict(X=X, y=y, split=split.astype(str),
                   participant=participant.astype(str), files=files.astype(str))
    if args.with_flow:
        save_kw["Xflow"] = Xflow
    np.savez_compressed(args.out, **save_kw)
```
After the RGB norm block, add flow norm (only when flow present) and write it to a sidecar:
```python
    if args.with_flow:
        flow_tr = Xflow[split == "train"].reshape(-1, 2).astype(np.float64)
        flow_norm = {"mean": flow_tr.mean(0).tolist(),
                     "std": (flow_tr.std(0) + 1e-6).tolist()}
        flow_norm_out = args.norm_out.replace(".json", "_flow.json")
        json.dump(flow_norm, open(flow_norm_out, "w"), indent=2)
        print(f"flow norm -> {flow_norm_out}: {flow_norm}")
```

- [ ] **Step 3: Build the flow-augmented 75-class cache**

Run:
```bash
PYTHONPATH=src python -m asl.preprocess \
  --manifest artifacts/manifest/manifest.json \
  --videos data/ASL_Citizen/videos \
  --out artifacts/cache/clips_roi_flow.npz \
  --norm-out artifacts/manifest/norm_roi.json \
  --roi --with-flow
```
Expected: writes `clips_roi_flow.npz` (RGB uint8 + `Xflow` float16) and `artifacts/manifest/norm_roi_flow.json`. (RGB norm matches the existing `norm_roi.json`.)

- [ ] **Step 4: Commit code (not the cache)**

```bash
git add src/asl/preprocess.py
git commit -m "feat(flow): cache 2ch flow + flow-norm in 75-class preprocess

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task C3: Add `in_ch` to `FrameEncoder` and a `TwoStreamClassifier`

**Files:**
- Modify: `src/asl/model.py:25-46` (`FrameEncoder`), add `TwoStreamClassifier`, update `build`
- Test: `tests/test_model_twostream.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_twostream.py
import torch
from asl.model import FrameEncoder, build, TwoStreamClassifier


def test_encoder_accepts_2ch_input():
    enc = FrameEncoder(emb=384, width=48, in_ch=2)
    out = enc(torch.randn(4, 2, 112, 112))
    assert out.shape == (4, 384)


def test_two_stream_forward_shape():
    m = build(75, emb=384, width=48, two_stream=True)
    assert isinstance(m, TwoStreamClassifier)
    rgb = torch.randn(2, 16, 3, 112, 112)
    flow = torch.randn(2, 16, 2, 112, 112)
    assert m(rgb, flow).shape == (2, 75)


def test_two_stream_under_10mb_fp32():
    m = build(75, two_stream=True)
    n = sum(p.numel() for p in m.parameters())
    assert n * 4 / 1e6 < 10.0          # fp32 under the 10MB browser cap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_model_twostream.py -v`
Expected: FAIL — `FrameEncoder.__init__() got an unexpected keyword argument 'in_ch'` / `cannot import name 'TwoStreamClassifier'`.

- [ ] **Step 3: Implement the changes**

In `FrameEncoder.__init__`, change the signature and stem:
```python
    def __init__(self, emb=384, width=48, in_ch=3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, 2, 1, bias=False),    # 56
            nn.BatchNorm2d(w), nn.ReLU(inplace=True))
```
(Leave `body`, `pool`, `forward` unchanged. Update the docstring's `112x112x3` to `112x112xin_ch`.)

Add the two-stream model (after `SignClassifier`):
```python
class TwoStreamClassifier(nn.Module):
    """RGB appearance stream + optical-flow motion stream, late-fused."""
    def __init__(self, num_classes, emb=384, width=48, dropout=0.3, flow_ch=2):
        super().__init__()
        self.encoder = FrameEncoder(emb, width, in_ch=3)        # name matches ckpt
        self.encoder_flow = FrameEncoder(emb, width, in_ch=flow_ch)
        self.pool = AttnPool(emb)
        self.pool_flow = AttnPool(emb)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * emb, num_classes)

    def forward(self, rgb, flow):                    # (B,F,3,H,W), (B,F,2,H,W)
        rgb, flow = rgb.contiguous(), flow.contiguous()
        b, f = rgb.shape[:2]
        er = self.encoder(rgb.reshape(b * f, *rgb.shape[2:])).reshape(b, f, -1)
        ef = self.encoder_flow(flow.reshape(b * f, *flow.shape[2:])).reshape(b, f, -1)
        fused = torch.cat([self.pool(er), self.pool_flow(ef)], dim=1)
        return self.fc(self.drop(fused))
```

Update `build` to branch:
```python
def build(num_classes, two_stream=False, **kw):
    if two_stream:
        return TwoStreamClassifier(
            num_classes, emb=kw.get("emb", 384), width=kw.get("width", 48),
            dropout=kw.get("dropout", 0.3))
    return SignClassifier(num_classes, **kw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_model_twostream.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_model_twostream.py src/asl/model.py
git commit -m "feat(model): FrameEncoder in_ch + late-fusion TwoStreamClassifier

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task C4: Two-stream loading in `ClipDataset`

**Files:**
- Modify: `src/asl/dataset.py` (`ClipDataset`)
- Test: `tests/test_augment.py` (add a dataset case using a tiny synthetic npz)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_augment.py`:
```python
def test_clipdataset_two_stream_returns_pair(tmp_path):
    import numpy as np, json, os
    from asl.dataset import ClipDataset
    n = 6
    X = (np.random.default_rng(0).random((n, 16, 112, 112, 3)) * 255).astype(np.uint8)
    Xflow = np.random.default_rng(1).random((n, 16, 112, 112, 2)).astype(np.float16)
    y = np.arange(n) % 3
    part = np.array(["P22"] * n)              # a known train signer
    split = np.array(["train"] * n)
    npz = os.path.join(tmp_path, "c.npz")
    np.savez(npz, X=X, Xflow=Xflow, y=y, participant=part, split=split,
             files=part)
    norm = os.path.join(tmp_path, "n.json")
    json.dump({"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}, open(norm, "w"))
    fnorm = os.path.join(tmp_path, "n_flow.json")
    json.dump({"mean": [0.0, 0.0], "std": [1.0, 1.0]}, open(fnorm, "w"))
    ds = ClipDataset(npz, "train", norm, train=False, two_stream=True,
                     flow_norm_path=fnorm, signer_splits=None)
    (rgb, flow), label = ds[0]
    assert rgb.shape == (16, 3, 112, 112)
    assert flow.shape == (16, 2, 112, 112)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_augment.py::test_clipdataset_two_stream_returns_pair -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'two_stream'`.

- [ ] **Step 3: Implement two-stream support in `ClipDataset`**

Replace `ClipDataset.__init__` and `__getitem__` with:
```python
    def __init__(self, npz_path, split, norm_path="artifacts/manifest/norm.json",
                 train=False, size=112,
                 signer_splits="artifacts/manifest/signer_splits.json",
                 two_stream=False, flow_norm_path=None):
        data = np.load(npz_path, allow_pickle=True)
        if signer_splits and os.path.exists(signer_splits):
            policy = json.load(open(signer_splits))
            part = data["participant"].astype(str)
            mask = np.array([policy.get(p) == split for p in part])
        else:
            mask = data["split"].astype(str) == split
        self.X = data["X"][mask]
        self.y = data["y"][mask].astype(np.int64)
        self.two_stream = two_stream
        if two_stream:
            self.Xflow = data["Xflow"][mask]
            fm = json.load(open(flow_norm_path))
            self.fmean = np.array(fm["mean"], dtype=np.float32)
            self.fstd = np.array(fm["std"], dtype=np.float32)
        self.train = train
        self.size = size
        self.mean, self.std = _load_norm(norm_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        clip = self.X[i].astype(np.float32) / 255.0
        if self.train:
            clip = augment_clip(clip, self.size)
        clip = (clip - self.mean) / self.std
        rgb = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        if not self.two_stream:
            return rgb, int(self.y[i])
        flow = self.Xflow[i].astype(np.float32)
        flow = (flow - self.fmean) / self.fstd
        flow = torch.from_numpy(flow).permute(0, 3, 1, 2).float()
        return (rgb, flow), int(self.y[i])
```
(Note: flow is NOT augmented in this version — geometric aug would have to be applied jointly and vector-rotation-correctly; deferred. RGB aug still applies to the appearance stream.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_augment.py -v`
Expected: PASS (all augment + dataset tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_augment.py src/asl/dataset.py
git commit -m "feat(data): two-stream (rgb+flow) loading in ClipDataset

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task C5: Two-stream training path in `train.py` + config

**Files:**
- Modify: `src/asl/train.py`
- Create: `configs/finetune_twostream.yaml`

- [ ] **Step 1: Create the config**

```yaml
# configs/finetune_twostream.yaml — RGB + optical-flow late fusion, ROI clips.
cache: artifacts/cache/clips_roi_flow.npz
norm: artifacts/manifest/norm_roi.json
flow_norm: artifacts/manifest/norm_roi_flow.json
manifest: artifacts/manifest/manifest.json

two_stream: true
pretrained_encoder: artifacts/checkpoints/pretrain/encoder.pt   # RGB encoder

head: attn
emb: 384
width: 48
dropout: 0.3
tf_layers: 1
tf_heads: 4

epochs: 60
batch_size: 32
lr: 0.0015
weight_decay: 0.05
warmup_epochs: 3
label_smoothing: 0.1
early_stop_patience: 12
seed: 1337

out_dir: artifacts/checkpoints/finetune_twostream
```

- [ ] **Step 2: Wire two-stream into `train.py`**

After `cfg = yaml.safe_load(...)`, define a flag near the top of `main`:
```python
    two_stream = bool(cfg.get("two_stream", False))
```
Replace the three dataset constructions (lines 66-68) with:
```python
    fnorm = cfg.get("flow_norm")
    tr = ClipDataset(cfg["cache"], "train", cfg["norm"], train=True,
                     two_stream=two_stream, flow_norm_path=fnorm)
    va = ClipDataset(cfg["cache"], "val", cfg["norm"], train=False,
                     two_stream=two_stream, flow_norm_path=fnorm)
    te = ClipDataset(cfg["cache"], "test", cfg["norm"], train=False,
                     two_stream=two_stream, flow_norm_path=fnorm)
```
Replace the model build (lines 75-77) with:
```python
    if two_stream:
        model = build(n_classes, two_stream=True, emb=cfg["emb"],
                      dropout=cfg["dropout"], width=cfg.get("width", 48)).to(dev)
    else:
        model = build(n_classes, emb=cfg["emb"], head=cfg["head"],
                      tf_layers=cfg["tf_layers"], tf_heads=cfg["tf_heads"],
                      dropout=cfg["dropout"], width=cfg.get("width", 48)).to(dev)
```
Replace the pretrained-encoder load block (lines 78-85) with a version that warm-starts BOTH streams (flow gets the body, fresh 2ch stem):
```python
    pre = cfg.get("pretrained_encoder")
    if pre and os.path.exists(pre):
        ck = torch.load(pre, map_location=dev)
        model.encoder.load_state_dict(ck["encoder"])
        if two_stream:
            # warm-start flow encoder body from RGB; keep its fresh 2ch stem
            flow_sd = {k: v for k, v in ck["encoder"].items()
                       if not k.startswith("stem.")}
            model.encoder_flow.load_state_dict(flow_sd, strict=False)
            print("warm-started flow encoder body from RGB encoder")
        elif "pool" in ck:
            model.pool.load_state_dict(ck["pool"])
        print(f"loaded pretrained encoder from {pre} "
              f"(pretrain val {ck.get('val_acc')}, {ck.get('pretrain_classes')} classes)")
```
The train/eval loops call `model(x.to(dev))`. Make the forward two-stream-aware. Replace the `evaluate` function body's `logits = model(x.to(dev))` and the train-loop `loss = crit(model(x.to(dev)), ...)` by first defining a helper near the top of `main` (after `dev = device()`):
```python
    def fwd(model, x):
        if two_stream:
            rgb, flow = x
            return model(rgb.to(dev), flow.to(dev))
        return model(x.to(dev))
```
Then in `evaluate`, change its signature to `evaluate(model, loader, dev, fwd)` and use `logits = fwd(model, x)`; update its two call sites to pass `fwd`. In the train loop change to `loss = crit(fwd(model, x), y.to(dev))`.

- [ ] **Step 3: Smoke-test the wiring on CPU (1 epoch, tiny)**

Run:
```bash
PYTHONPATH=src python -u -m asl.train --config configs/finetune_twostream.yaml --epochs 1
```
Expected: prints `device=…`, `classes=75 train=… val=… test=…`, `warm-started flow encoder body from RGB encoder`, one `ep 00 …` line, and a final `BEST … TEST …` line — no shape errors. (Accuracy after 1 epoch is meaningless; this only verifies the path runs.)

- [ ] **Step 4: Commit**

```bash
git add src/asl/train.py configs/finetune_twostream.yaml
git commit -m "feat(train): two-stream fine-tune path with dual-encoder warm start

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task C6: Full two-stream fine-tune + A/B vs RGB-only

**Files:** none (run + analyze). Run on the local M4 (single job — do NOT run a second MPS training concurrently).

- [ ] **Step 1: Train**

Run:
```bash
PYTHONPATH=src python -u -m asl.train --config configs/finetune_twostream.yaml
```
Expected: a `BEST val_top1=… TEST top1=…` line in `artifacts/checkpoints/finetune_twostream/`.

- [ ] **Step 2: Analyze, focusing on the targeted confusions**

Run:
```bash
PYTHONPATH=src python -c "
import numpy as np; from asl import analysis, dataset, model
import torch, json
# two-stream needs a custom predict; reuse analysis for RGB-only baseline,
# and compute two-stream test acc + confusions inline.
ck = torch.load('artifacts/checkpoints/finetune_twostream/best.pt', map_location='cpu')
labels = json.load(open('artifacts/manifest/manifest.json'))['labels']
m = model.build(len(labels), two_stream=True, emb=ck['cfg']['emb'],
                dropout=ck['cfg']['dropout'], width=ck['cfg'].get('width',48))
m.load_state_dict(ck['state_dict']); m.eval()
ds = dataset.ClipDataset('artifacts/cache/clips_roi_flow.npz','test',
        'artifacts/manifest/norm_roi.json', train=False, two_stream=True,
        flow_norm_path='artifacts/manifest/norm_roi_flow.json')
import numpy as np
P=[]; T=[]
with torch.no_grad():
    for i in range(len(ds)):
        (rgb,flow),y=ds[i]
        logit=m(rgb[None],flow[None])
        P.append(int(logit.argmax(1))); T.append(y)
P=np.array(P); T=np.array(T)
print('two-stream TEST top1', round((P==T).mean(),3))
# dead signs
dead=sum(1 for c in range(len(labels)) if (T==c).any() and (P[T==c]==c).mean()==0)
print('dead signs', dead, '/', len(labels))
"
```
Expected: prints two-stream test top-1 + dead-sign count. **Decision gate:** compare to the RGB-only Phase-B number; the win must show up as either higher top-1 OR fewer of the targeted movement confusions (`SORRY→HUNGRY`, `WORK→COOK`). Record both.

### Task C7: Two-input ONNX export + flow in meta.json

**Files:**
- Modify: `src/asl/export.py`

- [ ] **Step 1: Add a two-stream export branch**

In `export`, after loading `cfg`, branch the model build and the ONNX export. Replace the model build (lines 30-33) with:
```python
    two_stream = bool(cfg.get("two_stream", False))
    if two_stream:
        model = build(len(labels), two_stream=True, emb=cfg["emb"],
                      dropout=cfg["dropout"], width=cfg.get("width", 48))
    else:
        model = build(len(labels), emb=cfg["emb"], head=cfg["head"],
                      tf_layers=cfg["tf_layers"], tf_heads=cfg["tf_heads"],
                      dropout=cfg["dropout"], width=cfg.get("width", 48))
    model.load_state_dict(ck["state_dict"])
    model.eval()
```
Replace the dummy/export/verify block (lines 37-52) with:
```python
    F = manifest["input"]["frames"]; S = manifest["input"]["size"]
    if two_stream:
        d_rgb = torch.randn(1, F, 3, S, S); d_flow = torch.randn(1, F, 2, S, S)
        torch.onnx.export(
            model, (d_rgb, d_flow), onnx_path,
            input_names=["clip", "flow"], output_names=["logits"],
            dynamic_axes={"clip": {0: "batch"}, "flow": {0: "batch"},
                          "logits": {0: "batch"}}, opset_version=17)
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(d_rgb, d_flow).numpy()
        got = sess.run(None, {"clip": d_rgb.numpy(), "flow": d_flow.numpy()})[0]
    else:
        dummy = torch.randn(1, F, 3, S, S)
        torch.onnx.export(
            model, dummy, onnx_path, input_names=["clip"], output_names=["logits"],
            dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17)
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(dummy).numpy()
        got = sess.run(None, {"clip": dummy.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    assert max_diff < 1e-3, f"ONNX/torch mismatch {max_diff}"
    print(f"ONNX verified: max|torch-onnx| = {max_diff:.2e}")
```
In the `meta` dict, add a flow input spec when two-stream. After building `meta`, insert:
```python
    if two_stream:
        fn = json.load(open(norm_path.replace(".json", "_flow.json")))
        meta["input"]["flow"] = {"channels": 2, "algo": "farneback",
                                 "mean": fn["mean"], "std": fn["std"],
                                 "note": "app must compute Farneback (dx,dy) on ROI frames"}
```

- [ ] **Step 2: Export and verify size**

Run:
```bash
PYTHONPATH=src python -m asl.export \
  --ckpt artifacts/checkpoints/finetune_twostream/best.pt \
  --out-dir "$CLAUDE_JOB_DIR/models_test" --version v0.2-twostream \
  --norm artifacts/manifest/norm_roi.json --quantize
```
Expected: `ONNX verified: …`, then `Exported … (<10 MB) + meta.json`. Confirm no 10MB warning and that `meta.json` has `input.flow`.

- [ ] **Step 3: Commit**

```bash
git add src/asl/export.py
git commit -m "feat(export): two-input ONNX + flow spec in meta.json

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task C8: Update the model story + workstream doc

**Files:**
- Modify: `scripts/build_story_notebook.py` (add a two-stream iteration section if it won), `MODEL_WORKSTREAM.md` (verify flow contract row exists).

- [ ] **Step 1: Verify the flow contract row is present**

Run: `grep -n "Motion-ROI crop\|Farneback\|two input" MODEL_WORKSTREAM.md`
Expected: the Motion-ROI crop row exists (added previously). If the two-stream ships, add an `input.flow` note referencing the meta.json field.

- [ ] **Step 2: If two-stream won, add a story section**

Add an iteration section to `scripts/build_story_notebook.py` mirroring the existing pattern (markdown + a code cell that computes the two-stream test acc and the targeted-confusion deltas), then rebuild:
```bash
PYTHONPATH=src python scripts/build_story_notebook.py
jupyter nbconvert --to html notebooks/model_story.ipynb
```
Expected: notebook re-executes clean; HTML written.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_story_notebook.py notebooks/model_story.ipynb notebooks/model_story.html MODEL_WORKSTREAM.md
git commit -m "docs: add optical-flow two-stream iteration to model story

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Run the whole test suite: `PYTHONPATH=src python -m pytest tests/ -v` — all green.
- [ ] Confirm the shipped checkpoint's ONNX export is <10MB (Task C7).
- [ ] Confirm held-out discipline intact: no val/test signer in any train/pretrain manifest (`build_pretrain_manifest` already excludes them; spot-check `pretrain_manifest_1500.json` participants against `signer_splits.json`).
- [ ] Record final numbers (Phase A/B/C test top-1 + dead signs) for the model story.

## Notes carried from the spec

- **From-scratch**: Farneback + ROI are classical (no weights); both encoders train from random init on ASL Citizen only.
- **Browser fallback**: if the app can't compute Farneback flow, ship the single-stream RGB checkpoint from Phase B (it exists independently). Documented in `MODEL_WORKSTREAM.md`.
- **Deferred**: temporal jitter (conflicts with precomputed flow), flow augmentation (needs vector-correct geometric transform), Transformer temporal head.
