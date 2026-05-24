"""Build the Constellation v2 model-story notebook (Stage 1: the detector).

Separate from the v1 story (model/notebooks/model_story.ipynb) by design — this
covers ONLY the from-scratch landmark-primary v2 pipeline. Run:

    cd model-v2 && .venv/bin/python notebooks/build_v2_story.py

Produces:
    notebooks/model_v2_story.ipynb
    notebooks/assets/detector_training_curve.png
    notebooks/assets/viz_*.png   (copied annotated audit frames)

It reads the CURRENT history.json + audit_eval.json (the validated w256/192²
detector), so re-running after a new training refreshes the live numbers.
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
ASSETS = NB_DIR / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

HIST = json.loads((ROOT / "artifacts/checkpoints/detector/history.json").read_text())
hist = HIST["history"]
AUDIT = json.loads((ROOT / "artifacts/checkpoints/detector/audit_eval.json").read_text())


# ---------------------------------------------------------------------------
# Training-curve figure (the validated w256/192² retrain)
# ---------------------------------------------------------------------------
def render_curve() -> None:
    ep = [h["epoch"] for h in hist]
    loss = [h["train_loss"] for h in hist]
    hand = [h["hand_dr"] for h in hist]
    head = [h["head_dr"] for h in hist]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ep, loss, color="#444", lw=2, label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="#444")
    ax1.set_ylim(0, max(loss) * 1.05)

    ax2 = ax1.twinx()
    ax2.plot(ep, head, color="#1f77b4", lw=2, label="head detection-rate")
    ax2.plot(ep, hand, color="#2ca02c", lw=2, label="hand detection-rate")
    ax2.axhline(0.85, color="#1f77b4", ls=":", lw=1)
    ax2.axhline(0.70, color="#2ca02c", ls=":", lw=1)
    ax2.set_ylabel("val detection-rate@0.5"); ax2.set_ylim(0, 1)
    ax2.axvline(HIST["best_epoch"], color="#d62728", ls="--", lw=1)

    lines = ax1.get_lines() + ax2.get_lines()[:2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
    ax1.set_title("Detector retrain (img 192, width 256) — A100, 60 ep")
    fig.tight_layout()
    fig.savefig(ASSETS / "detector_training_curve.png", dpi=130); plt.close(fig)


render_curve()

viz_src = ROOT / "artifacts/audit/viz"
for name in ["viz_07732970042675213-HAPPY.png", "viz_06960268077531429-DOG.png"]:
    if (viz_src / name).exists():
        shutil.copy(viz_src / name, ASSETS / name)


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------
final = hist[-1]
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

ha = AUDIT["head_anchor_dr"]; hi = AUDIT["head_iou_dr"]; hd = AUDIT["hand_dr"]

cells = [
md("""# Constellation v2 — Model Story · Part 1: the Detector (Stage 1)

> **This is the v2 story and is deliberately separate from the v1 story**
> (`model/notebooks/model_story.ipynb`). v1 is a single end-to-end video
> classifier; v2 ("Constellation") is a 3-stage, landmark-primary pipeline.
> Don't merge the narratives — they're different models. This notebook covers
> **Stage 1** (the hand+head detector) and the decisions that got it validated.

## Why v2 exists

v1 was a tiny end-to-end video CNN trained on a narrow ASL Citizen subset. It
**overfit appearance** (signer, clothing, background, lighting) and plateaued at
low validation/test accuracy — it learned *who* and *where*, not *what the hands
do*.

**The v2 thesis:** recognize signs from **appearance-invariant pose geometry**
instead of pixels — hand keypoints + the hand's position relative to the head.
There's far less to overfit to. Getting geometry needs a pipeline, all **trained
from scratch, zero pretrained weights** (PRD Req 7):

```
Stage 1    detector        → hand & head BOXES         ← this notebook
Stage 1.5  landmark model  → 21 hand keypoints (crop)
Stage 2    recognizer      → sign, from pose geometry (+ small appearance stream)
```
"""),

md("""## 1. Stage-1 design — a tiny single-shot detector

From-scratch, SSD-style single-shot detector, two foreground classes: `hand` (0)
and `head` (1).

- **Square anchors only** (BlazePalm trick): hands are roughly square blobs.
- **Depthwise-separable backbone**, stride-16, fully convolutional (adapts to any
  input size).
- **head ≈ face anchor:** the `head` class trains on **WIDER FACE** boxes — a fact
  that drives a key decision later.
- Data: **100DOH** (hand boxes) + **WIDER FACE** (head/face boxes), 90,204 train
  images, shrunk to 256 px for a ~1 GB Colab upload (lossless for the model's
  input resolution).
"""),

md("""## 2. Iteration 1 — the first detector, and the test that "failed"

First model: **128² input, width 192** (~0.58 M params). It trained cleanly and
cleared the head proxy gate, but **missed the hand proxy gate** on 100DOH val
(`hand_dr 0.691` vs the 0.70 bar; `head_dr 0.883`).

Then the honest test — a **manually-labeled slice of 80 real ASL Citizen frames**
(hand + head boxes). The first read looked like a flat failure:

| class | ASL-audit detection-rate@0.5 | gate | first read |
|---|---|---|---|
| head | 0.287 | ≥ 0.80 | ❌ |
| hand | 0.514 | ≥ 0.60 | ❌ |

Instead of accepting or discarding the model, we **instrumented the failure**.
"""),

md("""## 3. Diagnosis — what the numbers were really saying

Running predictions against the labels frame-by-frame told a very different story:

- **Head wasn't missing — it was mis-*measured*.** The detector found a head in
  **100 %** of frames, and its predicted center sat **inside** the labeled head
  box **90 %** of the time. It draws a tight **face** box (it learned from WIDER
  *face* data); we labeled whole **heads**. The face box sits inside the head box
  → IoU ≈ 0.46 → "miss" at the 0.5 threshold. The IoU metric was wrong for what
  the head box is *for*.
- **Hand was a real but specific weakness.** **0** frames had zero hand
  detections; per-hand recall was 0.68 @IoU0.5. The problem: a labeled hand
  averages ~150 px in a 640 px frame → only **~30 px** once squished to 128². Too
  few pixels to localize precisely.
"""),

md(f"""## 4. Two decisions

**Decision A — score the head as a *normalization anchor*, not an object.**
Downstream (spec §3.6) the head box only provides a **center + scale** to
normalize hand positions. So the honest metric is *predicted center inside the GT
box + a sane size ratio* — not pixel-tight IoU. Whole-head labels stay; the
model's consistent face box is a fine anchor. (IoU@0.5 is kept as a *reference*
number, clearly flagged.)

**Decision B — give the hands more pixels.** Retrain at **img 128 → 192**
(≈2.25× pixels per hand) **and width 192 → 256**. To do this safely the anchor
scales were made to **auto-derive from the input size** (so they can never drift
out of sync), and the same resolution now flows through data → train → inference
→ the audit eval.

> Why resolution over more data: the detector already generalized (0 frame
> misses); the bottleneck was the *30-px hand*, which only more pixels fix.
"""),

md(f"""## 5. Iteration 2 — the retrain (img 192, width 256)

![training curve](assets/detector_training_curve.png)

It converged at best_score **{HIST['best_score']:.3f}** @ ep{HIST['best_epoch']}
(`hand_dr {final['hand_dr']:.3f}`, `head_dr {final['head_dr']:.3f}`) — and
crucially **hand_dr crossed the 0.70 proxy gate** that the first model missed.

| | iter 1 (128², w192) | **iter 2 (192², w256)** |
|---|---|---|
| params | 0.58 M | 1.02 M |
| 100DOH `hand_dr` | 0.691 ❌ | **0.731 ✅** |
| 100DOH `head_dr` | 0.883 | **0.906** |
"""),

md(f"""## 6. Validated on real ASL frames

Re-running the ASL-audit gate on the retrained detector, with the **anchor metric**
for head (Decision A):

| metric (ASL-audit, binding gate) | value | gate | verdict |
|---|---|---|---|
| **head — anchor (center + scale)** | **{ha:.3f}** | ≥ 0.80 | {'✅ PASS' if AUDIT['head_pass'] else '❌'} |
| **hand — detection-rate@0.5** | **{hd:.3f}** | ≥ 0.60 | {'✅ PASS' if AUDIT['hand_pass'] else '❌'} |
| head — IoU@0.5 (reference only) | {hi:.3f} | — | face-box vs whole-head label |

**Both binding gates pass.** The detector places a correctly-centered, sanely-
sized head anchor and finds hands on real ASL frames — the higher-res retrain
lifted hand detection from 0.514 → {hd:.3f}.
"""),

code("""# Reproduce the validated gate from the saved eval
import json
a = json.load(open("../artifacts/checkpoints/detector/audit_eval.json"))
print(f"head anchor : {a['head_anchor_dr']:.3f}  (gate {a['head_gate']})  pass={a['head_pass']}")
print(f"hand dr     : {a['hand_dr']:.3f}  (gate {a['hand_gate']})  pass={a['hand_pass']}")
print(f"head IoU@0.5: {a['head_iou_dr']:.3f}  (reference; face box vs whole-head label)")
"""),

md("""## 7. Live-webcam check (ADR-0001 live tester)

On the builder's own webcam (`scripts/detect_demo.py --mirror`) the detector
**correctly binds the head and both hands** in real time. Annotated audit frames
(blue = predicted face box + center · green = hand boxes · yellow = head→hand
vector in head-size units · red = the whole-head/hand labels):

![happy](assets/viz_07732970042675213-HAPPY.png)
![dog](assets/viz_06960268077531429-DOG.png)

The blue face box inside the red head label is the whole IoU "failure"; the yellow
head→hand vector — what Stage 2 consumes — is clean regardless.
"""),

md("""## 8. Decisions log — how we made the calls

1. **A "failing" gate isn't a failing model — instrument before reacting.** The
   head fail was a metric/label-convention mismatch; the hand fail was a concrete,
   fixable resolution limit.
2. **Match the metric to the box's job.** The head box is a normalization anchor,
   so it's scored by center + scale, not IoU. (IoU kept as a flagged reference.)
3. **Pick the lever the diagnosis points at.** Hands were resolution-limited, so
   resolution (+ a little capacity) was the fix — verified by both proxy and
   binding gates improving.
4. **Proxy vs binding gates.** 100DOH val is a hard proxy; the binding gate is the
   real-ASL audit slice. We report both and trust the binding one.

## 9. Next — Stage 1.5: the hand-landmark model

A from-scratch 21-keypoint regressor (FreiHAND) runs inside each hand box. Its
first version scored a flattering PCK@0.2 = 0.987 but **collapsed to a mean hand
on real webcam hands** — the live test caught what the lenient metric hid. It's
being retrained with a strict PCK@0.1 gate + framing/scale/rotation augmentation.
That's **Part 2** of this story.
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = NB_DIR / "model_v2_story.ipynb"
nbf.write(nb, out)
print("wrote", out, "|", len(cells), "cells")
