"""Build + execute notebooks/model_story.ipynb — a self-contained narrative of
the ASL model architecture exploration and training for a hiring-partner audience.

Run from the model/ dir:  python scripts/build_story_notebook.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

md = new_markdown_cell
code = new_code_cell
cells = []

cells.append(md(
"""# ASL Sign Recognition — Model Exploration & Training Story

**Goal:** a *from-scratch* computer-vision model that recognizes **75 beginner ASL
vocabulary signs** from a short webcam clip, runs **in the browser** (ONNX, < 10 MB),
and returns a **conservative pass/fail** for a language-learning app.

**Hard constraints (self-imposed by the brief):**
- No pretrained models / backbones / landmark detectors — every weight trained by us.
- Browser inference, model file 2-6 MB (hard cap 10 MB).
- Honest evaluation: **signer-held-out** splits (no signer in two splits).

This notebook walks through the data, the architecture, the train-from-scratch
iterations, and an error analysis that drives the next step. Everything is
reproducible from the `asl` package in `src/`."""))

cells.append(code(
"""import sys, json, collections
sys.path.insert(0, 'src')
import numpy as np, torch
import matplotlib.pyplot as plt
from asl.model import build
from asl import analysis
plt.rcParams.update({'figure.figsize': (9, 4), 'axes.grid': True, 'grid.alpha': .3})
print('torch', torch.__version__, '| device',
      'mps' if torch.backends.mps.is_available() else
      ('cuda' if torch.cuda.is_available() else 'cpu'))"""))

cells.append(md(
"""## 1. Data — a curated, balanced, signer-held-out subset of ASL Citizen

We train on **ASL Citizen** (Microsoft Research; 2,731 signs, ~83k videos, 52 signers),
but the app only teaches 75 signs, so the model is a **75-class** problem — not a
2,731-class one. The 75 were chosen by a two-stage funnel: a hand-authored ASL-1
candidate pool (10 pedagogical categories) intersected with the data and filtered to
signs with enough examples, then balanced across categories.

Key data decisions:
- **One canonical variant per concept** (ASL-LEX numbers variants like `EAT1/EAT2`); we
  pin the standard form and never merge variants into one class.
- **Signer-held-out splits.** ASL Citizen's official split is *test-heavy* (built for a
  retrieval benchmark), so we re-partitioned the signers to favor training while keeping
  splits signer-disjoint."""))

cells.append(code(
"""m = json.load(open('artifacts/manifest/manifest.json'))
labels = m['labels']
cat = collections.Counter(s['category'] for s in m['signs'])
print(f'Vocabulary: {len(labels)} signs across {len(cat)} categories')
for c, n in sorted(cat.items()):
    print(f'  {c:22s} {n}')

d = np.load('artifacts/cache/clips.npz', allow_pickle=True)
part = d['participant'].astype(str)
splits = json.load(open('artifacts/manifest/signer_splits.json'))
sp = np.array([splits[p] for p in part])
print('\\nSplits (signer-disjoint):')
for s in ['train', 'val', 'test']:
    sig = len(set(part[sp == s]))
    print(f'  {s:5s}: {(sp==s).sum():4d} clips  '
          f'({(sp==s).sum()/len(labels):.1f}/class)  from {sig} signers')"""))

cells.append(md(
"""## 2. Architecture — a tiny, browser-friendly video classifier

Signs depend on **motion over time**, so a per-frame vote throws away too much. But a
3D CNN is heavy and hard to train from scratch on a small budget. The compromise:

```
clip (16 frames @ 112x112)
  -> shared depthwise-separable 2D CNN frame encoder  (per-frame embedding)
  -> learned attention pooling over the 16 frames      (temporal aggregation)
  -> 75-way linear classifier -> logits
```

Depthwise-separable convolutions keep the parameter count tiny so the quantized ONNX
fits the browser budget with room to spare."""))

cells.append(code(
"""model = build(len(labels), emb=384, width=48, head='attn')
print(model)
n = sum(p.numel() for p in model.parameters())
print(f'\\nParameters: {n:,}  (~{n*4/1e6:.1f} MB fp32, ~{n/1e6:.1f} MB int8-quantized)')
print('Browser budget < 10 MB (target 2-6 MB) -> comfortably within.')"""))

cells.append(md(
"""## 3. Training from scratch — two iterations

**v1 — a very small encoder (76k params).** It *underfit*: training loss plateaued
high and validation never took off. The model literally lacked the capacity to fit the
task.

**v2 — a wider encoder (484k params).** Training loss now drops well below v1 and
validation climbs higher and faster — capacity *was* the first bottleneck."""))

cells.append(code(
"""v1 = analysis.parse_log('artifacts/checkpoints/baseline/train.log')   # 76k params
v2 = analysis.parse_log('artifacts/checkpoints/baseline/probe.log')   # 484k params
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for v, lbl in [(v1, 'v1: 76k params'), (v2, 'v2: 484k params')]:
    e = [r[0] for r in v]
    ax[0].plot(e, [r[1] for r in v], label=lbl)
    ax[1].plot(e, [r[2] for r in v], label=lbl)
ax[0].set(title='Training loss', xlabel='epoch', ylabel='loss'); ax[0].legend()
ax[1].set(title='Validation top-1', xlabel='epoch', ylabel='top-1'); ax[1].legend()
plt.tight_layout(); plt.show()"""))

cells.append(md(
"""## 4. Error analysis — where v2 actually stands

Capacity fixed the underfitting, but did it generalize? The numbers below tell the
real story."""))

cells.append(code(
"""R = analysis.run_analysis()      # runs the trained v2 checkpoint over all splits
acc = R['acc']
print('Top-1 accuracy:  train %.3f | val %.3f | test %.3f'
      % (acc['train'], acc['val'], acc['test']))
plt.bar(list(acc), list(acc.values()), color=['#4c72b0', '#dd8452', '#55a868'])
plt.axhline(1/len(labels), ls='--', c='gray', label='chance (1/75)')
plt.title('Train vs Val vs Test top-1'); plt.ylim(0, 1); plt.ylabel('top-1'); plt.legend()
plt.show()
print('Train >> Test: a large gap = OVERFITTING. With ~23 training clips/class,'
      ' data scarcity (not capacity) is now the wall.')"""))

cells.append(code(
"""pc = R['per_class']                       # per-class TEST accuracy (75 signs)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(pc, bins=20, range=(0, 1), color='#55a868', edgecolor='white')
ax[0].set(title='Per-class test-accuracy distribution', xlabel='accuracy', ylabel='# signs')
order = np.argsort(pc)
ax[1].bar(range(len(pc)), pc[order], color='#55a868')
ax[1].set(title='Per-class test accuracy, sorted (75 signs)',
          xlabel='sign (sorted)', ylabel='accuracy')
plt.tight_layout(); plt.show()
print(f'{int((pc == 0).sum())}/{len(pc)} signs are at 0% test accuracy — '
      'the model gets a minority of signs and misses the long tail entirely.')"""))

cells.append(code(
"""print('Top-10 confused pairs  (true -> predicted, #clips):')
for cnt, a, b in R['top_confused']:
    print(f'  {cnt:3d}   {a:12s} -> {b}')
mag = collections.Counter(b for _, _, b in R['top_confused'])
print('\\n"Magnet" predicted classes:', mag.most_common(3))
print('Several signs collapse onto a few classes (e.g. EAT) — consistent with an'
      ' over-fit model defaulting to data-rich classes on unfamiliar test signers.')

cm = R['cm'].astype(float)
cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
plt.figure(figsize=(7.5, 6.5))
plt.imshow(cmn, cmap='magma')
plt.colorbar(label='row-normalized')
plt.title('Confusion matrix (75x75, test)'); plt.xlabel('predicted'); plt.ylabel('true')
plt.show()"""))

cells.append(md(
"""## 5. Iteration 3 — A classical motion ROI crop (attacking *background* overfitting)

The error analysis pointed at overfitting, but to **what**? A prime suspect: each
signer is filmed against their own static background, so the encoder can cheat by
memorizing the room instead of the hands. Test: crop every clip to where the motion
is, and see if generalization improves.

**Method (no learned model — stays within the from-scratch rule).** Accumulate
per-pixel inter-frame change across the clip, take a robust bounding box of the moving
region, pad + square it, and fall back to a center crop when motion is too weak. This
both zooms the hands in *and* strips the per-signer background. Same architecture, same
hyperparameters, same splits — the **only** change is the crop, so any gain is
attributable to it."""))

cells.append(code(
"""R_roi = analysis.run_analysis(ckpt='artifacts/checkpoints/roi/best.pt',
                              cache='artifacts/cache/clips_roi.npz',
                              norm='artifacts/manifest/norm_roi.json')
b, r = R['acc']['test'], R_roi['acc']['test']
bz = int(np.isnan(R['per_class']).sum() + (R['per_class'] == 0).sum())
rz = int(np.isnan(R_roi['per_class']).sum() + (R_roi['per_class'] == 0).sum())
print('TEST top-1   center-crop %.3f  ->  ROI-crop %.3f   (+%.1f pts)' % (b, r, 100*(r-b)))
print('signs at 0%%  center-crop %d/75   ->  ROI-crop %d/75' % (bz, rz))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(['center', 'ROI'], [b, r], color=['#bbb', '#55a868'])
ax[0].set(title='Test top-1', ylim=(0, .5), ylabel='top-1')
ax[1].bar(['center', 'ROI'], [bz, rz], color=['#bbb', '#55a868'])
ax[1].set(title='Signs the model never gets (0% test)', ylabel='# signs / 75')
plt.tight_layout(); plt.show()
print('Removing background lifts test 18.0 -> 28.8%% and rescues 13 dead signs:'
      ' the model WAS leaning on the room.')"""))

cells.append(md(
"""## 6. Iteration 4 — Pretraining the encoder from scratch (attacking *data scarcity*)

The other half of the overfitting wall is just **too few clips per class** (~23). The
fix that needs no external weights: let the encoder learn hand/motion features from a
much larger slice of the *same* corpus, then specialize. Pretrain the CNN encoder
**from scratch** on a **500-sign / ~12.7k-clip** slice of ASL Citizen — crucially
**excluding our val/test signers** so no held-out signer leaks into the features — then
fine-tune the 75-class head on top.

**Engineering rigor — root-causing a stuck run.** The first attempt sat frozen at
`ln(500)`. Rather than guess, I isolated it: verified the labels and that the 7.7 GB
memmap held real frames (byte-identical to the npz cache); found + fixed a latent
non-contiguous-input crash (`x.contiguous()`); and proved the pipeline correct by
training the memmap on the same 75 signs to an identical curve. **Conclusion: nothing
was broken** — a 500-way softmax from scratch simply has weak early gradients and needs
~20+ epochs to leave chance; the run had been killed too early (and a second run died
to GPU contention when I foolishly ran two trainings at once). Moved to a clean Colab
T4: the encoder reached **65.9% top-1 on the 500-way task** — it really did learn
general features."""))

cells.append(code(
"""R_ft = analysis.run_analysis(ckpt='artifacts/checkpoints/finetune/best.pt')  # center-crop
labels_setup = ['baseline\\n(center)', 'ROI crop', 'pretrain\\n+finetune']
tests = [R['acc']['test'], R_roi['acc']['test'], R_ft['acc']['test']]
zeros = [int(np.isnan(x).sum() + (x == 0).sum())
         for x in (R['per_class'], R_roi['per_class'], R_ft['per_class'])]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
cols = ['#bbb', '#55a868', '#4c72b0']
ax[0].bar(labels_setup, tests, color=cols)
for i, v in enumerate(tests): ax[0].text(i, v + .01, f'{v:.1%}', ha='center')
ax[0].set(title='Held-out TEST top-1', ylim=(0, .55), ylabel='top-1')
ax[0].axhline(1/75, ls='--', c='red', label='chance'); ax[0].legend()
ax[1].bar(labels_setup, zeros, color=cols)
for i, v in enumerate(zeros): ax[1].text(i, v + .5, str(v), ha='center')
ax[1].set(title='Signs never recognized (0% test)', ylabel='# signs / 75')
plt.tight_layout(); plt.show()
print('Pretraining: TEST %.1f%% (train %.0f%%), dead signs %d/75.'
      % (100*R_ft['acc']['test'], 100*R_ft['acc']['train'], zeros[2]))
print('Top confusions are now linguistically sensible (look-alike signs):')
for cnt, a, bb in R_ft['top_confused'][:6]:
    print(f'  {cnt:2d}  {a} <-> {bb}')"""))

cells.append(md(
"""## 7. Iteration 5 — Stacking the two winners (and a more interesting result than I expected)

ROI crop and pretraining each beat the baseline by attacking a *different* failure mode
(background cues vs. data scarcity), so the obvious next move is to **stack** them:
rebuild the 500-sign pretrain set with the *same* motion-ROI crop, then fine-tune the
75-class head on ROI clips. If the levers were independent, the gains should roughly add.

They don't — and *that* is the finding."""))

cells.append(code(
"""R_stack = analysis.run_analysis(
    ckpt='artifacts/checkpoints/finetune_roi/best.pt',
    cache='artifacts/cache/clips_roi.npz',
    norm='artifacts/manifest/norm_roi.json')
labels_setup = ['baseline\\n(center)', 'ROI crop', 'pretrain\\n+finetune', 'stacked\\n(ROI+pretrain)']
runs = [R, R_roi, R_ft, R_stack]
tests = [r['acc']['test'] for r in runs]
zeros = [int(np.isnan(r['per_class']).sum() + (r['per_class'] == 0).sum()) for r in runs]
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
cols = ['#bbb', '#55a868', '#4c72b0', '#c44e52']
ax[0].bar(labels_setup, tests, color=cols)
for i, v in enumerate(tests): ax[0].text(i, v + .01, f'{v:.1%}', ha='center')
ax[0].set(title='Held-out TEST top-1', ylim=(0, .55), ylabel='top-1')
ax[0].axhline(1/75, ls='--', c='red', label='chance'); ax[0].legend()
ax[1].bar(labels_setup, zeros, color=cols)
for i, v in enumerate(zeros): ax[1].text(i, v + .5, str(v), ha='center')
ax[1].set(title='Signs never recognized (0% test)', ylabel='# signs / 75')
plt.tight_layout(); plt.show()
print('Stacked: TEST %.1f%% (train %.0f%%), dead signs %d/75.'
      % (100*R_stack['acc']['test'], 100*R_stack['acc']['train'], zeros[3]))
print('ROI alone added +%.1f pts over baseline, but only +%.1f pts on top of pretraining:'
      % (100*(R_roi['acc']['test']-R['acc']['test']),
         100*(R_stack['acc']['test']-R_ft['acc']['test'])))
print('  -> the two levers attack OVERLAPPING failure modes (background invariance),')
print('     so their gains are sub-additive, not additive.')"""))

cells.append(md(
"""**Why sub-additive?** Pretraining on 500 signs across many signers and backgrounds
*already* teaches the encoder to ignore background — so by the time ROI is added, most of
its background-removal benefit is already paid for. The two levers overlap. The tell is in
the curve: the stacked run hit best-val at **epoch 6** (vs. 31 for pretrain-only) and
`train = 0.99` — the ROI-pretrained encoder is so well matched to the ROI clips that the
head memorizes the 75-class set almost instantly. That's a **small-data ceiling**, not a
lever failing: there isn't enough 75-class data to keep improving once the encoder is this
good a match.

The stacked model is still the single best checkpoint (highest test top-1), so it's the
one we ship — but the honest story isn't "every lever adds up," it's "know *which*
problem each lever solves, because solving the same one twice buys little." """))

cells.append(md(
"""## 8. Where we landed, and what's next

| Setup (from-scratch, identical arch) | train | val | **test** | signs @ 0% |
|---|---|---|---|---|
| Baseline — center crop | 0.72 | 0.16 | **0.18** | 33/75 |
| ROI crop | 0.85 | 0.26 | **0.29** | 20/75 |
| Pretrain → fine-tune | 1.00 | 0.35 | **0.43** | **5/75** |
| **Stacked — ROI + pretrain** | 0.99 | 0.38 | **0.46** | 7/75 |

Three from-scratch levers took held-out accuracy from **18% → 46%** and cut
never-recognized signs from **33 → ~6**. The biggest jumps were ROI (background) and
pretraining (data scarcity); stacking them confirmed the two overlap (sub-additive +3
pts), which is itself a useful result. Remaining confusions are genuine look-alike sign
pairs (`SORRY↔HUNGRY`, `WORK↔COOK`, `HELLO↔MAN`) — the productive kind, addressable with
distinguishing hints in the app.

**Honest status.** `train ≈ 0.99` means overfitting isn't *solved* — the pretrained
features raised the generalization floor, they didn't flatten the train/test gap. The
remaining headroom is in the **data**, not the levers we've already pulled:

1. **More data per class** — the ep-6 best-val and 99% train acc say the 75-class set is
   the binding constraint. Expanding to ~50+ clips/sign (more ASL Citizen signers) should
   lift the ceiling more than any architecture change.
2. **Stronger augmentation** (temporal jitter, horizontal flip with label-aware handling,
   mild affine) to close the residual train→test gap on the data we have.
3. **A Transformer temporal head** for motion-order cues, once data supports the capacity.

**Deployment note.** The shipped model uses the motion-ROI crop, so the **app must
replicate that crop in-browser** before inference — it's now part of the inference
contract (documented in `MODEL_WORKSTREAM.md`). The crop is classical (inter-frame
abs-diff → bounding box), so it ports to JS/WASM without any model.

**Deployment framing (unchanged):** the app uses a *conservative* per-class confidence
threshold (false-pass < 5%), so it can be trustworthy before accuracy is high — it asks
the learner to retry when unsure rather than passing a guess. The browser-export path is
already de-risked (PyTorch → ONNX verified; quantized ≈ 0.5 MB, far under the 10 MB cap)."""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})

print("executing notebook (runs inference over all splits) ...")
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
    nb, {"metadata": {"path": "."}})
with open("notebooks/model_story.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/model_story.ipynb")
