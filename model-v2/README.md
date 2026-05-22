# Constellation — ASL Recognizer v2 (`model-v2/`)

**Separate workspace from `model/`.** The original single-stream recognizer
(`model/`) is owned by another agent — do not edit it from here. This directory
holds the **v2** architecture, codenamed **Constellation**.

## What Constellation is (one paragraph)

A three-stage, from-scratch, **landmark-primary** ASL sign recognizer (MediaPipe-
method, no pretrained weights). **Stage 1:** our own tiny single-shot detector
finds **hand** and **head** boxes per frame. **Stage 1.5:** our own hand-landmark
model turns each hand crop into **21 keypoints**. **Stage 2:** recognition runs
*primarily* on the appearance-invariant **pose geometry** (keypoints + head
anchor, head-normalized), with a small hand-appearance CNN as a secondary cue —
fused per frame and read over time by a temporal head into 75-way sign logits.
The name: keypoints are a constellation of points, the head is the fixed anchor
star, and a sign is recognized from how that constellation moves. Recognizing
from geometry instead of pixels is the cure for v1's appearance-overfitting.

## Why v2 exists

v1 (`model/`) plateaued at **16% val / 18% test** on 75 classes: a single CNN on
the full 112×112 frame spends its capacity on background/clothing/face and never
learns the hands (Risk 3 in `../vision-model-plan.md`). Constellation redirects
capacity onto the hands and hands the "location/movement" parameters to the model
as explicit geometry.

## Design source of truth

`../docs/superpowers/specs/2026-05-22-constellation-v2-model-design.md`

Implementation plan and code land here after the plan is written.
