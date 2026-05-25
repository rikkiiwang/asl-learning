# Constellation v2 — Plan 6: COCO-WholeBody → better landmark (+ detector) → better recognizer

**Goal:** Fix the *pose-distribution mismatch* capping the front-end. FreiHAND only
has hands grasping objects (curled fingers); ASL is full of flat/open/pointing
handshapes. **COCO-WholeBody** has 21-keypoint hand labels **and hand boxes** on
in-the-wild images (people gesturing) — the missing pose variety. We add it to
both the landmark and (optionally) detector training, re-cache, and re-measure the
recognizer. **Fully from-scratch** — COCO-WholeBody is labeled data, not a model.

**Why this, not "ASL fine-tune":** we have no ASL keypoint labels, and the
from-scratch mandate rules out MediaPipe pseudo-labels. COCO-WholeBody is the clean
way to get ASL-like pose coverage.

---

## The bottlenecks (measured)
- Landmark: trained on FreiHAND grasps → regresses to a mean hand on ASL open palms.
- Detector: finds a hand in only ~0.58 of ASL frames (0.35 on WLASL) — co-bottleneck.
- Both feed geometry; recognizer is at test 47.9% (top-3 69%), ~2.7× v1.

## Data: COCO-WholeBody
- Annotations: `coco_wholebody_{train,val}_v1.0.json` (github.com/jin-s13/COCO-WholeBody).
- Images: COCO **val2017** (~1 GB, ~5k images) to start; **train2017** (~19 GB) to scale.
- Per person annotation: `lefthand_kpts`/`righthand_kpts` (21×3 x,y,vis),
  `lefthand_box`/`righthand_box` (xywh), `lefthand_valid`/`righthand_valid`.
- Filter: valid hand, box min-side ≥ ~30 px, ≥ ~18/21 keypoints visible.

## Phases (each a measured step)

### Phase 1 — adapter (TDD, local)
`scripts/cocowholebody_adapt.py` + `src/aslv2/landmark/cocowholebody.py`:
parse the wholebody JSON → emit (a) **landmark** records `{image, box(xyxy),
keypoints(21,2)}` (same schema as FreiHAND), and (b) **detector** hand-box records
`{image, boxes, labels:[0]}`. Filter to valid/large-enough hands. Unit-test the
pure parse on a synthetic annotation.

### Phase 2 — landmark retrain (Colab) ← do first
Merge FreiHAND + COCO-WholeBody landmark manifests → shrink → package → retrain on
Colab (existing `train_landmark_colab.ipynb`, same strict PCK@0.1 gate). Compare to
current landmark (pck@0.1 0.84) **and** eyeball on the audit frames (the real test:
do keypoints track ASL open hands?).

### Phase 3 — detector retrain (Colab) ← optional, after Phase 2
Add COCO-WholeBody hand boxes to the detector's hand class (keep WIDER for head,
100DOH for hands). Retrain at img=192/width=256. **Re-verify the ASL-audit gates**
(head-anchor ≥0.80, hand ≥0.60) — must not regress. Goal: lift ASL hand-present.

### Phase 4 — re-cache + recognizer (local + train)
Re-run the cache (ASL Citizen + WLASL) with the improved front-end, retrain the
recognizer (3 seeds, same config), and compare test top-1/top-3 to the current
47.9% / 69%. **Gate:** keep the improved front-end only if the recognizer improves.

## Honest expectations
- Better keypoints/boxes → cleaner geometry → bounded recognizer gain (maybe a few pts).
- Some missed hands are rest frames (unfixable). COCO hands are noisy (filter hard).
- Each phase is measured; we keep only what demonstrably helps.

## Reuses
`aslv2.landmark.{data,train,freihand}`, `aslv2.detect.*`, `aslv2.recog.cache`,
the FreiHAND Colab pipeline, the ASL-audit gate. New code: the COCO-WholeBody adapter only.
