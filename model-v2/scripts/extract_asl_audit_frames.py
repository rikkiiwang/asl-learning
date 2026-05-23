"""Extract ~80 diverse ASL Citizen frames for the audit ground-truth slice.

Selects ~80 videos spanning >=8 distinct participants from the manifest,
decodes the middle frame from each video, writes it to
  model-v2/artifacts/audit/frames/<id>.png
and writes a stub JSON list to
  model-v2/artifacts/audit/asl_audit_slice.json

The stub JSON matches the schema expected by aslv2.audit.load_audit_slice
(hands/head/keypoints are empty — to be hand-labeled later).

Run from the repo root with:
  model-v2/.venv/bin/python model-v2/scripts/extract_asl_audit_frames.py
"""
import json
import sys
from pathlib import Path
import random

import cv2

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]   # /…/ASL Learning
MODEL_ROOT  = REPO_ROOT / "model-v2"
VIDEO_DIR   = REPO_ROOT / "model" / "data" / "ASL_Citizen" / "videos"
MANIFEST    = REPO_ROOT / "model" / "artifacts" / "manifest" / "manifest.json"
OUT_DIR     = MODEL_ROOT / "artifacts" / "audit" / "frames"
STUB_JSON   = MODEL_ROOT / "artifacts" / "audit" / "asl_audit_slice.json"

TARGET_FRAMES      = 80
MIN_PARTICIPANTS   = 8


def load_clips():
    """Return list of (file_stem, participant, gloss) from manifest."""
    data = json.loads(MANIFEST.read_text())
    clips = []
    for sign in data["signs"]:
        gloss = sign["gloss"]
        for clip in sign["clips"]:
            clips.append({
                "file":        clip["file"],
                "stem":        clip["file"].replace(".mp4", ""),
                "participant": clip["participant"],
                "gloss":       gloss,
                "split":       clip["split"],
            })
    return clips


def select_diverse(clips, target=TARGET_FRAMES, min_participants=MIN_PARTICIPANTS):
    """Select ~target clips with diverse participants and glosses."""
    # Build a pool of clips that exist on disk
    available = [c for c in clips if (VIDEO_DIR / c["file"]).exists()]
    print(f"[select] {len(available)} / {len(clips)} manifest clips found on disk")

    # Group by participant
    by_participant: dict[str, list] = {}
    for c in available:
        by_participant.setdefault(c["participant"], []).append(c)

    participants = sorted(by_participant.keys())
    print(f"[select] {len(participants)} unique participants available")

    # Round-robin across participants to maximise diversity
    # Then within each participant, pick different glosses
    rng = random.Random(42)  # reproducible
    selected = []
    seen_ids = set()
    rounds = 0
    while len(selected) < target:
        made_progress = False
        for p in participants:
            if len(selected) >= target:
                break
            pool = [c for c in by_participant[p] if c["stem"] not in seen_ids]
            if not pool:
                continue
            # prefer a gloss not yet seen for this participant
            pick = rng.choice(pool)
            selected.append(pick)
            seen_ids.add(pick["stem"])
            made_progress = True
        rounds += 1
        if not made_progress:
            break

    n_participants = len({c["participant"] for c in selected})
    print(f"[select] selected {len(selected)} clips across {n_participants} participants")
    if n_participants < min_participants:
        print(f"WARNING: only {n_participants} participants covered (need ≥{min_participants})")
    return selected


def extract_middle_frame(video_path: Path) -> cv2.typing.MatLike | None:
    """Return the middle frame of the video as a BGR numpy array, or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fall back to frame 0
        mid = 0
    else:
        mid = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips = load_clips()
    selected = select_diverse(clips, target=TARGET_FRAMES, min_participants=MIN_PARTICIPANTS)

    stubs = []
    n_written = 0
    n_failed  = 0

    for clip in selected:
        video_path = VIDEO_DIR / clip["file"]
        stem       = clip["stem"]
        out_png    = OUT_DIR / f"{stem}.png"

        frame = extract_middle_frame(video_path)
        if frame is None:
            print(f"  [WARN] could not read {clip['file']}, skipping")
            n_failed += 1
            continue

        cv2.imwrite(str(out_png), frame)
        n_written += 1

        # Relative path from model-v2/ root (matches how callers will reference it)
        rel_path = str(out_png.relative_to(MODEL_ROOT))
        stubs.append({
            "image":     rel_path,
            "hands":     [],
            "head":      None,
            "keypoints": None,
        })

        if n_written % 10 == 0:
            print(f"  [{n_written}/{len(selected)}] wrote {out_png.name}")

    # Write stub JSON
    STUB_JSON.parent.mkdir(parents=True, exist_ok=True)
    STUB_JSON.write_text(json.dumps(stubs, indent=2))

    participants_covered = {c["participant"] for c in selected
                            if (OUT_DIR / f"{c['stem']}.png").exists()}
    print(f"\n[done] {n_written} PNGs written, {n_failed} skipped")
    print(f"       {len(participants_covered)} distinct participants covered")
    print(f"       stub JSON → {STUB_JSON}")


if __name__ == "__main__":
    main()
