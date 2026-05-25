"""Download the MS-ASL clips matching our 75-sign vocab + write a flat manifest.

MS-ASL ships as YouTube URLs + time ranges (MSASL_{train,val,test}.json), NOT video
files. This script matches each annotation's gloss to our 75 ASL Citizen signs
(stripping ASL-LEX trailing digits, e.g. CAT1 -> CAT, same rule as WLASL), then uses
yt-dlp to fetch ONLY the [start_time, end_time] section of each YouTube video into
data/msasl/videos/, and writes the flat clips.json the cache step consumes.

All MS-ASL splits are merged and tagged participant="MSASL" so they join our TRAIN
only (val/test stay pure ASL Citizen signer-held-out) — same policy as WLASL.

Designed to run on Colab (yt-dlp + ffmpeg present after `pip install yt-dlp`):
    python scripts/build_msasl_subset.py --ann-dir data/msasl/ann --max-per-sign 40

Robust to YouTube link-rot: dead/blocked/region-locked URLs are skipped and counted;
already-downloaded clips are skipped so the run is resumable. clips.json is written
incrementally, so an interrupted run still yields a usable (partial) manifest.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELV2 = Path(__file__).resolve().parents[1]
# manifest lives in the v1 tree locally, but is bundled under model-v2/artifacts on Colab
_MANIFEST_CANDIDATES = [
    REPO / "model" / "artifacts" / "manifest" / "manifest.json",
    MODELV2 / "artifacts" / "manifest" / "manifest.json",
]
OUT_DIR = MODELV2 / "data" / "msasl"
OUT_VIDEOS = OUT_DIR / "videos"
CLIPS_JSON = OUT_DIR / "clips.json"


def _resolve_manifest(arg: str | None) -> Path:
    for p in ([Path(arg)] if arg else []) + _MANIFEST_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(f"manifest.json not found; tried {_MANIFEST_CANDIDATES} — pass --manifest")

_YT_ID = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([0-9A-Za-z_-]{11})")


def _norm(g: str) -> str:
    return re.sub(r"\d+$", "", g.strip().upper())   # strip trailing ASL-LEX digits


def _yt_id(url: str) -> str | None:
    m = _YT_ID.search(url or "")
    return m.group(1) if m else None


def _label_map(manifest: Path) -> dict:
    """gloss (exact + normalized) -> our label_idx."""
    our = json.loads(manifest.read_text())["signs"]
    label_of: dict[str, int] = {}
    for s in our:
        g = s["gloss"].upper()
        label_of.setdefault(g, s["label_idx"])
        label_of.setdefault(_norm(g), s["label_idx"])
    return label_of


def _load_matched(ann_dir: Path, splits, label_of):
    """Yield (label_idx, gloss, url, start, end, signer) for matched annotations."""
    seen = set()
    for split in splits:
        fp = ann_dir / f"MSASL_{split}.json"
        if not fp.exists():
            print(f"[skip missing annotation] {fp}")
            continue
        for a in json.loads(fp.read_text()):
            g = a.get("clean_text", "")
            idx = label_of.get(g.upper(), label_of.get(_norm(g)))
            if idx is None:
                continue                                   # not one of our 75 signs
            yid = _yt_id(a.get("url", ""))
            if yid is None:
                continue
            start, end = float(a["start_time"]), float(a["end_time"])
            # de-dup identical (video, time-range) across splits
            key = (yid, round(start, 2), round(end, 2))
            if key in seen:
                continue
            seen.add(key)
            yield idx, a["clean_text"], a["url"], start, end, a.get("signer_id", -1)


def _download(url, start, end, dst: Path) -> bool:
    """Fetch only [start, end] of the YouTube video to dst.mp4 via yt-dlp. Returns ok."""
    if dst.exists():
        return True
    cmd = [
        "yt-dlp", "-q", "--no-warnings", "--no-playlist", "--ignore-errors",
        "--geo-bypass", "--retries", "3", "--socket-timeout", "20",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio/best",
        "--download-sections", f"*{start:.2f}-{end:.2f}",
        "--force-keyframes-at-cuts",
        "--recode-video", "mp4",
        "-o", str(dst.with_suffix("")) + ".%(ext)s",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    return dst.exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ann-dir", default=str(OUT_DIR / "ann"),
                    help="dir holding MSASL_{train,val,test}.json")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="which MS-ASL annotation splits to pull (all -> our TRAIN)")
    ap.add_argument("--max-per-sign", type=int, default=40,
                    help="cap clips per sign for class balance + bounded download (0=no cap)")
    ap.add_argument("--limit", type=int, default=None, help="global cap (debug)")
    ap.add_argument("--manifest", default=None,
                    help="path to the 75-sign manifest.json (auto-detected if omitted)")
    args = ap.parse_args()

    OUT_VIDEOS.mkdir(parents=True, exist_ok=True)
    label_of = _label_map(_resolve_manifest(args.manifest))
    ann_dir = Path(args.ann_dir)

    cands = list(_load_matched(ann_dir, args.splits, label_of))
    print(f"matched annotations: {len(cands)} (across {len(args.splits)} splits)")

    clips, per_sign, ok, dead = [], {}, 0, 0
    for n, (idx, gloss, url, start, end, signer) in enumerate(cands):
        if args.limit and ok >= args.limit:
            break
        if args.max_per_sign and per_sign.get(idx, 0) >= args.max_per_sign:
            continue
        yid = _yt_id(url)
        fname = f"{idx:02d}_{yid}_{int(start * 1000)}.mp4"
        dst = OUT_VIDEOS / fname
        if _download(url, start, end, dst):
            ok += 1
            per_sign[idx] = per_sign.get(idx, 0) + 1
            clips.append({"file": fname, "participant": "MSASL",
                          "label_idx": int(idx), "gloss": gloss, "signer_id": signer})
            if ok % 25 == 0:
                CLIPS_JSON.write_text(json.dumps(clips, indent=2))   # incremental save
                print(f"  {ok} downloaded ({dead} dead) ... last: {gloss}")
        else:
            dead += 1

    CLIPS_JSON.write_text(json.dumps(clips, indent=2))
    n_glosses = len({c["label_idx"] for c in clips})
    print(f"\ndownloaded {ok} clips ({dead} dead/blocked) for {n_glosses}/75 signs")
    print(f"-> {OUT_VIDEOS}/  +  {CLIPS_JSON}")
    print("Next: cache.py --clips-json data/msasl/clips.json --video-dir data/msasl/videos")


if __name__ == "__main__":
    main()
