"""Extract the WLASL clips matching our 75-sign vocab + write a flat clip manifest.

WLASL (archive.zip from Kaggle) has WLASL_v0.3.json (gloss -> instances{video_id})
and videos/<video_id>.mp4. We match its glosses to our 75 ASL Citizen signs
(stripping ASL-LEX trailing digits, e.g. CAT1 -> CAT), extract only the matched
videos from the zip, and write a flat manifest the cache step consumes.

These clips are tagged participant="WLASL" so they join TRAIN only (val/test stay
pure ASL Citizen signer-held-out).

    cd model-v2 && python scripts/build_wlasl_subset.py --zip ~/Downloads/archive.zip
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUR_MANIFEST = REPO / "model" / "artifacts" / "manifest" / "manifest.json"
OUT_DIR = REPO / "model-v2" / "data" / "wlasl"
OUT_VIDEOS = OUT_DIR / "videos"
CLIPS_JSON = OUT_DIR / "clips.json"


def _norm(g: str) -> str:
    return re.sub(r"\d+$", "", g.upper())   # strip trailing ASL-LEX digits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True, help="path to WLASL archive.zip")
    args = ap.parse_args()

    OUT_VIDEOS.mkdir(parents=True, exist_ok=True)
    our = json.loads(OUR_MANIFEST.read_text())["signs"]
    # map both exact gloss and normalized gloss -> our label_idx
    label_of = {}
    for s in our:
        g = s["gloss"].upper()
        label_of.setdefault(g, s["label_idx"])
        label_of.setdefault(_norm(g), s["label_idx"])

    with zipfile.ZipFile(args.zip) as z:
        wl = json.loads(z.read("WLASL_v0.3.json").decode())
        members = set(z.namelist())

        clips, extracted, missing = [], 0, 0
        for entry in wl:
            g = entry["gloss"].upper()
            idx = label_of.get(g, label_of.get(_norm(g)))
            if idx is None:
                continue                              # not one of our signs
            for inst in entry["instances"]:
                vid = inst["video_id"]
                member = f"videos/{vid}.mp4"
                if member not in members:
                    missing += 1
                    continue
                dst = OUT_VIDEOS / f"{vid}.mp4"
                if not dst.exists():
                    dst.write_bytes(z.read(member))
                extracted += 1
                clips.append({"file": f"{vid}.mp4", "participant": "WLASL",
                              "label_idx": int(idx), "gloss": entry["gloss"]})

    CLIPS_JSON.write_text(json.dumps(clips, indent=2))
    n_glosses = len({c["gloss"] for c in clips})
    print(f"extracted {extracted} videos ({missing} listed-but-absent) for {n_glosses} glosses")
    print(f"-> {OUT_VIDEOS}/  +  {CLIPS_JSON}")


if __name__ == "__main__":
    main()
