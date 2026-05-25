"""Fill asl_audit_slice.json with hand+head boxes from a makesense.ai export.

In makesense.ai: load all 80 frames, define exactly two labels `hand` and `head`,
draw boxes, then Actions → Export Annotations and pick either format:

  COCO JSON (single file):
    .venv/bin/python scripts/convert_makesense_to_audit.py --coco ~/Downloads/labels.json

  VOC XML (a .zip of per-image .xml, or an unzipped directory):
    .venv/bin/python scripts/convert_makesense_to_audit.py --voc ~/Downloads/labels.zip

By default writes back to artifacts/audit/asl_audit_slice.json (use --out to redirect).
"""
import argparse
import json
import zipfile
from pathlib import Path

from aslv2.audit_label import coco_to_audit, voc_to_audit


def _read_voc_texts(path: str) -> list[str]:
    """Return the XML document strings from a VOC export (.zip or directory)."""
    p = Path(path)
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            return [z.read(n).decode("utf-8") for n in z.namelist() if n.endswith(".xml")]
    if p.is_dir():
        return [f.read_text() for f in sorted(p.glob("*.xml"))]
    raise SystemExit(f"--voc must be a .zip or a directory of .xml files: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="makesense export → labeled audit slice.")
    ap.add_argument("--coco", help="path to the COCO JSON export")
    ap.add_argument("--voc", help="VOC export: a .zip or a directory of .xml files")
    ap.add_argument("--slice", default="artifacts/audit/asl_audit_slice.json")
    ap.add_argument("--out", default=None, help="default: overwrite --slice in place")
    args = ap.parse_args()

    if not args.coco and not args.voc:
        raise SystemExit("provide --coco <file.json> or --voc <file.zip|dir>")

    stub = json.loads(Path(args.slice).read_text())
    if args.voc:
        labeled, missing = voc_to_audit(_read_voc_texts(args.voc), stub)
    else:
        labeled, missing = coco_to_audit(json.loads(Path(args.coco).read_text()), stub)

    out = args.out or args.slice
    Path(out).write_text(json.dumps(labeled, indent=2))

    n_hands = sum(len(f["hands"]) for f in labeled)
    n_heads = sum(1 for f in labeled if f["head"] is not None)
    print(f"labeled {len(labeled)} frames: {n_hands} hand boxes, {n_heads} heads")
    if missing:
        shown = ", ".join(missing[:10]) + ("…" if len(missing) > 10 else "")
        print(f"WARNING: {len(missing)} frame(s) have NO boxes: {shown}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
