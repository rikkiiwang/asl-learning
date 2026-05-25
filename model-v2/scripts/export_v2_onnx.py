"""Export the three trained Constellation (v2) stages to ONNX for the browser.

Separate ONNX files (not a fused graph): the geometry/tracking glue between
stages is dynamic numpy (slot_hands, resolve_head_anchor, normalize_geometry)
that does not fuse cleanly into one exportable graph, so the app runs the three
nets and ports the glue to TypeScript (mirrors v1's ROI-crop parity approach).

Run:  cd model-v2 && .venv/bin/python scripts/export_v2_onnx.py
Writes: <out_dir>/{detector,landmark,recognizer}.onnx  (default ../app/public/models/v2)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aslv2.detect.model import Detector
from aslv2.landmark.model import Landmark
from aslv2.recog.model import RecognizerA
from aslv2.geometry import GEOM_DIM

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "artifacts/checkpoints"
OPSET = 17


def _load_cfg(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("cfg", {}), ck["state_dict"]


def export_detector(out: Path) -> dict:
    cfg, sd = _load_cfg(CK / "detector/best.pt")
    img = cfg.get("img", 192)   # detector runs at its TRAINED resolution (192²)
    m = Detector(
        n_classes=cfg.get("n_classes", 2),
        n_anchors=cfg.get("n_anchors", 3),
        width=cfg.get("width", 256),
    ).eval()
    m.load_state_dict(sd)
    dummy = torch.randn(1, 3, img, img)
    torch.onnx.export(
        m, dummy, str(out / "detector.onnx"),
        input_names=["frames"], output_names=["cls_logits", "box_pred"],
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
        dynamic_axes={"frames": {0: "n"}, "cls_logits": {0: "n"}, "box_pred": {0: "n"}},
    )
    return {"width": cfg.get("width", 256), "n_anchors": cfg.get("n_anchors", 3),
            "n_classes": cfg.get("n_classes", 2), "img": img,
            "stride": 16, "score_thr": 0.3, "iou_thr": 0.45,
            "norm": cfg.get("norm")}


def export_landmark(out: Path) -> dict:
    cfg, sd = _load_cfg(CK / "landmark/best.pt")
    m = Landmark(width=cfg.get("width", 32)).eval()
    m.load_state_dict(sd)
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        m, dummy, str(out / "landmark.onnx"),
        input_names=["crops"], output_names=["kps"],
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
        dynamic_axes={"crops": {0: "m"}, "kps": {0: "m"}},
    )
    return {"width": cfg.get("width", 32), "crop_size": 64,
            "frame_scale": 2.0, "norm": cfg.get("norm")}


def export_recognizer(out: Path) -> dict:
    cfg, sd = _load_cfg(CK / "recog_a/best.pt")
    m = RecognizerA(
        n_classes=cfg.get("n_classes", 75),
        emb=cfg.get("emb", 192),
        head=cfg.get("head", "transformer"),
        dropout=cfg.get("dropout", 0.3),
        use_velocity=cfg.get("use_velocity", True),
        geom_dim=GEOM_DIM,
    ).eval()
    m.load_state_dict(sd)
    dummy = torch.randn(1, 16, GEOM_DIM)
    torch.onnx.export(
        m, dummy, str(out / "recognizer.onnx"),
        input_names=["geom"], output_names=["logits"],
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
        dynamic_axes={"geom": {0: "b", 1: "f"}, "logits": {0: "b"}},
    )
    return {"emb": cfg.get("emb", 192), "head": cfg.get("head", "transformer"),
            "use_velocity": cfg.get("use_velocity", True), "geom_dim": GEOM_DIM,
            "n_classes": cfg.get("n_classes", 75), "frames": 16}


def verify(out: Path):
    """ORT round-trip each model against the torch reference."""
    import onnxruntime as ort

    report = {}
    # detector
    cfg, sd = _load_cfg(CK / "detector/best.pt")
    img = cfg.get("img", 192)
    m = Detector(n_classes=cfg.get("n_classes", 2), n_anchors=cfg.get("n_anchors", 3),
                 width=cfg.get("width", 256)).eval()
    m.load_state_dict(sd)
    x = torch.randn(2, 3, img, img)
    with torch.no_grad():
        rc, rb = (t.numpy() for t in m(x))
    s = ort.InferenceSession(str(out / "detector.onnx"), providers=["CPUExecutionProvider"])
    gc, gb = s.run(None, {"frames": x.numpy()})
    report["detector"] = bool(np.allclose(rc, gc, atol=1e-3) and np.allclose(rb, gb, atol=1e-3))

    # landmark
    cfg, sd = _load_cfg(CK / "landmark/best.pt")
    m = Landmark(width=cfg.get("width", 32)).eval(); m.load_state_dict(sd)
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        rk = m(x).numpy()
    s = ort.InferenceSession(str(out / "landmark.onnx"), providers=["CPUExecutionProvider"])
    gk = s.run(None, {"crops": x.numpy()})[0]
    report["landmark"] = bool(np.allclose(rk, gk, atol=1e-3))

    # recognizer
    cfg, sd = _load_cfg(CK / "recog_a/best.pt")
    m = RecognizerA(n_classes=cfg.get("n_classes", 75), emb=cfg.get("emb", 192),
                    head=cfg.get("head", "transformer"), dropout=cfg.get("dropout", 0.3),
                    use_velocity=cfg.get("use_velocity", True), geom_dim=GEOM_DIM).eval()
    m.load_state_dict(sd)
    x = torch.randn(1, 16, GEOM_DIM)
    with torch.no_grad():
        rl = m(x).numpy()
    s = ort.InferenceSession(str(out / "recognizer.onnx"), providers=["CPUExecutionProvider"])
    gl = s.run(None, {"geom": x.numpy()})[0]
    report["recognizer"] = bool(np.allclose(rl, gl, atol=1e-3))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "app/public/models/v2"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = {"contract": "3 separate ONNX models; geometry glue ported to TS"}
    meta["detector"] = export_detector(out)
    meta["landmark"] = export_landmark(out)
    meta["recognizer"] = export_recognizer(out)

    sizes = {p.name: p.stat().st_size for p in out.glob("*.onnx")}
    meta["bytes"] = sizes
    rep = verify(out)
    meta["ort_roundtrip"] = rep

    (out / "v2_meta.json").write_text(json.dumps(meta, indent=2))
    print("exported to", out)
    for k, v in sizes.items():
        print(f"  {k}: {v/1024:.0f} KB")
    print("ORT round-trip:", rep)


if __name__ == "__main__":
    main()
