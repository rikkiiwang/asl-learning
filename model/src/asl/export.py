"""Phase 5 — export a trained checkpoint to ONNX + meta.json for ONNX Runtime Web.

Verifies the ONNX output matches PyTorch within tolerance (the Phase-1 export
smoke test runs this on a tiny/untrained model first to de-risk the browser path).
Optionally applies dynamic quantization and reports file size.

meta.json is the app contract: input spec, label index map, normalization,
per-class thresholds (filled at calibration), and hint metadata.

Usage:
    python -m asl.export --ckpt artifacts/checkpoints/baseline/best.pt \
        --out-dir ../models --version v0.1 --quantize
"""
import argparse
import json
import os

import numpy as np
import torch

from asl.model import build


def export(ckpt_path, out_dir, version, manifest_path, norm_path, quantize):
    os.makedirs(out_dir, exist_ok=True)
    manifest = json.load(open(manifest_path))
    labels = manifest["labels"]
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["cfg"]
    model = build(len(labels), emb=cfg["emb"], head=cfg["head"],
                  tf_layers=cfg["tf_layers"], tf_heads=cfg["tf_heads"],
                  dropout=cfg["dropout"], width=cfg.get("width", 48))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    onnx_path = os.path.join(out_dir, f"asl-{version}.onnx")
    dummy = torch.randn(1, manifest["input"]["frames"], 3,
                        manifest["input"]["size"], manifest["input"]["size"])
    torch.onnx.export(
        model, dummy, onnx_path, input_names=["clip"], output_names=["logits"],
        dynamic_axes={"clip": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17)

    # verify ORT == torch
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"clip": dummy.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    assert max_diff < 1e-3, f"ONNX/torch mismatch {max_diff}"
    print(f"ONNX verified: max|torch-onnx| = {max_diff:.2e}")

    final_path = onnx_path
    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        qpath = os.path.join(out_dir, f"asl-{version}.quant.onnx")
        quantize_dynamic(onnx_path, qpath, weight_type=QuantType.QUInt8)
        final_path = qpath

    norm = json.load(open(norm_path))
    meta = {
        "version": version,
        "input": {**manifest["input"], "mean": norm["mean"], "std": norm["std"]},
        "num_classes": len(labels),
        "labels": labels,
        "thresholds": {lab: {"t": 0.0, "margin": 0.0} for lab in labels},
        "hints": {s["label"]: {"category": s["category"], "gloss": s["gloss"]}
                  for s in manifest["signs"]},
        "notes": "thresholds are placeholders until Phase 4 calibration",
    }
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)

    size = os.path.getsize(final_path) / 1e6
    print(f"Exported {final_path} ({size:.2f} MB) + meta.json to {out_dir}/")
    if size > 10:
        print(f"[WARN] {size:.1f} MB exceeds the 10 MB browser cap.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="../models")
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--manifest", default="artifacts/manifest/manifest.json")
    ap.add_argument("--norm", default="artifacts/manifest/norm.json")
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()
    export(args.ckpt, args.out_dir, args.version, args.manifest, args.norm,
           args.quantize)


if __name__ == "__main__":
    main()
