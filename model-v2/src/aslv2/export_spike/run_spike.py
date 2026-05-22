"""Prove-first export spike (spec §6). Exports the combined graph and records the
(a)-vs-(b) decision. Run as a script to write artifacts/export_decision.md."""
from pathlib import Path

import torch

from .combined import CombinedConstellation

OPSET = 17


def _seeded_model():
    torch.manual_seed(0)
    return CombinedConstellation().eval()


def export_combined(out_path) -> Path:
    out_path = Path(out_path)
    model = _seeded_model()
    dummy = torch.randn(16, 3, 128, 128)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["frames"], output_names=["logits"],
        opset_version=OPSET, do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter; dynamo=True requires onnxscript
    )
    return out_path


def main():
    art = Path(__file__).resolve().parents[4] / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    decision = art / "export_decision.md"
    try:
        export_combined(art / "combined_spike.onnx")
        import numpy as np
        import onnxruntime as ort
        m = _seeded_model()
        x = torch.randn(16, 3, 128, 128)
        with torch.no_grad():
            ref = m(x).numpy()
        sess = ort.InferenceSession(str(art / "combined_spike.onnx"),
                                    providers=["CPUExecutionProvider"])
        got = sess.run(None, {"frames": x.numpy()})[0]
        ok = bool(np.allclose(ref, got, atol=1e-3))
        decision.write_text(
            f"# Export decision\n\nCombined-graph (b) Python-ORT round-trip: "
            f"{'PASS' if ok else 'MISMATCH (atol=1e-3)'}.\n\n"
            "Next: run scripts/ort_web_probe.mjs to confirm ONNX Runtime Web "
            "(WASM) supports the ops. If web probe passes -> choose (b). "
            "If export raised or web probe fails -> choose (a) separate files.\n"
        )
        print("Python-ORT round-trip:", "PASS" if ok else "MISMATCH")
    except Exception as e:           # noqa: BLE001 - we WANT to record failure
        decision.write_text(
            f"# Export decision\n\nCombined-graph (b) export FAILED: {e!r}.\n\n"
            "Decision: choose (a) separate ONNX files; coordinate §B with the app "
            "agent before training (spec §6).\n"
        )
        print("Combined export failed -> decision (a). See", decision)
        raise


if __name__ == "__main__":
    main()
