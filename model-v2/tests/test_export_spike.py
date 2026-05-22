import torch
from aslv2.export_spike.combined import CombinedConstellation


def test_combined_forward_outputs_logits():
    model = CombinedConstellation().eval()
    frames = torch.randn(16, 3, 128, 128)
    with torch.no_grad():
        logits = model(frames)
    assert logits.shape == (1, 75)


def _seeded_model():
    torch.manual_seed(0)
    return CombinedConstellation().eval()


def test_combined_graph_exports_and_roundtrips(tmp_path):
    import numpy as np
    import onnxruntime as ort
    from aslv2.export_spike.run_spike import export_combined, _seeded_model as spike_seeded

    onnx_path = export_combined(tmp_path / "combined.onnx")   # may raise -> (b) infeasible

    torch.manual_seed(42)
    frames = torch.randn(16, 3, 128, 128)
    model = spike_seeded()
    with torch.no_grad():
        ref = model(frames).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: frames.numpy()})[0]
    np.testing.assert_allclose(ref, got, atol=1e-3)
