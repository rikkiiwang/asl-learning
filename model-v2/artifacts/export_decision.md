# Export decision (spec §6) — 2026-05-22

## Probe results

- Combined-graph (b) Python ORT round-trip: PASS (atol=1e-3, opset 17)
- Combined-graph (b) ONNX Runtime Web (WASM) probe: PASS (logits [1,75])

## Notes

- Export used the legacy TorchScript exporter (`dynamo=False`). The new
  torch.export-based exporter (`dynamo=True`, default in PyTorch 2.9+) was
  not used because `onnxscript` is not installed in this environment.
- `torchvision`'s ONNX registration forces `RoIAlign` `sampling_ratio` to 0
  for any non-zero value. This has no functional impact on the shape-only
  spike; the real Plan 5 export must verify this is acceptable or configure
  the op accordingly.
- ORT-Web ran under Node v24 with `onnxruntime-web` (WASM provider).

## DECISION: (b) combined graph

App contract unchanged — input: `frames` (float32, [16,3,128,128]);
output: `logits` (float32, [1,75]).

Plan 5 exports the real 3-stage models fused into one combined graph.
No §B app-contract change is needed.
