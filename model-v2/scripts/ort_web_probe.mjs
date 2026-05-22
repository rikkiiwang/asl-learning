// Loads the exported combined graph in onnxruntime-web (WASM) and runs once.
// Usage: node scripts/ort_web_probe.mjs artifacts/combined_spike.onnx
import * as ort from "onnxruntime-web";
import { readFileSync } from "node:fs";

const path = process.argv[2] ?? "artifacts/combined_spike.onnx";
const bytes = readFileSync(path);
try {
  const sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const frames = new ort.Tensor("float32", new Float32Array(16 * 3 * 128 * 128), [16, 3, 128, 128]);
  const out = await sess.run({ frames });
  const logits = out[Object.keys(out)[0]];
  console.log("ORT-Web OK; logits dims =", logits.dims);
  process.exit(logits.dims.join(",") === "1,75" ? 0 : 2);
} catch (e) {
  console.error("ORT-Web FAILED (op unsupported in WASM?):", String(e));
  process.exit(1);
}
