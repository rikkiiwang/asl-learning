import type { InferenceSession } from 'onnxruntime-web';
import type { ModelOption } from '../lib/models';

/** A recognizer turns an NFCHW input tensor into raw logits over the 75-sign vocabulary. */
export interface Recognizer {
  recognize(input: Float32Array): Promise<Float32Array>;
}

/**
 * Deterministic dev stub so the capture → inference → result loop runs
 * end-to-end before the real ONNX model is published (M7). Output varies by
 * `seedSalt` (derived from the model id) so the comparison switcher shows
 * different "models". DEV ONLY.
 */
export class StubRecognizer implements Recognizer {
  private readonly numClasses: number;
  private readonly seedSalt: number;

  constructor(numClasses: number, seedSalt = 0) {
    this.numClasses = numClasses;
    this.seedSalt = seedSalt;
  }

  async recognize(input: Float32Array): Promise<Float32Array> {
    let seed = this.seedSalt + 1;
    for (let i = 0; i < input.length; i += 257) seed = (seed + Math.abs(input[i]) * 31.7) % 9973;
    const logits = new Float32Array(this.numClasses);
    for (let c = 0; c < this.numClasses; c++) logits[c] = Math.sin(seed * 0.137 + c * 0.61);
    logits[Math.floor(seed) % this.numClasses] += 3; // simulate a confident-ish top class
    return logits;
  }
}

/**
 * ONNX Runtime Web recognizer. `onnxruntime-web` is imported dynamically so the
 * wasm runtime only loads when a real model is actually used (and so unit tests
 * that touch the stub don't pull it in).
 */
export class OnnxRecognizer implements Recognizer {
  private session: InferenceSession | null = null;
  private readonly modelUrl: string;
  private readonly opts: { inputName?: string; shape?: number[] };

  constructor(modelUrl: string, opts: { inputName?: string; shape?: number[] } = {}) {
    this.modelUrl = modelUrl;
    this.opts = opts;
  }

  private async ensureSession(): Promise<InferenceSession> {
    if (!this.session) {
      const ort = await import('onnxruntime-web');
      // Single-threaded WASM: avoids the SharedArrayBuffer / cross-origin-isolation
      // (COOP/COEP) requirement of the threaded build, which otherwise stalls in a
      // plain dev server. The model is tiny (~2 MB), so single-thread is plenty.
      ort.env.wasm.numThreads = 1;
      // Serve ORT's own .wasm/.mjs from the version-matched CDN. Without this, the
      // bundler-resolved loader can't locate the wasm in a dev server ("both async
      // and sync fetching of the wasm failed"). Keep the version in sync with the
      // onnxruntime-web dependency in package.json.
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ['wasm'],
      });
    }
    return this.session;
  }

  async recognize(input: Float32Array): Promise<Float32Array> {
    const session = await this.ensureSession();
    const ort = await import('onnxruntime-web');
    const shape = this.opts.shape ?? [1, 16, 3, 112, 112];
    const inputName = this.opts.inputName ?? session.inputNames[0];
    const out = await session.run({ [inputName]: new ort.Tensor('float32', input, shape) });
    return out[session.outputNames[0]].data as Float32Array;
  }
}

// Per-model artifact URLs. Empty until the model side publishes to /models;
// until then every model resolves to a (per-id) stub so the loop still runs.
const MODEL_URLS: Record<string, string> = {
  'own-v1': '/models/asl-v1.onnx',
  // 'own-v2': '/models/asl-v0.2.onnx',
};

export function createRecognizer(model: ModelOption, numClasses: number): Recognizer {
  const url = MODEL_URLS[model.id];
  if (url) return new OnnxRecognizer(url);
  const salt = [...model.id].reduce((a, c) => a + c.charCodeAt(0), 0);
  return new StubRecognizer(numClasses, salt);
}
