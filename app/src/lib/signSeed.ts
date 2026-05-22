import type { Manifest, SignSeedRow } from './types';

/**
 * Transform the model's manifest into `signs` catalog rows.
 * model_class_index comes from the model's label_idx — never invented here —
 * and must be contiguous 0..N-1 so it lines up with the model's softmax outputs.
 */
export function buildSignSeed(manifest: Manifest): SignSeedRow[] {
  const rows: SignSeedRow[] = manifest.signs
    .map((s) => ({
      model_class_index: s.label_idx,
      label: s.label,
      gloss: s.gloss,
      category: s.category,
    }))
    .sort((a, b) => a.model_class_index - b.model_class_index);

  const seen = new Set<number>();
  rows.forEach((r, i) => {
    if (seen.has(r.model_class_index)) {
      throw new Error(`duplicate model_class_index: ${r.model_class_index}`);
    }
    seen.add(r.model_class_index);
    if (r.model_class_index !== i) {
      throw new Error(
        `model_class_index must be contiguous from 0; expected ${i}, got ${r.model_class_index} (${r.label})`,
      );
    }
  });

  return rows;
}
