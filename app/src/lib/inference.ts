export interface Ranked {
  index: number;
  prob: number;
}

/** Numerically stable softmax (subtracts the max before exponentiating). */
export function softmax(logits: ArrayLike<number>): number[] {
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];

  const exps = new Array<number>(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) exps[i] /= sum;
  return exps;
}

/** Top-`k` probabilities with their class indices, sorted descending. */
export function topK(probs: ArrayLike<number>, k: number): Ranked[] {
  const ranked: Ranked[] = [];
  for (let i = 0; i < probs.length; i++) ranked.push({ index: i, prob: probs[i] });
  ranked.sort((a, b) => b.prob - a.prob);
  return ranked.slice(0, Math.min(k, ranked.length));
}
