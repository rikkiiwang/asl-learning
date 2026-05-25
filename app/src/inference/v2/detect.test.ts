import { describe, it, expect } from 'vitest';
import { anchorsFor, decodeDetections } from './detect';
import golden from './__fixtures__/decode_golden.json';

describe('v2 detector decode (parity vs Python)', () => {
  it('builds 432 anchors at 192² (12×12×3)', () => {
    expect(anchorsFor(192).length).toBe(432);
  });

  it('matches Python decode_post: same hands + head boxes', () => {
    const out = decodeDetections(golden.cls, golden.box, {
      img: golden.img,
      origW: golden.origW,
      origH: golden.origH,
      scoreThr: golden.scoreThr,
      iouThr: golden.iouThr,
    });

    expect(out.hands.length).toBe(golden.expect.hands.length);
    out.hands.forEach((b, i) => {
      b.forEach((v, j) => expect(v).toBeCloseTo(golden.expect.hands[i][j], 1));
    });

    if (golden.expect.head === null) {
      expect(out.head).toBeNull();
    } else {
      expect(out.head).not.toBeNull();
      out.head!.forEach((v, j) => expect(v).toBeCloseTo((golden.expect.head as number[])[j], 1));
    }
  });
});
