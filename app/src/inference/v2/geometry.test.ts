import { describe, it, expect } from 'vitest';
import { buildClipGeometry, GEOM_DIM, type ClipKps, type FrameDet } from './geometry';
import type { Box } from './boxes';
import golden from './__fixtures__/geometry_golden.json';

describe('v2 geometry builder (parity vs Python build_clip_features)', () => {
  it('GEOM_DIM is 93', () => {
    expect(GEOM_DIM).toBe(93);
  });

  it('matches Python geom (16×93) within float tolerance', () => {
    const dets: FrameDet[] = golden.dets.map((d) => ({
      hands: d.hands as Box[],
      head: (d.head as number[] | null) as Box | null,
    }));
    const kps = golden.kps as ClipKps;

    const geom = buildClipGeometry(dets, kps, golden.frameW, golden.frameH);
    expect(geom.length).toBe(golden.expectGeom.length);
    for (let i = 0; i < geom.length; i++) {
      expect(geom[i]).toBeCloseTo(golden.expectGeom[i], 4);
    }
  });
});
