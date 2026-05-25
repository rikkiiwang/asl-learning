// xyxy box helpers — direct port of model-v2/src/aslv2/boxes.py.
export type Box = [number, number, number, number];

export function iou(a: Box, b: Box): number {
  const ix0 = Math.max(a[0], b[0]);
  const iy0 = Math.max(a[1], b[1]);
  const ix1 = Math.min(a[2], b[2]);
  const iy1 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  if (inter === 0) return 0;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

/** Greedy NMS. Returns indices into `boxes`, highest score first. */
export function nms(boxes: Box[], scores: number[], iouThr = 0.5): number[] {
  // argsort descending by score (matches numpy argsort(-scores)).
  let order = scores.map((_, i) => i).sort((i, j) => scores[j] - scores[i]);
  const keep: number[] = [];
  while (order.length) {
    const i = order.shift()!;
    keep.push(i);
    order = order.filter((j) => iou(boxes[i], boxes[j]) <= iouThr);
  }
  return keep;
}
