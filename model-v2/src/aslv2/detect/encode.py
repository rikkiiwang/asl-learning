"""Anchor<->GT encoding (SSD-style) and inference decode with NMS."""
import numpy as np
from aslv2.boxes import iou, nms

def _xyxy_to_cwh(b):
    return np.stack([(b[:,0]+b[:,2])/2, (b[:,1]+b[:,3])/2, b[:,2]-b[:,0], b[:,3]-b[:,1]], 1)

def encode_targets(anchors, gt, labels, iou_pos=0.5, iou_neg=0.4):
    """Returns:
      cls_tgt (n,2) float : one-hot at the GT class for positives; ALL ZEROS for
                            background (a valid negative); zeros for ignored too.
      box_t   (n,4) float : SSD center-form deltas for positives (0 elsewhere).
      pos     (n,)  bool  : positive anchors (IoU>=iou_pos, plus best anchor/GT).
      valid   (n,)  bool  : anchors INCLUDED in the cls loss = positives OR clear
                            background (max IoU < iou_neg). The in-between band
                            (iou_neg<=maxIoU<iou_pos, non-positive) is valid=False.
    """
    n = len(anchors)
    cls_tgt = np.zeros((n, 2), np.float32)   # all-zero = background
    box_t = np.zeros((n, 4), np.float32)
    pos = np.zeros(n, bool)
    if len(gt) == 0:
        valid = np.ones(n, bool)             # all background, all valid
        return cls_tgt, box_t, pos, valid
    ious = np.zeros((n, len(gt)))
    for i in range(n):
        for j in range(len(gt)):
            ious[i, j] = iou(anchors[i], gt[j])
    best_gt = ious.argmax(1); best_iou = ious.max(1)
    for j in range(len(gt)):              # force best anchor per GT positive
        best_iou[ious[:, j].argmax()] = 1.0; best_gt[ious[:, j].argmax()] = j
    pos = best_iou >= iou_pos
    a = _xyxy_to_cwh(anchors); g = _xyxy_to_cwh(gt)
    for i in np.where(pos)[0]:
        j = best_gt[i]
        cls_tgt[i, labels[j]] = 1.0        # one-hot at GT class
        box_t[i] = [(g[j,0]-a[i,0])/a[i,2], (g[j,1]-a[i,1])/a[i,3],
                    np.log(g[j,2]/a[i,2]), np.log(g[j,3]/a[i,3])]
    # valid = positives OR clear background (max_iou < iou_neg)
    # the in-between band [iou_neg, iou_pos) that is not positive is excluded
    valid = pos | (best_iou < iou_neg)
    return cls_tgt, box_t, pos, valid

def decode(anchors, deltas, scores, score_thr=0.5, iou_thr=0.5):
    a = _xyxy_to_cwh(anchors)
    cx = deltas[:,0]*a[:,2]+a[:,0]; cy = deltas[:,1]*a[:,3]+a[:,1]
    w = np.exp(deltas[:,2])*a[:,2]; h = np.exp(deltas[:,3])*a[:,3]
    boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 1)
    # scores: (n, 2) logits per foreground class
    prob = 1/(1+np.exp(-scores)); cls = prob.argmax(1); conf = prob.max(1)
    keep_b, keep_s, keep_l = [], [], []
    for c in (0, 1):
        m = (cls == c) & (conf >= score_thr)
        if m.sum() == 0: continue
        idx = nms(boxes[m], conf[m], iou_thr)
        bm = boxes[m]; cm = conf[m]
        for k in idx:
            keep_b.append(bm[k]); keep_s.append(cm[k]); keep_l.append(c)
    return (np.array(keep_b).reshape(-1,4), np.array(keep_s), np.array(keep_l, int))
