// Recognition model registry for the in-app comparison switcher.
//
// DEV/EVAL ONLY: the `baseline` (pretrained) option exists to test the pipeline
// and benchmark against the from-scratch model. It is NOT part of the graded
// pilot recognition path (PRD Req 7) and must be stripped from the submission
// build. The actual recognizer implementations are wired in A6 behind a shared
// inference interface; this registry only holds selectable metadata.

export type ModelKind = 'own' | 'baseline';

export interface ModelOption {
  id: string;
  label: string;
  kind: ModelKind;
  note: string;
  /** Short held-out scoreboard, shown in the picker. */
  stats?: string;
  /** Experimental models run but aren't the recommended/shipped recognizer. */
  experimental?: boolean;
}

export const MODELS: ModelOption[] = [
  {
    id: 'own-v1',
    label: 'My model v1 (from scratch)',
    kind: 'own',
    note: 'Shipped recognizer: an end-to-end RGB CNN with a 1500-gloss pretrained encoder.',
    stats: '73% top-1 · 86% top-3 (held-out)',
  },
  {
    id: 'own-v2',
    label: 'My model v2 (experimental)',
    kind: 'own',
    experimental: true,
    note: 'From-scratch 3-stage landmark-geometry pipeline (detector → keypoints → recognizer). Runs live in your browser; lower accuracy than v1.',
    stats: '48% top-1 · 69% top-3 (held-out)',
  },
  {
    id: 'baseline',
    label: 'Baseline (pretrained — eval only)',
    kind: 'baseline',
    note: 'Comparison only. Not part of the graded pilot (Req 7).',
  },
];

export const DEFAULT_MODEL_ID = 'own-v1';

/**
 * Models the user may select. The pretrained `baseline` is dev/eval only
 * (Req 7) and is excluded unless dev tools are enabled, so the graded build
 * cannot route recognition through a pretrained model.
 */
export function selectableModels(includeBaseline: boolean): ModelOption[] {
  return includeBaseline ? MODELS : MODELS.filter((m) => m.kind === 'own');
}

/**
 * Resolve a model id to an option. When `includeBaseline` is false, a stale
 * `baseline` id (e.g. left in localStorage) resolves to the default so it can't
 * reactivate the pretrained path.
 */
export function getModelById(id: string | null | undefined, includeBaseline = true): ModelOption {
  const pool = selectableModels(includeBaseline);
  return pool.find((m) => m.id === id) ?? pool.find((m) => m.id === DEFAULT_MODEL_ID)!;
}
