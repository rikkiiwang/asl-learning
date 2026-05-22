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
}

export const MODELS: ModelOption[] = [
  {
    id: 'own-v1',
    label: 'My model v1 (from scratch)',
    kind: 'own',
    note: 'From-scratch pilot recognizer — version 1.',
  },
  {
    id: 'own-v2',
    label: 'My model v2 (from scratch)',
    kind: 'own',
    note: 'From-scratch pilot recognizer — version 2.',
  },
  {
    id: 'baseline',
    label: 'Baseline (pretrained — eval only)',
    kind: 'baseline',
    note: 'Comparison only. Not part of the graded pilot (Req 7).',
  },
];

export const DEFAULT_MODEL_ID = 'own-v1';

export function getModelById(id: string | null | undefined): ModelOption {
  return MODELS.find((m) => m.id === id) ?? MODELS.find((m) => m.id === DEFAULT_MODEL_ID)!;
}
