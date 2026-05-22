// Domain types shared across the app.
// The model side is the source of truth for vocabulary + class indices;
// these mirror the relevant slices of the model's manifest.json / meta.json.

/** A sign entry as it appears in the model's authoritative manifest.json. */
export interface ManifestSign {
  label: string;
  gloss: string;
  label_idx: number;
  category: string | null;
  counts?: Record<string, number>;
}

/** Relevant slice of model/artifacts/manifest/manifest.json. */
export interface Manifest {
  version: string;
  num_classes: number;
  labels: string[];
  signs: ManifestSign[];
  input?: { frames: number; size: number; window_seconds?: number };
}

/** A row to upsert into the `signs` catalog table. */
export interface SignSeedRow {
  model_class_index: number;
  label: string;
  gloss: string;
  category: string | null;
}

export type MasteryStatus = 'not_started' | 'learning' | 'mastered';

/** A row from the `signs` catalog (as read by the app). */
export interface Sign {
  id: string;
  model_class_index: number;
  label: string;
  gloss: string;
  category: string | null;
}

/** A per-user-per-sign rollup row from `sign_mastery`. */
export interface SignMastery {
  sign_id: string;
  mastery_status: MasteryStatus;
  total_attempts: number;
  total_passes: number;
  first_try_pass_session_count: number;
  last_practiced_at: string | null;
  last_result: 'pass' | 'fail' | null;
}

/** Dashboard rollup of a learner's progress across the vocabulary. */
export interface ProgressSummary {
  total: number;
  mastered: number;
  learning: number;
  notStarted: number;
  percentComplete: number;
}
