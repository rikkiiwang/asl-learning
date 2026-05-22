import type { FailReason } from './decision';

/** Teachable aspects of a sign (sourced from the model's meta.json hint metadata). */
export interface SignHints {
  handshape?: string;
  movement?: string;
  location?: string;
  orientation?: string;
  timing?: string;
  framing?: string;
}

// Most pedagogically useful first.
const ASPECT_ORDER: (keyof SignHints)[] = [
  'movement',
  'handshape',
  'location',
  'orientation',
  'timing',
  'framing',
];

function firstCue(hints?: SignHints | null): string | null {
  if (!hints) return null;
  for (const aspect of ASPECT_ORDER) {
    const value = hints[aspect];
    if (value) return value;
  }
  return null;
}

export interface HintInput {
  failReason: FailReason;
  promptedLabel: string;
  promptedHints?: SignHints | null;
  competitorLabel?: string | null;
}

/**
 * Build a targeted, rule-based hint (PRD Req 10). Uses a teachable cue from the
 * prompted sign's metadata when present; otherwise falls back to clarity/framing
 * guidance so the learner never sees a bare "incorrect".
 */
export function buildHint({ failReason, promptedLabel, promptedHints, competitorLabel }: HintInput): string {
  const cue = firstCue(promptedHints);

  switch (failReason) {
    case 'wrong_sign': {
      const lead = competitorLabel
        ? `That looked more like ${competitorLabel}.`
        : `That didn't match ${promptedLabel}.`;
      return cue
        ? `${lead} For ${promptedLabel}, focus on the ${cue}.`
        : `${lead} For ${promptedLabel}, check your handshape and movement, then try again.`;
    }
    case 'low_confidence':
      return cue
        ? `Almost — make it clearer. For ${promptedLabel}, ${cue}.`
        : `Almost — sign ${promptedLabel} clearly and hold it fully in frame.`;
    case 'ambiguous': {
      const confusedWith = competitorLabel
        ? ` it was easy to confuse with ${competitorLabel}.`
        : ' it was a bit ambiguous.';
      return cue
        ? `Close, but${confusedWith} Emphasize the ${cue}.`
        : `Close, but${confusedWith} Make ${promptedLabel} more distinct.`;
    }
  }
}
