import type { Sign, SignMastery, ProgressSummary } from './types';

/**
 * Roll a learner's per-sign mastery rows up into a dashboard summary.
 * Signs without a mastery row count as not_started; mastery rows whose
 * sign is not in the catalog are ignored.
 */
export function summarizeProgress(signs: Sign[], mastery: SignMastery[]): ProgressSummary {
  const statusBySignId = new Map(mastery.map((m) => [m.sign_id, m.mastery_status]));

  let mastered = 0;
  let learning = 0;
  let notStarted = 0;

  for (const s of signs) {
    switch (statusBySignId.get(s.id) ?? 'not_started') {
      case 'mastered':
        mastered++;
        break;
      case 'learning':
        learning++;
        break;
      default:
        notStarted++;
    }
  }

  const total = signs.length;
  const percentComplete = total === 0 ? 0 : Math.round((mastered / total) * 100);

  return { total, mastered, learning, notStarted, percentComplete };
}

export interface ProgressBarSegments {
  masteredPct: number;
  learningPct: number;
  notStartedPct: number;
}

/** Whole-percent widths for a stacked progress bar; not-started absorbs rounding so they sum to 100. */
export function progressBarSegments(s: ProgressSummary): ProgressBarSegments {
  if (s.total === 0) return { masteredPct: 0, learningPct: 0, notStartedPct: 0 };
  const masteredPct = Math.round((s.mastered / s.total) * 100);
  const learningPct = Math.round((s.learning / s.total) * 100);
  return { masteredPct, learningPct, notStartedPct: 100 - masteredPct - learningPct };
}
