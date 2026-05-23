import type { MasteryStatus, SignMastery } from './types';

/** sign-id → mastery status lookup for the vocab map grid. */
export function statusMap(mastery: SignMastery[]): Map<string, MasteryStatus> {
  return new Map(mastery.map((m) => [m.sign_id, m.mastery_status]));
}

/** Display color (CSS var) + label for each mastery status — used by the vocab map. */
export function statusStyle(status: MasteryStatus): { color: string; label: string } {
  switch (status) {
    case 'mastered':
      return { color: 'var(--seg-mastered)', label: 'Mastered' };
    case 'learning':
      return { color: 'var(--seg-learning)', label: 'Learning' };
    default:
      return { color: 'var(--seg-togo)', label: 'Not started' };
  }
}
