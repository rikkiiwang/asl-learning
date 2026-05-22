import type { Sign, SignMastery } from './types';

/**
 * Build a practice deck (spec §3): unmastered signs first, ordered by
 * least-recently-practiced (never-practiced first); when the deck would be all
 * unmastered, reserve `reviewSlots` for mastered signs (spaced review).
 */
export function buildDeck(
  signs: Sign[],
  mastery: SignMastery[],
  size = 10,
  reviewSlots = 2,
): Sign[] {
  const byId = new Map(mastery.map((m) => [m.sign_id, m]));
  const isMastered = (s: Sign) => byId.get(s.id)?.mastery_status === 'mastered';
  const lastPracticed = (s: Sign) => byId.get(s.id)?.last_practiced_at ?? ''; // '' (never) sorts first
  const byLeastRecent = (a: Sign, b: Sign) => lastPracticed(a).localeCompare(lastPracticed(b));

  const unmastered = signs.filter((s) => !isMastered(s)).sort(byLeastRecent);
  const mastered = signs.filter(isMastered).sort(byLeastRecent);

  if (unmastered.length >= size && mastered.length > 0) {
    return [...unmastered.slice(0, size - reviewSlots), ...mastered.slice(0, reviewSlots)];
  }
  return [...unmastered, ...mastered].slice(0, size);
}
