import type { AttemptRow } from './session';

// Offline durability for attempt writes (spec §8): if a write fails, stash the
// row locally and re-send on next load. Capped so it can't grow unbounded.
const KEY = 'asl.pendingAttempts';
const MAX = 50;

function storageOrNull(storage?: Storage): Storage | null {
  if (storage) return storage;
  return typeof localStorage !== 'undefined' ? localStorage : null;
}

export function readPending(storage?: Storage): AttemptRow[] {
  const s = storageOrNull(storage);
  if (!s) return [];
  try {
    const parsed = JSON.parse(s.getItem(KEY) ?? '[]');
    return Array.isArray(parsed) ? (parsed as AttemptRow[]) : [];
  } catch {
    return [];
  }
}

export function addPending(row: AttemptRow, storage?: Storage): void {
  const s = storageOrNull(storage);
  if (!s) return;
  const queue = [...readPending(s), row].slice(-MAX);
  s.setItem(KEY, JSON.stringify(queue));
}

export function clearPending(storage?: Storage): void {
  storageOrNull(storage)?.removeItem(KEY);
}
