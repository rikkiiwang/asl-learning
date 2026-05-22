import { describe, it, expect, beforeEach } from 'vitest';
import { readPending, addPending, clearPending } from './pendingAttempts';
import type { AttemptRow } from './session';

// Minimal in-memory Storage stand-in.
function fakeStorage(): Storage {
  const map = new Map<string, string>();
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => void map.set(k, v),
    removeItem: (k) => void map.delete(k),
    clear: () => map.clear(),
    key: () => null,
    length: 0,
  };
}

const row = (n: number): AttemptRow => ({
  user_id: 'u',
  session_id: 's',
  sign_id: `sign${n}`,
  attempt_number: 1,
  is_first_try: true,
  result: 'fail',
  predicted_sign_id: null,
  confidence: 0.1,
  margin: 0.1,
});

let storage: Storage;
beforeEach(() => {
  storage = fakeStorage();
});

describe('pendingAttempts', () => {
  it('reads an empty queue when nothing is stored', () => {
    expect(readPending(storage)).toEqual([]);
  });

  it('reads an empty queue when storage is corrupt', () => {
    storage.setItem('asl.pendingAttempts', '{not json');
    expect(readPending(storage)).toEqual([]);
  });

  it('appends rows and reads them back', () => {
    addPending(row(1), storage);
    addPending(row(2), storage);
    expect(readPending(storage).map((r) => r.sign_id)).toEqual(['sign1', 'sign2']);
  });

  it('caps the queue so it cannot grow unbounded', () => {
    for (let i = 0; i < 80; i++) addPending(row(i), storage);
    expect(readPending(storage).length).toBeLessThanOrEqual(50);
  });

  it('clears the queue', () => {
    addPending(row(1), storage);
    clearPending(storage);
    expect(readPending(storage)).toEqual([]);
  });
});
