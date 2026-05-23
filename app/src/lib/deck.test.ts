import { describe, it, expect } from 'vitest';
import { buildDeck, targetedDeck } from './deck';
import type { Sign, SignMastery, MasteryStatus } from './types';

function sign(id: string, idx: number): Sign {
  return { id, model_class_index: idx, label: `S${idx}`, gloss: `S${idx}`, category: null };
}
function mastery(sign_id: string, status: MasteryStatus, last: string | null): SignMastery {
  return {
    sign_id,
    mastery_status: status,
    total_attempts: 1,
    total_passes: status === 'mastered' ? 2 : 0,
    first_try_pass_session_count: status === 'mastered' ? 2 : 0,
    last_practiced_at: last,
    last_result: null,
  };
}

const signs: Sign[] = Array.from({ length: 12 }, (_, i) => sign(`s${i}`, i));

describe('buildDeck', () => {
  it('returns the first `size` signs when nothing has been practiced', () => {
    const deck = buildDeck(signs, [], 10);
    expect(deck).toHaveLength(10);
    expect(deck.map((s) => s.id)).toEqual(signs.slice(0, 10).map((s) => s.id));
  });

  it('returns all signs when there are fewer than `size`', () => {
    expect(buildDeck(signs.slice(0, 4), [], 10)).toHaveLength(4);
  });

  it('orders unmastered by least-recently-practiced (never-practiced first)', () => {
    const three = [sign('a', 0), sign('b', 1), sign('c', 2)];
    const m = [mastery('a', 'learning', '2026-05-20'), mastery('c', 'learning', '2026-05-21')];
    // b never practiced → first; a (older) → before c (newer)
    expect(buildDeck(three, m, 10).map((s) => s.id)).toEqual(['b', 'a', 'c']);
  });

  it('reserves review slots for mastered signs when the deck is full of unmastered', () => {
    const m = [mastery('s10', 'mastered', '2026-05-19'), mastery('s11', 'mastered', '2026-05-19')];
    const deck = buildDeck(signs, m, 10, 2);
    expect(deck).toHaveLength(10);
    const ids = deck.map((s) => s.id);
    expect(ids).toContain('s10');
    expect(ids).toContain('s11');
    // last two slots are the mastered reviews
    expect(ids.slice(-2).sort()).toEqual(['s10', 's11']);
  });

  it('puts unmastered ahead of mastered', () => {
    const m = [mastery('s0', 'mastered', '2026-05-19')];
    const deck = buildDeck(signs.slice(0, 3), m, 10);
    expect(deck[deck.length - 1].id).toBe('s0'); // mastered last
  });
});

describe('targetedDeck', () => {
  it('returns the single matching sign', () => {
    const deck = targetedDeck(signs, 's3');
    expect(deck).toHaveLength(1);
    expect(deck[0].id).toBe('s3');
  });
  it('returns an empty deck for an unknown id', () => {
    expect(targetedDeck(signs, 'nope')).toEqual([]);
  });
});
