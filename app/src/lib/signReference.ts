import type { Sign } from './types';

/**
 * External demonstration link for a sign. We can't host the dataset videos
 * (license), so the Learn tab and the practice "reveal" both point at a free
 * public ASL dictionary. Single source of truth for that URL.
 */
export function signingSavvyUrl(label: string): string {
  return `https://www.signingsavvy.com/search/${encodeURIComponent(label.toLowerCase())}`;
}

export interface SignCategoryGroup {
  category: string;
  signs: Sign[];
}

const UNCATEGORIZED = 'Other';

/**
 * Group signs by category for the Learn tab. Categories are sorted
 * alphabetically (uncategorized last); signs within a category keep input
 * order (callers pass them ordered by model_class_index / label).
 */
export function groupByCategory(signs: Sign[]): SignCategoryGroup[] {
  const byCat = new Map<string, Sign[]>();
  for (const s of signs) {
    const cat = s.category?.trim() || UNCATEGORIZED;
    const list = byCat.get(cat) ?? [];
    list.push(s);
    byCat.set(cat, list);
  }
  return [...byCat.entries()]
    .sort(([a], [b]) => {
      if (a === UNCATEGORIZED) return 1;
      if (b === UNCATEGORIZED) return -1;
      return a.localeCompare(b);
    })
    .map(([category, list]) => ({ category, signs: list }));
}

/** Case-insensitive filter on label + gloss. Empty query returns all. */
export function filterSigns(signs: Sign[], query: string): Sign[] {
  const q = query.trim().toLowerCase();
  if (!q) return signs;
  return signs.filter(
    (s) => s.label.toLowerCase().includes(q) || (s.gloss ?? '').toLowerCase().includes(q),
  );
}
