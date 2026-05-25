/**
 * Dev/eval tooling gate (PRD Req 7). OFF by default; enabled only when
 * `VITE_DEV_TOOLS=true` (set in the gitignored .env.local for local dev).
 * The graded pilot build omits the flag, so the pretrained-baseline model and
 * the comparison switcher are excluded from the shipped recognition path.
 */
export function devToolsEnabled(): boolean {
  return import.meta.env.VITE_DEV_TOOLS === 'true';
}
