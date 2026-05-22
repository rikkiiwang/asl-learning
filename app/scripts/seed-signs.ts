/**
 * Seed the `signs` catalog from the model's authoritative manifest.
 * Run server-side with the service role key (bypasses RLS):
 *
 *   SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... npm run seed
 *
 * Defaults to ../model/artifacts/manifest/manifest.json (override with MANIFEST_PATH).
 */
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createClient } from '@supabase/supabase-js';
import { buildSignSeed } from '../src/lib/signSeed';
import type { Manifest } from '../src/lib/types';

const __dirname = dirname(fileURLToPath(import.meta.url));

const url = process.env.SUPABASE_URL;
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
if (!url || !serviceKey) {
  console.error('Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (see .env.example).');
  process.exit(1);
}

const manifestPath =
  process.env.MANIFEST_PATH || resolve(__dirname, '../../model/artifacts/manifest/manifest.json');

const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Manifest;
const rows = buildSignSeed(manifest);

const supabase = createClient(url, serviceKey, { auth: { persistSession: false } });

const { error } = await supabase.from('signs').upsert(rows, { onConflict: 'model_class_index' });
if (error) {
  console.error('Seed failed:', error.message);
  process.exit(1);
}
console.log(`Seeded ${rows.length} signs from ${manifestPath}`);
