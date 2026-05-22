# Supabase Setup (Stream A)

One-time setup to back the app with auth + progress storage.

## 1. Create a project
1. Go to https://supabase.com → sign in → **New project**.
2. Name it (e.g. `asl-learning`), set a **database password** (save it), pick the closest region.
3. Wait ~2 min for provisioning.

## 2. Get the API credentials
**Project Settings → API** (or **API Keys**). Copy:
- **Project URL** → used for both `VITE_SUPABASE_URL` and `SUPABASE_URL`.
- **anon / publishable** key → `VITE_SUPABASE_ANON_KEY` (safe in the browser; RLS-gated).
- **service_role / secret** key → `SUPABASE_SERVICE_ROLE_KEY` (**secret — server/seed only, never in the browser bundle**).

## 3. Fill local env
```bash
cd app
cp .env.example .env.local
# edit .env.local with the values above
```
`.env.local` is gitignored. The service-role key has no `VITE_` prefix, so Vite never ships it to the browser.

## 4. Apply the schema
**Dashboard → SQL Editor → New query** → paste the contents of
`app/supabase/migrations/0001_init.sql` → **Run**.

Creates `signs`, `sessions`, `attempts`, `sign_mastery`, the mastery trigger, and RLS policies.

_(CLI alternative: `npx supabase link --project-ref <ref>` then `npx supabase db push`.)_

## 5. Seed the 75 signs
```bash
cd app
npm run seed          # reads ../model/artifacts/manifest/manifest.json
# → "Seeded 75 signs ..."
```

## 6. Verify
- **Dashboard → Table Editor → signs** shows **75 rows** (label / gloss / model_class_index / category).
- `model_class_index` runs 0–74 (must match the model's training order).

## 7. Auth (for pilot testing)
**Authentication → Providers → Email** is on by default. For frictionless test
accounts, consider **Authentication → Sign In / Providers → disable "Confirm email"**
during the pilot (re-enable for anything beyond controlled testing).

---
**Privacy note:** only outcome metadata is ever written here (see `attempts`). Camera
frames never reach Supabase — there is no upload path (PRD Req 5 & 13).
