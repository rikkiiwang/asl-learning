-- ASL Learning — initial schema (spec §5)
-- Tables: signs (catalog), sessions, attempts (metadata only), sign_mastery (rollup).
-- Privacy: attempts store outcome metadata only — never frames/video (PRD Req 13).

-- ---------------------------------------------------------------------------
-- signs: vocabulary catalog. Seeded from the model's authoritative manifest;
-- model_class_index is owned by the model side and never invented in the app.
-- hints / reference_clip_ref are enriched later from meta.json (M7).
-- ---------------------------------------------------------------------------
create table if not exists public.signs (
  id                  uuid primary key default gen_random_uuid(),
  model_class_index   int  not null unique,
  label               text not null unique,
  gloss               text not null,
  category            text,
  hints               jsonb,
  reference_clip_ref  text,
  created_at          timestamptz not null default now()
);

create table if not exists public.sessions (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  started_at  timestamptz not null default now(),
  ended_at    timestamptz
);

create table if not exists public.attempts (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references auth.users(id) on delete cascade,
  session_id         uuid not null references public.sessions(id) on delete cascade,
  sign_id            uuid not null references public.signs(id),
  attempt_number     int  not null,            -- 1..3 within one encounter
  is_first_try       boolean not null,         -- attempt_number = 1
  result             text not null check (result in ('pass','fail')),
  predicted_sign_id  uuid references public.signs(id),
  confidence         real,
  margin             real,
  created_at         timestamptz not null default now()
);
create index if not exists attempts_user_sign_idx on public.attempts (user_id, sign_id);
create index if not exists attempts_session_idx   on public.attempts (session_id);

create table if not exists public.sign_mastery (
  user_id                       uuid not null references auth.users(id) on delete cascade,
  sign_id                       uuid not null references public.signs(id) on delete cascade,
  total_attempts                int  not null default 0,
  total_passes                  int  not null default 0,
  first_try_pass_session_count  int  not null default 0,
  mastery_status                text not null default 'not_started'
                                  check (mastery_status in ('not_started','learning','mastered')),
  last_practiced_at             timestamptz,
  last_result                   text check (last_result in ('pass','fail')),
  primary key (user_id, sign_id)
);

-- ---------------------------------------------------------------------------
-- Mastery rollup trigger (spec §5): mastered = 2 first-try passes across
-- 2 DISTINCT sessions. Recomputed from attempts on each insert (idempotent,
-- correct at pilot data scale). security definer so it can write through RLS.
-- ---------------------------------------------------------------------------
create or replace function public.update_sign_mastery()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  v_total    int;
  v_passes   int;
  v_sessions int;
  v_status   text;
begin
  select count(*),
         count(*) filter (where result = 'pass'),
         count(distinct session_id) filter (where is_first_try and result = 'pass')
    into v_total, v_passes, v_sessions
    from public.attempts
   where user_id = new.user_id and sign_id = new.sign_id;

  v_status := case
                when v_sessions >= 2 then 'mastered'
                when v_total    >  0 then 'learning'
                else 'not_started'
              end;

  insert into public.sign_mastery as m
    (user_id, sign_id, total_attempts, total_passes, first_try_pass_session_count,
     mastery_status, last_practiced_at, last_result)
  values
    (new.user_id, new.sign_id, v_total, v_passes, v_sessions,
     v_status, new.created_at, new.result)
  on conflict (user_id, sign_id) do update set
    total_attempts               = excluded.total_attempts,
    total_passes                 = excluded.total_passes,
    first_try_pass_session_count = excluded.first_try_pass_session_count,
    mastery_status               = excluded.mastery_status,
    last_practiced_at            = excluded.last_practiced_at,
    last_result                  = excluded.last_result;

  return new;
end;
$$;

drop trigger if exists trg_update_sign_mastery on public.attempts;
create trigger trg_update_sign_mastery
  after insert on public.attempts
  for each row execute function public.update_sign_mastery();

-- ---------------------------------------------------------------------------
-- Row-level security: learners see only their own data; signs catalog is
-- read-only to authenticated users (writes go through the service role seed).
-- ---------------------------------------------------------------------------
alter table public.signs        enable row level security;
alter table public.sessions     enable row level security;
alter table public.attempts     enable row level security;
alter table public.sign_mastery enable row level security;

create policy "signs are readable by authenticated users"
  on public.signs for select to authenticated using (true);

create policy "own sessions"
  on public.sessions for all to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());

create policy "own attempts"
  on public.attempts for all to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());

create policy "own mastery is readable"
  on public.sign_mastery for select to authenticated
  using (user_id = auth.uid());
