import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { supabase } from '../db/supabase';
import { summarizeProgress, progressBarSegments } from '../lib/progress';
import { timeAgo } from '../lib/time';
import type { ProgressSummary, Sign, SignMastery } from '../lib/types';
import { ModelSwitcher } from '../components/ModelSwitcher';

interface RecentItem {
  label: string;
  result: 'pass' | 'fail';
  at: string;
}

export function DashboardPage() {
  const [summary, setSummary] = useState<ProgressSummary | null>(null);
  const [recent, setRecent] = useState<RecentItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const [signsRes, masteryRes, attemptsRes] = await Promise.all([
        supabase.from('signs').select('id,model_class_index,label,gloss,category').order('model_class_index'),
        supabase
          .from('sign_mastery')
          .select('sign_id,mastery_status,total_attempts,total_passes,first_try_pass_session_count,last_practiced_at,last_result'),
        supabase.from('attempts').select('sign_id,result,created_at').order('created_at', { ascending: false }).limit(8),
      ]);
      if (signsRes.error || masteryRes.error) {
        setError((signsRes.error ?? masteryRes.error)!.message);
        setLoading(false);
        return;
      }
      const signs = (signsRes.data ?? []) as Sign[];
      const idToLabel = new Map(signs.map((s) => [s.id, s.label]));
      setSummary(summarizeProgress(signs, (masteryRes.data ?? []) as SignMastery[]));
      setRecent(
        ((attemptsRes.data ?? []) as { sign_id: string; result: 'pass' | 'fail'; created_at: string }[]).map((a) => ({
          label: idToLabel.get(a.sign_id) ?? '—',
          result: a.result,
          at: a.created_at,
        })),
      );
      setLoading(false);
    })();
  }, []);

  return (
    <main className="page stack">
      <div>
        <div className="page-header">
          <h1 className="page-title">ASL Practice</h1>
        </div>
        <p className="subtitle">Level up your signs!</p>
      </div>

      {loading && <p className="muted">Loading…</p>}
      {error && <p style={{ color: '#be123c' }}>Couldn’t load progress: {error}</p>}

      {summary && (
        <>
          <ProgressCard summary={summary} />

          <Link to="/practice" className="btn btn-primary btn-block">
            ▶ Start practice
          </Link>

          <RecentHistory items={recent} />

          <ModelSwitcher />
        </>
      )}
    </main>
  );
}

function ProgressCard({ summary }: { summary: ProgressSummary }) {
  const seg = progressBarSegments(summary);
  return (
    <section className="card">
      <div className="progress-head">
        <span style={{ fontWeight: 800 }}>Your progress</span>
        <span className="pill pill-soft">{summary.percentComplete}% mastered</span>
      </div>
      <div className="progress-track">
        <div className="progress-seg" style={{ width: `${seg.masteredPct}%`, background: 'var(--seg-mastered)' }} />
        <div className="progress-seg" style={{ width: `${seg.learningPct}%`, background: 'var(--seg-learning)' }} />
        <div className="progress-seg" style={{ width: `${seg.notStartedPct}%`, background: 'var(--seg-togo)' }} />
      </div>
      <div className="legend-row">
        <Legend color="var(--seg-mastered)" label={`${summary.mastered} mastered`} />
        <Legend color="var(--seg-learning)" label={`${summary.learning} learning`} />
        <Legend color="var(--seg-togo)" label={`${summary.notStarted} to go`} />
        <span style={{ marginLeft: 'auto' }}>{summary.total} signs</span>
      </div>
    </section>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="legend">
      <span className="dot" style={{ background: color }} />
      {label}
    </span>
  );
}

function RecentHistory({ items }: { items: RecentItem[] }) {
  if (items.length === 0) return null;
  return (
    <section className="card">
      <div className="card-title">Recent practice 🏅</div>
      <div className="spacer-top">
        {items.map((it, i) => (
          <div key={i} className="recent-row">
            <span className="dot" style={{ background: it.result === 'pass' ? 'var(--pass)' : 'var(--fail)' }} />
            <span style={{ fontWeight: 700 }}>{it.label}</span>
            <span className="muted">{it.result === 'pass' ? '✓ passed' : '✗ missed'}</span>
            <span className="when">{timeAgo(it.at)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
