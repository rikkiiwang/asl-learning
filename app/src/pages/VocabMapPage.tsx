import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { supabase } from '../db/supabase';
import { summarizeProgress } from '../lib/progress';
import type { ProgressSummary, Sign, SignMastery } from '../lib/types';
import { VocabGrid } from '../components/VocabGrid';
import { SignDetailModal } from '../components/SignDetailModal';
import { statusMap } from '../lib/vocabStatus';

/** Vocabulary map: a status-colored grid of every sign; tap one for detail + practice. */
export function VocabMapPage() {
  const [signs, setSigns] = useState<Sign[]>([]);
  const [mastery, setMastery] = useState<SignMastery[]>([]);
  const [summary, setSummary] = useState<ProgressSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<Sign | null>(null);

  useEffect(() => {
    (async () => {
      const [signsRes, masteryRes] = await Promise.all([
        supabase.from('signs').select('id,model_class_index,label,gloss,category').order('model_class_index'),
        supabase
          .from('sign_mastery')
          .select('sign_id,mastery_status,total_attempts,total_passes,first_try_pass_session_count,last_practiced_at,last_result'),
      ]);
      if (signsRes.error || masteryRes.error) {
        setError((signsRes.error ?? masteryRes.error)!.message);
        setLoading(false);
        return;
      }
      const s = (signsRes.data ?? []) as Sign[];
      const m = (masteryRes.data ?? []) as SignMastery[];
      setSigns(s);
      setMastery(m);
      setSummary(summarizeProgress(s, m));
      setLoading(false);
    })();
  }, []);

  const masteryById = new Map(mastery.map((m) => [m.sign_id, m]));

  return (
    <main className="page stack">
      <div>
        <div className="page-header">
          <h1 className="page-title">Vocabulary map</h1>
          {summary && <span className="pill pill-soft">{summary.percentComplete}% mastered</span>}
          <Link to="/" className="btn-ghost" style={{ marginLeft: 'auto', padding: '6px 12px', borderRadius: 999 }}>
            ← Back
          </Link>
        </div>
        <p className="subtitle">Every sign, colored by how far you’ve gotten. Tap one to practice it.</p>
      </div>

      {loading && <p className="muted">Loading…</p>}
      {error && <p style={{ color: '#be123c' }}>Couldn’t load the map: {error}</p>}

      {summary && (
        <>
          <div className="legend-row">
            <span className="legend"><span className="dot" style={{ background: 'var(--seg-mastered)' }} />{summary.mastered} mastered</span>
            <span className="legend"><span className="dot" style={{ background: 'var(--seg-learning)' }} />{summary.learning} learning</span>
            <span className="legend"><span className="dot" style={{ background: 'var(--seg-togo)' }} />{summary.notStarted} to go</span>
          </div>
          <VocabGrid signs={signs} statusById={statusMap(mastery)} onSelect={setSelected} />
        </>
      )}

      {selected && (
        <SignDetailModal sign={selected} mastery={masteryById.get(selected.id)} onClose={() => setSelected(null)} />
      )}
    </main>
  );
}
