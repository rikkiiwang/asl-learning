import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { supabase } from '../db/supabase';
import type { Sign } from '../lib/types';
import { signingSavvyUrl, groupByCategory, filterSigns } from '../lib/signReference';

/**
 * Learn tab: a browsable reference for every sign. Each entry links out to a
 * free ASL dictionary demonstration (we can't host the dataset videos) and
 * deep-links into focused practice. Read-only — no progress tracking here.
 */
export function LearnPage() {
  const navigate = useNavigate();
  const [signs, setSigns] = useState<Sign[]>([]);
  const [query, setQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      const { data, error } = await supabase
        .from('signs')
        .select('id,model_class_index,label,gloss,category')
        .order('label');
      if (error) {
        setError(error.message);
      } else {
        setSigns((data ?? []) as Sign[]);
      }
      setLoading(false);
    })();
  }, []);

  const groups = useMemo(() => groupByCategory(filterSigns(signs, query)), [signs, query]);
  const matchCount = useMemo(() => filterSigns(signs, query).length, [signs, query]);

  return (
    <main className="page stack">
      <div>
        <div className="page-header">
          <h1 className="page-title">Learn signs</h1>
          <Link to="/" className="btn-ghost" style={{ marginLeft: 'auto', padding: '6px 12px', borderRadius: 999 }}>
            ← Back
          </Link>
        </div>
        <p className="subtitle">Watch how each sign is made, then jump straight into practice.</p>
      </div>

      <input
        type="search"
        placeholder="Search a word…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Search signs"
        style={{
          width: '100%', padding: '10px 14px', borderRadius: 12,
          border: '1px solid var(--border, #d4d4d8)', fontSize: 15,
        }}
      />

      {loading && <p className="muted">Loading…</p>}
      {error && <p style={{ color: '#be123c' }}>Couldn’t load signs: {error}</p>}

      {!loading && !error && matchCount === 0 && (
        <p className="muted">No signs match “{query}”.</p>
      )}

      {groups.map((g) => (
        <section key={g.category} className="card">
          <div className="card-title">{g.category}</div>
          <div className="spacer-top">
            {g.signs.map((s) => (
              <div key={s.id} className="recent-row" style={{ alignItems: 'center' }}>
                <span style={{ fontWeight: 700 }}>{s.label}</span>
                {s.gloss && s.gloss !== s.label && <span className="muted">{s.gloss}</span>}
                <span style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
                  <a
                    className="btn-ghost"
                    style={{ padding: '4px 10px', borderRadius: 999, fontSize: 13, textDecoration: 'none' }}
                    href={signingSavvyUrl(s.label)}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Watch ↗
                  </a>
                  <button
                    type="button"
                    className="btn-primary"
                    style={{ padding: '4px 10px', borderRadius: 999, fontSize: 13 }}
                    onClick={() => navigate(`/practice?sign=${s.id}`)}
                  >
                    Practice
                  </button>
                </span>
              </div>
            ))}
          </div>
        </section>
      ))}
    </main>
  );
}
