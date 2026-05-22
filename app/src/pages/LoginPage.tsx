import { useState, type FormEvent } from 'react';
import { Navigate } from 'react-router-dom';
import { supabase } from '../db/supabase';
import { useSession } from '../auth/SessionProvider';

type Mode = 'signin' | 'signup';

export function LoginPage() {
  const { session, loading } = useSession();
  const [mode, setMode] = useState<Mode>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (!loading && session) return <Navigate to="/" replace />;

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setNotice(null);
    const { error } =
      mode === 'signin'
        ? await supabase.auth.signInWithPassword({ email, password })
        : await supabase.auth.signUp({ email, password });
    setBusy(false);
    if (error) {
      setError(error.message);
    } else if (mode === 'signup') {
      setNotice('Account created. If email confirmation is on, check your inbox; otherwise sign in.');
      setMode('signin');
    }
  }

  return (
    <main className="centered" style={{ maxWidth: 380 }}>
      <h1 className="page-title" style={{ fontSize: 28 }}>
        ASL Practice
      </h1>
      <p className="subtitle">{mode === 'signin' ? 'Sign in to continue.' : 'Create an account.'}</p>
      <form onSubmit={onSubmit} className="stack spacer-top" style={{ gap: 12 }}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          autoComplete="email"
          required
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
          required
          minLength={6}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit" className="btn-primary btn-block" disabled={busy}>
          {busy ? '…' : mode === 'signin' ? 'Sign in' : 'Sign up'}
        </button>
      </form>
      {error && <p style={{ color: '#be123c' }}>{error}</p>}
      {notice && <p style={{ color: '#15803d' }}>{notice}</p>}
      <button
        type="button"
        className="btn-ghost spacer-top"
        onClick={() => setMode(mode === 'signin' ? 'signup' : 'signin')}
      >
        {mode === 'signin' ? 'Need an account? Sign up' : 'Have an account? Sign in'}
      </button>
    </main>
  );
}
