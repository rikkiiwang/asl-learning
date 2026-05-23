import { useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useCamera } from '../camera/useCamera';
import { sampleMeanLuminance } from '../camera/sampleLuminance';
import { assessBrightness, type BrightnessStatus } from '../lib/camera';
import { recordClip } from '../capture/recordClip';
import { framesToTensor } from '../lib/clipTensor';
import { softmax, topK } from '../lib/inference';
import { decidePassFail, type Decision } from '../lib/decision';
import { buildHint } from '../lib/hints';
import { buildDeck, targetedDeck } from '../lib/deck';
import { wordOutcome, buildAttemptRow, type WordOutcome } from '../lib/session';
import { readPending, addPending, clearPending } from '../lib/pendingAttempts';
import { localMidnightISO } from '../lib/time';
import { computeStreak } from '../lib/streak';
import { goalProgress } from '../lib/dailyGoal';
import { SessionComplete } from '../components/SessionComplete';
import { createRecognizer } from '../inference/recognizer';
import { useModel } from '../models/ModelProvider';
import { useSession } from '../auth/SessionProvider';
import { supabase } from '../db/supabase';
import type { Sign, SignMastery } from '../lib/types';

type Phase = 'loading' | 'no_session' | 'practicing' | 'finished';
type Step = 'ready' | 'counting' | 'recording' | 'predicting' | 'result';

interface AttemptResult {
  decision: Decision;
  outcome: WordOutcome;
  hint: string | null;
  top: { label: string; prob: number }[];
}

const BRIGHTNESS_COPY: Record<BrightnessStatus, { text: string; color: string }> = {
  ok: { text: 'Lighting looks good', color: '#15803d' },
  too_dark: { text: 'Too dark — add more light', color: '#b4690e' },
  too_bright: { text: 'Too bright — reduce backlight', color: '#b4690e' },
};

// Replace with meta.json mean/std at M7; the stub recognizer ignores exact values.
const PLACEHOLDER_NORM = { mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5] } as const;
const DECK_SIZE = 10;

const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export function PracticePage() {
  const { session: authSession, loading: authLoading } = useSession();
  const [searchParams] = useSearchParams();
  const targetSignId = searchParams.get('sign');
  const { selected } = useModel();
  const { videoRef, state: cam, error: camError, start } = useCamera();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const initRef = useRef(false);

  const [phase, setPhase] = useState<Phase>('loading');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [byClassIndex, setByClassIndex] = useState<Map<number, Sign>>(new Map());
  const [deck, setDeck] = useState<Sign[]>([]);
  const [deckIndex, setDeckIndex] = useState(0);
  const [attemptNumber, setAttemptNumber] = useState(1);
  const [step, setStep] = useState<Step>('ready');
  const [countdown, setCountdown] = useState(0);
  const [brightness, setBrightness] = useState<BrightnessStatus | null>(null);
  const [result, setResult] = useState<AttemptResult | null>(null);
  const [stats, setStats] = useState({ completed: 0, passed: 0 });
  const [finishInfo, setFinishInfo] = useState({ streak: 0, goalClosed: false });

  const current = deck[deckIndex] ?? null;

  useEffect(() => {
    void start();
  }, [start]);

  // One-time session init once auth is resolved.
  useEffect(() => {
    if (authLoading || initRef.current) return;
    const uid = authSession?.user?.id;
    if (!uid) {
      setPhase('no_session');
      return;
    }
    initRef.current = true;
    void (async () => {
      // Re-send any attempts that failed to write on a previous (offline) visit.
      const pending = readPending();
      if (pending.length) {
        const { error } = await supabase.from('attempts').insert(pending);
        if (!error) clearPending();
      }

      const created = await supabase.from('sessions').insert({ user_id: uid }).select('id').single();
      if (created.error || !created.data) {
        setPhase('no_session');
        return;
      }
      const [signsRes, masteryRes] = await Promise.all([
        supabase.from('signs').select('id,model_class_index,label,gloss,category').order('model_class_index'),
        supabase
          .from('sign_mastery')
          .select('sign_id,mastery_status,total_attempts,total_passes,first_try_pass_session_count,last_practiced_at,last_result'),
      ]);
      const signs = (signsRes.data ?? []) as Sign[];
      setByClassIndex(new Map(signs.map((s) => [s.model_class_index, s])));
      // Deep-link from the vocab map (?sign=…) practices just that sign; otherwise the normal deck.
      const focused = targetSignId ? targetedDeck(signs, targetSignId) : [];
      setDeck(focused.length ? focused : buildDeck(signs, (masteryRes.data ?? []) as SignMastery[], DECK_SIZE));
      setUserId(uid);
      setSessionId(created.data.id);
      setPhase('practicing');
    })();
  }, [authLoading, authSession, targetSignId]);

  useEffect(() => {
    if (cam !== 'ready' || step === 'recording') return;
    const id = setInterval(() => {
      if (videoRef.current && canvasRef.current) {
        const luma = sampleMeanLuminance(videoRef.current, canvasRef.current);
        if (luma !== null) setBrightness(assessBrightness(luma));
      }
    }, 500);
    return () => clearInterval(id);
  }, [cam, step, videoRef]);

  async function runAttempt() {
    if (!videoRef.current || !current || !sessionId || !userId) return;
    setResult(null);
    setStep('counting');
    for (const n of [3, 2, 1]) {
      setCountdown(n);
      await delay(700);
    }

    setStep('recording');
    const clip = await recordClip(videoRef.current, { durationMs: 3000 });

    setStep('predicting');
    const tensor = framesToTensor(clip.frames, clip.size, PLACEHOLDER_NORM.mean, PLACEHOLDER_NORM.std);
    const recognizer = createRecognizer(selected, byClassIndex.size || 75);
    const probs = softmax(await recognizer.recognize(tensor));

    const promptIndex = current.model_class_index;
    const decision = decidePassFail(probs, promptIndex);
    const predicted = byClassIndex.get(decision.predictedIndex) ?? null;
    const outcome = wordOutcome(attemptNumber, decision.pass);
    const hint = decision.pass
      ? null
      : buildHint({ failReason: decision.failReason!, promptedLabel: current.label, competitorLabel: predicted?.label ?? null });
    const top = topK(probs, 3).map((r) => ({ label: byClassIndex.get(r.index)?.label ?? `class ${r.index}`, prob: r.prob }));

    // Persist (fires the mastery trigger). Never block the loop on a write error.
    const row = buildAttemptRow({ userId, sessionId, signId: current.id, attemptNumber, decision, predictedSignId: predicted?.id ?? null });
    supabase.from('attempts').insert(row).then(({ error }) => {
      if (error) {
        console.warn('attempt write failed, queued for retry:', error.message);
        addPending(row);
      }
    });

    setResult({ decision, outcome, hint, top });
    setStep('result');
  }

  function advance(passed: boolean) {
    setStats((s) => ({ completed: s.completed + 1, passed: s.passed + (passed ? 1 : 0) }));
    setResult(null);
    setAttemptNumber(1);
    if (deckIndex + 1 >= deck.length) {
      void finish();
    } else {
      setDeckIndex((i) => i + 1);
      setStep('ready');
    }
  }

  function retry() {
    setResult(null);
    setAttemptNumber((n) => n + 1);
    setStep('ready');
  }

  async function finish() {
    if (sessionId && userId) {
      await supabase.from('sessions').update({ ended_at: new Date().toISOString() }).eq('id', sessionId);
      const [sessionsRes, todayRes] = await Promise.all([
        supabase.from('sessions').select('started_at'),
        supabase.from('attempts').select('sign_id').gte('created_at', localMidnightISO()),
      ]);
      const streak = computeStreak(((sessionsRes.data ?? []) as { started_at: string }[]).map((s) => s.started_at));
      const practiced = new Set(((todayRes.data ?? []) as { sign_id: string }[]).map((a) => a.sign_id)).size;
      setFinishInfo({ streak, goalClosed: goalProgress(practiced).complete });
    }
    setPhase('finished');
  }

  if (phase === 'loading') return <Centered>Setting up your session…</Centered>;
  if (phase === 'no_session')
    return (
      <Centered>
        Couldn’t start a session. Enable “Anonymous sign-ins” in Supabase, then reload.{' '}
        <Link to="/">← Back</Link>
      </Centered>
    );
  if (phase === 'finished')
    return (
      <SessionComplete
        passed={stats.passed}
        completed={stats.completed}
        streak={finishInfo.streak}
        goalClosed={finishInfo.goalClosed}
      />
    );

  const busy = step === 'counting' || step === 'recording' || step === 'predicting';

  return (
    <main className="page stack">
      <header className="page-header">
        <h1 className="page-title">Practice</h1>
        <span className="pill pill-soft">
          {deckIndex + 1} / {deck.length}
        </span>
        <Link to="/" className="btn-ghost" style={{ marginLeft: 'auto', padding: '6px 12px', borderRadius: 999 }}>
          ← Back
        </Link>
      </header>

      {current && (
        <div className="prompt">
          <div className="prompt-label">Sign this</div>
          <div className="prompt-word">{current.label}</div>
          {attemptNumber > 1 && <div className="prompt-attempt">Attempt {attemptNumber} of 3</div>}
        </div>
      )}

      {cam === 'requesting' && <p className="muted">Requesting camera access…</p>}
      {cam === 'error' && camError && (
        <div className="error-card">
          <p>Camera unavailable</p>
          <p className="spacer-top">{camError.message}</p>
          <button type="button" className="btn-primary spacer-top" onClick={() => void start()}>
            Try again
          </button>
        </div>
      )}

      <div className="video-wrap" style={{ display: cam === 'ready' ? 'block' : 'none' }}>
        <video ref={videoRef} muted playsInline />
        <div aria-hidden className="frame-guide" />
        {step === 'counting' && (
          <div className="overlay-center">
            <span className="countdown">{countdown}</span>
          </div>
        )}
        {step === 'recording' && (
          <div className="rec-badge">
            <span className="rec-dot" /> Recording…
          </div>
        )}
        <div
          className="lighting-chip"
          style={{ color: brightness ? BRIGHTNESS_COPY[brightness].color : '#555' }}
        >
          {brightness ? BRIGHTNESS_COPY[brightness].text : 'Checking lighting…'}
        </div>
      </div>

      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {cam === 'ready' && step !== 'result' && (
        <div className="btn-row">
          <button type="button" className="btn-primary" onClick={() => void runAttempt()} disabled={busy}>
            {step === 'predicting' ? 'Recognizing…' : busy ? 'Capturing…' : `▶ Record ${current?.label} (~3s)`}
          </button>
          <button type="button" className="btn-ghost" onClick={() => advance(false)} disabled={busy}>
            Skip →
          </button>
        </div>
      )}

      {step === 'result' && result && (
        <ResultCard result={result} label={current?.label ?? ''} model={selected.label} onRetry={retry} onAdvance={() => advance(result.decision.pass)} />
      )}

      {cam === 'ready' && <p className="muted" style={{ fontSize: 12 }}>Recognizer: {selected.label}</p>}
    </main>
  );
}

function ResultCard({
  result,
  label,
  model,
  onRetry,
  onAdvance,
}: {
  result: AttemptResult;
  label: string;
  model: string;
  onRetry: () => void;
  onAdvance: () => void;
}) {
  const { decision, outcome, hint, top } = result;
  return (
    <div className={`result ${decision.pass ? 'pass' : 'fail'}`}>
      <div className="result-title">{decision.pass ? `✓ Pass — ${label}` : '✗ Not quite'}</div>
      {hint && <p className="hint">💡 {hint}</p>}
      {outcome === 'reveal' && (
        <p className="muted" style={{ fontSize: 13 }}>
          Here’s the reference for <strong>{label}</strong> — (reference clips arrive with the dataset; M6.)
        </p>
      )}

      <details className="spacer-top">
        <summary className="muted" style={{ fontSize: 12, cursor: 'pointer' }}>
          details
        </summary>
        <ol className="muted" style={{ margin: '6px 0 0', paddingLeft: 18, fontSize: 13 }}>
          {top.map((t) => (
            <li key={t.label}>
              {t.label} — {Math.round(t.prob * 100)}%
            </li>
          ))}
        </ol>
      </details>

      <div className="btn-row spacer-top">
        {outcome === 'retry' ? (
          <>
            <button type="button" className="btn-primary" onClick={onRetry}>
              Try again
            </button>
            <button type="button" className="btn-ghost" onClick={onAdvance}>
              Skip →
            </button>
          </>
        ) : (
          <button type="button" className="btn-primary" onClick={onAdvance}>
            Continue →
          </button>
        )}
      </div>

      <p className="note spacer-top">
        ⚠ Stubbed recognizer ({model}) — predictions are placeholder until the model publishes (M7).
      </p>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <main className="centered">{children}</main>;
}
