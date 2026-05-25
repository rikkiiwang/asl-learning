import { Link } from 'react-router-dom';
import { Confetti } from './Confetti';
import { sessionXp } from '../lib/xp';

/** Celebration screen shown when a practice deck is finished. */
export function SessionComplete({
  passed,
  completed,
  streak,
  goalClosed,
}: {
  passed: number;
  completed: number;
  streak: number;
  goalClosed: boolean;
}) {
  return (
    <main className="page" style={{ textAlign: 'center' }}>
      <Confetti />
      <h1 className="page-title" style={{ fontSize: 28 }}>
        Session complete 🎉
      </h1>
      <p className="spacer-top" style={{ fontSize: 18 }}>
        Passed <strong>{passed}</strong> of <strong>{completed}</strong> words.
      </p>
      <p className="spacer-top" style={{ fontSize: 18 }}>
        <span className="pill pill-soft">+{sessionXp(passed)} XP</span>
      </p>
      {streak >= 1 && (
        <p className="spacer-top">
          <span className="pill pill-gold">
            🔥 {streak} day{streak === 1 ? '' : 's'} streak
          </span>
        </p>
      )}
      {goalClosed && <p className="spacer-top muted">You closed today’s goal ring! 🟣</p>}
      <div className="spacer-top">
        <Link to="/" className="btn btn-primary">
          ← Back to dashboard
        </Link>
      </div>
    </main>
  );
}
