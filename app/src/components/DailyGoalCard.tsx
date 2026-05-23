import { GoalRing } from './GoalRing';
import { goalProgress } from '../lib/dailyGoal';
import { sessionXp } from '../lib/xp';

/** Layout A: today's-goal ring as its own card with copy + a reward chip. */
export function DailyGoalCard({ practicedToday }: { practicedToday: number }) {
  const g = goalProgress(practicedToday);
  return (
    <section className="card">
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <GoalRing done={g.done} target={g.target} />
        <div>
          <div style={{ fontWeight: 800 }}>Today’s goal</div>
          <div className="muted" style={{ fontSize: 13 }}>
            {g.complete
              ? 'Goal complete! 🎉'
              : `Practice ${g.remaining} more sign${g.remaining === 1 ? '' : 's'} to close the ring`}
          </div>
        </div>
        <span className="pill pill-soft" style={{ marginLeft: 'auto' }}>
          +{sessionXp(g.target)} XP
        </span>
      </div>
    </section>
  );
}
