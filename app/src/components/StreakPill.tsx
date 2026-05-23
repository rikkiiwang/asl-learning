/** Gold streak chip. Renders nothing until the learner has a 1+ day streak. */
export function StreakPill({ days }: { days: number }) {
  if (days < 1) return null;
  return (
    <span className="pill pill-gold">
      🔥 {days} day{days === 1 ? '' : 's'}
    </span>
  );
}
