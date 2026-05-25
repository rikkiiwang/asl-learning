/** Apple-Watch-style progress ring. `done/target` shown in the center; closes as done→target. */
export function GoalRing({ done, target, size = 84 }: { done: number; target: number; size?: number }) {
  const stroke = 9;
  const r = size / 2 - stroke / 2 - 1;
  const c = 2 * Math.PI * r;
  const frac = target > 0 ? Math.min(done / target, 1) : 0;
  const gid = 'goalRingGrad';
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--seg-togo)" strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={`url(#${gid})`}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - frac)}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text
        x="50%"
        y="50%"
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={size * 0.26}
        fontWeight={800}
        fill="var(--text)"
      >
        {done}/{target}
      </text>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="var(--accent)" />
          <stop offset="1" stopColor="var(--accent-2)" />
        </linearGradient>
      </defs>
    </svg>
  );
}
