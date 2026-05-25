import type { MasteryStatus } from '../lib/types';
import { statusStyle } from '../lib/vocabStatus';

/** One sign in the vocabulary map, tinted by mastery status. */
export function SignTile({
  label,
  status,
  onClick,
}: {
  label: string;
  status: MasteryStatus;
  onClick: () => void;
}) {
  const { color, label: statusLabel } = statusStyle(status);
  return (
    <button type="button" className="sign-tile" onClick={onClick} title={statusLabel} style={{ borderColor: color }}>
      <span className="sign-tile-dot" style={{ background: color }} />
      <span className="sign-tile-label">{label}</span>
    </button>
  );
}
