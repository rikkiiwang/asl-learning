import type { Sign, MasteryStatus } from '../lib/types';
import { SignTile } from './SignTile';

/** Responsive grid of every sign, tinted by mastery status. */
export function VocabGrid({
  signs,
  statusById,
  onSelect,
}: {
  signs: Sign[];
  statusById: Map<string, MasteryStatus>;
  onSelect: (sign: Sign) => void;
}) {
  return (
    <div className="sign-grid">
      {signs.map((s) => (
        <SignTile
          key={s.id}
          label={s.label}
          status={statusById.get(s.id) ?? 'not_started'}
          onClick={() => onSelect(s)}
        />
      ))}
    </div>
  );
}
