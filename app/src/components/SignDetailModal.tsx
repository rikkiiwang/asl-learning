import { useNavigate } from 'react-router-dom';
import type { Sign, SignMastery } from '../lib/types';
import { statusStyle } from '../lib/vocabStatus';
import { timeAgo } from '../lib/time';

/** Detail overlay for a tapped sign, with a deep-link into focused practice. */
export function SignDetailModal({
  sign,
  mastery,
  onClose,
}: {
  sign: Sign;
  mastery: SignMastery | undefined;
  onClose: () => void;
}) {
  const navigate = useNavigate();
  const status = mastery?.mastery_status ?? 'not_started';
  const { color, label: statusLabel } = statusStyle(status);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="prompt-word" style={{ fontSize: 26 }}>
          {sign.label}
        </div>
        {sign.gloss && sign.gloss !== sign.label && <p className="muted">{sign.gloss}</p>}
        <p className="spacer-top">
          <span className="pill" style={{ background: color, color: '#fff' }}>
            {statusLabel}
          </span>
        </p>
        <p className="muted" style={{ fontSize: 13 }}>
          {mastery
            ? `${mastery.total_passes}/${mastery.total_attempts} passed · last ${timeAgo(mastery.last_practiced_at ?? new Date())}`
            : 'Not practiced yet'}
        </p>
        <div className="btn-row spacer-top">
          <button type="button" className="btn-primary" onClick={() => navigate(`/practice?sign=${sign.id}`)}>
            ▶ Practice this
          </button>
          <button type="button" className="btn-ghost" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
