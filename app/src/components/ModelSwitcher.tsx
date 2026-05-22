import { useModel } from '../models/ModelProvider';

/**
 * DEV/EVAL ONLY model selector (PRD Req 7). Lets us run the pipeline against a
 * pretrained baseline and compare it to the from-scratch model during testing.
 * Strip this from the graded pilot build.
 */
export function ModelSwitcher() {
  const { models, selected, setSelectedId } = useModel();
  return (
    <div className="dev-switcher">
      <div className="dev-label">Testing · model comparison (dev only)</div>
      <label style={{ display: 'block', marginTop: 8, fontSize: 14 }}>
        Active recognizer{' '}
        <select value={selected.id} onChange={(e) => setSelectedId(e.target.value)}>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      <p className="muted spacer-top" style={{ fontSize: 12 }}>
        {selected.note}
      </p>
    </div>
  );
}
