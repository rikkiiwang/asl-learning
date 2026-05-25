import { useModel } from '../models/ModelProvider';

/**
 * Recognition-model picker. Lists the from-scratch models the user trained
 * (v1 shipped, v2 experimental) so the active recognizer can be switched. The
 * pretrained `baseline` only appears when dev tools are on (PRD Req 7), and the
 * dev hint is shown only in that case.
 */
export function ModelSwitcher() {
  const { models, selected, setSelectedId } = useModel();
  const hasBaseline = models.some((m) => m.kind === 'baseline');
  return (
    <section className="card">
      <div className="card-title">Recognition model 🧠</div>
      <p className="muted" style={{ fontSize: 12, marginTop: 2 }}>
        Two models trained from scratch — switch the one that scores your signs.
      </p>
      <label style={{ display: 'block', marginTop: 10, fontSize: 14, fontWeight: 700 }}>
        Active model{' '}
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
      {selected.stats && (
        <p style={{ fontSize: 12, fontWeight: 700, marginTop: 4 }}>{selected.stats}</p>
      )}
      {hasBaseline && (
        <p className="muted" style={{ fontSize: 11, marginTop: 6 }}>
          Dev only: a pretrained baseline is listed for comparison (excluded from the shipped build).
        </p>
      )}
    </section>
  );
}
