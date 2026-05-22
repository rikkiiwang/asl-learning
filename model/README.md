# ASL Learning — Model Workstream

From-scratch isolated-sign recognizer for the ASL Learning pilot (75 beginner
signs), trained on Google Colab, exported to ONNX for in-browser inference.

See **`MODEL_WORKSTREAM.md`** for progress + the model↔app interface contract
(shared with the system-design agent). Plan: `../vision-model-plan.md`.

## Layout
```
model/
  src/asl/        # Python package (runs locally or on Colab)
    audit.py          # Phase 0: data audit over ASL Citizen split CSVs
    beginner_vocab.py # candidate ASL-1 vocab pool
  configs/        # training configs (checked in, drive every run)
  notebooks/      # thin Colab wrappers that call src/asl
  artifacts/      # audit reports, manifests, validation reports (caches/weights gitignored)
  data/           # raw dataset + tensor caches (gitignored)
```

## Setup
```bash
pip install -r requirements.txt   # torch, opencv, onnx, etc.
```

## Phase 0 — data audit (run after the dataset is extracted)
```bash
cd model
PYTHONPATH=src python -m asl.audit \
  --data-root "data/ASL_Citizen" \
  --min-videos 25 --min-signers 4 \
  --out artifacts/audit
```
Outputs `artifacts/audit/audit_candidates.csv` and a summary of which beginner
signs clear the per-class bar. The frozen 75-sign vocab is built from this.

## Compute
- Audit + preprocessing: local (Apple M4, MPS).
- Training: Google Colab GPU, from the compact preprocessed subset on Drive.
- Export: PyTorch → ONNX → ONNX Runtime Web.
