# ASL Learning — Context Glossary

The shared language for this project. Definitions only — no implementation
detail, no decisions log (those live in `architecture.md`, `task.md`, ADRs).
When a term here conflicts with how a doc or conversation uses a word, this
file wins until deliberately changed.

---

## Data distributions

The data the model sees comes from three distinct populations. They are NOT
interchangeable, and a number measured on one does not predict another.

- **In-distribution** — ASL Citizen, evaluated signer-held-out (no signer
  appears in two splits). Measures generalization to *new signers within
  ASL Citizen's recording conditions*. This is the primary reported number.

- **Cross-dataset** — WLASL. A *different* dataset (web/instructional video,
  mostly fluent signers). Measures whether the model survives a distribution
  it never trained on. An honesty check — explicitly **not** "deployment."

- **Pilot** — actual live webcam use outside the source datasets. The builder
  will act as a live tester, so the project can exercise the capture path,
  framing, latency, and dataset→webcam gap with real attempts. This is **not**
  representative learner validation: a single tester does not prove recognition
  for real users, and live-test claims only apply to predeclared signs/attempts
  checked against authoritative references. Public dataset video remains the
  primary verified-correct evidence — see ADR 0001.

## Split roles

Standard ML split semantics — the distinction is load-bearing here.

- **train** — fits the weights.
- **validation** — looked at repeatedly during development: model selection,
  early stopping, hyperparameters, **threshold calibration**. Anything tuned
  *toward* lives here. Stays inside ASL Citizen.
- **test** — touched once, drives zero decisions; the only honest
  generalization number. ASL Citizen test = in-distribution headline;
  WLASL = cross-dataset test (no tuning against it, ever).

A dataset used for tuning cannot also be reported as a test number for the
same examples — that is contamination.

**"Cross" trap — two unrelated terms:**
- **Cross-validation** — k-fold resampling *within one dataset* to get a
  robust *validation* estimate (you tune against it). Useful here as
  *signer-grouped k-fold on ASL Citizen* given thin per-class data.
- **Cross-dataset evaluation** — train on A (ASL Citizen), *test* on a
  different dataset B (WLASL). A held-out test, touched once, never tuned
  against. **This is what WLASL is.** It is NOT cross-validation.

## Other terms

- **Isolated sign** — one sign performed alone, as opposed to *continuous*
  signing (sentences). PRD term. Isolated ≠ static: an isolated sign can be
  dynamic (movement over time).

- **False pass** — model accepts a sign the learner performed incorrectly.
  The trust-killing error the thresholding is biased against (PRD Req 9).
- **False fail** — model rejects a correctly performed sign. The frustration
  error; rises as false-pass tolerance drops.

- **Graded deliverable** — **production-quality *engineering* + an honest
  from-scratch model result, demonstrated end-to-end by replaying held-out
  dataset clips through the app, plus a small documented builder live-tester
  check.** Recognition is validated primarily on datasets (in-distribution +
  cross-dataset); the live-webcam loop is checked with one tester but remains
  unvalidated for representative real users (ADR 0001). Engineering is held to
  production standard (deployed, auth, RLS, privacy, graceful failures). The
  claim is NOT "this teaches users ASL." A live student pilot is a deferred eval
  phase that justifies keeping account infrastructure, nothing more. (Confirmed
  2026-05-21; revised 2026-05-22.)

- **Production-ready** — here means *engineering* quality, NOT recognition
  proven for unseen users. The claim is bounded by what the app must survive
  gracefully: out-of-vocab / no-sign reject path, a declared supported-browser
  matrix, camera-denied handling, best-effort attempt writes, per-account
  isolation (RLS). It does NOT mean validated recognition accuracy for
  strangers. Out-of-vocab / no-sign reject path is **in scope** (required
  for the production claim — without it the model maps non-signs onto the
  nearest of 75 classes). (Confirmed 2026-05-21.)
