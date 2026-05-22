# 0001 — Validation strategy for dataset and live-tester evidence

Status: Accepted (2026-05-21); revised (2026-05-22)

## Context

The project originally had **no access to any correct signer**. That made public
datasets (ASL Citizen, WLASL) the only source of verified-correct signs. ASL
Citizen clips cannot be redistributed (MSR license, §D.6).

That constraint has changed: the builder will act as the live tester for the app.
This gives the project a real webcam/capture-path test subject and can surface
camera, timing, framing, latency, and obvious model-integration failures. It does
not automatically turn the live loop into population-level recognition evidence:
a single tester is not representative of ASL learners, and correctness of each
performed sign must be checked against authoritative reference material.

A learning app that recognizes signs has a hard dependency on *correct* signs
existing somewhere. Public datasets remain the primary verified source; the live
tester adds a small, documented pilot distribution, not a replacement for held-out
dataset evaluation.

## Decision

1. **Headline model evidence = held-out datasets.** Report
   in-distribution accuracy (ASL Citizen, signer-held-out test) and
   cross-dataset accuracy (WLASL). These remain the primary comparable numbers.
2. **Demo the full pipeline by replaying held-out dataset clips through the
   app capture path.** This shows the end-to-end system on verified-correct
   signs and avoids depending on the live tester's signing correctness.
3. **Use the builder as a documented live tester.** Record live attempts through
   the actual webcam flow after studying each target sign from authoritative
   references. Report this as a small live-tester check of the app path and
   dataset→webcam gap, not as proof that the system works for real users.
4. **Guard against live-test p-hacking.** Predeclare the target signs, attempt
   counts, lighting/framing conditions, and pass/fail summary before using live
   tester clips as evidence. Re-taking until pass is allowed for practice UX, but
   not for reported model quality.
5. **Threshold calibration** runs on the ASL Citizen validation split
   (signer-held-out), documented as dataset-distribution-calibrated. Live-tester
   clips may be reported separately as a sanity check, but must not be both tuned
   against and reported as an independent result.
6. **Reference clips and hint metadata** are sourced by linking out to
   authoritative ASL dictionaries (Handspeak / Lifeprint-ASLU / Signing
   Savvy), not produced by the builder. (Resolves interface Alignment #2.)

## Consequences

- The honest claim narrows to: "the system works end-to-end and the model
  generalizes across datasets, with a small documented live-tester check," NOT
  "this teaches users ASL." Overselling the latter would be dishonest given no
  representative learner validation.
- The project's strength shifts to **engineering + honest from-scratch ML
  result + candor about limits.**
- The 75-sign vocab stays trained (satisfies PRD count). The builder may test a
  selected subset live, but per-sign live validation is claimed only for signs
  actually attempted under the predeclared protocol.
- A later student/signer pilot remains the path to validating the live loop for
  real users; the builder's live tests are a bridge, not the endpoint.
